import os
import warnings
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torchvision
from torch import nn, Tensor
from torchinfo import summary
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models._utils import handle_legacy_interface, _ovewrite_value_param
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.backbone_utils import _validate_trainable_layers
from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.ssdlite import _normal_init, _mobilenet_extractor, SSDLiteHead, SSDLite320_MobileNet_V3_Large_Weights
from torchvision.models.quantization.utils import _fuse_modules, _replace_relu
from torchvision.ops.misc import Conv2dNormActivation
from torchvision.ops import boxes as box_ops
from torchvision.models.quantization.mobilenetv3 import (
    QuantizableSqueezeExcitation,
    QuantizableMobileNetV3, 
    _mobilenet_v3_model, 
    MobileNet_V3_Large_QuantizedWeights,
)
from torchvision.models.mobilenetv3 import (
    _mobilenet_v3_conf,
    MobileNet_V3_Large_Weights,
)

from datasets import get_coco_datasets

class QuantizableSSD(SSD):
    """
    Class for Quantizable SSD detector, inherit from SSD.

    Add the QuantStub and DeQuantStub module and modify `forward` function.
    """
    def __init__(self, backbone: nn.Module, anchor_generator: DefaultBoxGenerator, size: Tuple[int, int], num_classes: int, 
            image_mean: Optional[List[float]] = None, image_std: Optional[List[float]] = None, head: Optional[nn.Module] = None, 
            score_thresh: float = 0.01, nms_thresh: float = 0.45, detections_per_img: int = 200, iou_thresh: float = 0.5, 
            topk_candidates: int = 400, positive_fraction: float = 0.25, **kwargs: Any):
        super().__init__(backbone, anchor_generator, size, num_classes, 
                        image_mean, image_std, head, score_thresh, nms_thresh, 
                        detections_per_img, iou_thresh, topk_candidates, positive_fraction, **kwargs)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(
        self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        # get the original image sizes
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        if not self.training:
            images.tensors = self.quant(images.tensors)
        # get the features from the backbone
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        features = list(features.values())

        # compute the ssd heads outputs using the features
        head_outputs = self.head(features)

        # dequanitze features and head outputs
        for i, feature in enumerate(features):
            features[i] = self.dequant(feature)

        head_outputs['bbox_regression'] = self.dequant(head_outputs['bbox_regression'])
        head_outputs['cls_logits'] = self.dequant(head_outputs['cls_logits'])

        # create the set of anchors
        anchors = self.anchor_generator(images, features)

        losses = {}
        detections: List[Dict[str, Tensor]] = []
        if self.training:
            matched_idxs = []
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for anchors_per_image, targets_per_image in zip(anchors, targets):
                    if targets_per_image["boxes"].numel() == 0:
                        matched_idxs.append(
                            torch.full(
                                (anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device
                            )
                        )
                        continue

                    match_quality_matrix = box_ops.box_iou(targets_per_image["boxes"], anchors_per_image)
                    matched_idxs.append(self.proposal_matcher(match_quality_matrix))

                losses = self.compute_loss(targets, head_outputs, anchors, matched_idxs)
        else:
            detections = self.postprocess_detections(head_outputs, anchors, images.image_sizes)
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("SSD always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        return self.eager_outputs(losses, detections)

    def fuse_model(self, is_qat: Optional[bool] = None) -> None:
        for m in self.modules():
            if type(m) is Conv2dNormActivation:
                modules_to_fuse = ["0", "1"]
                if len(m) == 3 and type(m[2]) is nn.ReLU:
                    modules_to_fuse.append("2")
                _fuse_modules(m, modules_to_fuse, is_qat, inplace=True)
            elif type(m) is QuantizableSqueezeExcitation:
                m.fuse_model(is_qat)


def quantize_model(model: nn.Module, backend: str, calibrate: bool=False) -> None:
    """
    Quantize SSDLite model from `float32` to `int8`.

    Args:
        model(nn.Module): float32 model
        backend(str): a string representing the target backend. Currently supports `fbgemm`, `qnnpack` and `onednn`.
        calibrate(bool): select whether to use the dataset for calibration
    """
    if backend not in torch.backends.quantized.supported_engines:
        raise RuntimeError("Quantized backend not supported ")
    torch.backends.quantized.engine = backend
    model.eval()
    # Make sure that weight qconfig matches that of the serialized models
    if backend == "fbgemm":
        model.qconfig = torch.ao.quantization.QConfig(  # type: ignore[assignment]
            activation=torch.ao.quantization.default_observer,
            weight=torch.ao.quantization.default_per_channel_weight_observer,
        )
    elif backend == "qnnpack":
        model.qconfig = torch.ao.quantization.QConfig(  # type: ignore[assignment]
            activation=torch.ao.quantization.default_observer, 
            weight=torch.ao.quantization.default_weight_observer
        )
    model.fuse_model()  # type: ignore[operator]
    torch.ao.quantization.prepare(model, inplace=True)
    if calibrate:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Calibrate is enable, open {device} as computation device.")
        model = model.to(device)
        for epoch in range(100):
            os.makedirs(f"./weights/epoch{epoch}")
            for i, data in enumerate(get_coco_datasets(128)):
                print(f"Epoch:{epoch} | Batch:{i}")
                # print(data[0].shape)
                image = data[0].to(device)
                model(image)
            model = model.to('cpu')
            torch.save(model.state_dict,f"./weights/epoch{epoch}/ssdlite320_mobilenet_v3_large_float32.pth")
            torch.ao.quantization.convert(model, inplace=True)
            torch.save(model.state_dict,f"./weights/epoch{epoch}/ssdlite320_mobilenet_v3_large_int8.pth")
    else:
        _dummy_input_data = torch.rand(1, 3, 320, 320)
        model(_dummy_input_data)
        torch.ao.quantization.convert(model, inplace=True)


@handle_legacy_interface(
    weights=(
        "pretrained",
        lambda kwargs: MobileNet_V3_Large_QuantizedWeights.IMAGENET1K_QNNPACK_V1
        if kwargs.get("quantize", False)
        else MobileNet_V3_Large_Weights.IMAGENET1K_V1,
    )
)
def mobilenet_v3_large(
    *,
    weights: Optional[Union[MobileNet_V3_Large_QuantizedWeights, MobileNet_V3_Large_Weights]] = None,
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
) -> QuantizableMobileNetV3:
    weights = (MobileNet_V3_Large_QuantizedWeights if quantize else MobileNet_V3_Large_Weights).verify(weights)
    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_large", reduced_tail=True, **kwargs)
    return _mobilenet_v3_model(inverted_residual_setting, last_channel, weights, progress, quantize, **kwargs)


@handle_legacy_interface(
    weights=("pretrained", SSDLite320_MobileNet_V3_Large_Weights.COCO_V1),
    weights_backbone=("pretrained_backbone", MobileNet_V3_Large_Weights.IMAGENET1K_V1),
)
def qssdlite320_mobilenet_v3_large(
    *,
    quantize: bool=False,
    weights: Optional[SSDLite320_MobileNet_V3_Large_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    weights_backbone: Optional[MobileNet_V3_Large_Weights] = MobileNet_V3_Large_Weights.IMAGENET1K_V1,
    trainable_backbone_layers: Optional[int] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
    **kwargs: Any,
) -> QuantizableSSD:

    weights = SSDLite320_MobileNet_V3_Large_Weights.verify(weights)
    weights_backbone = MobileNet_V3_Large_Weights.verify(weights_backbone)

    if "size" in kwargs:
        warnings.warn("The size of the model is already fixed; ignoring the parameter.")

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param(num_classes, len(weights.meta["categories"]))
    elif num_classes is None:
        num_classes = 91

    trainable_backbone_layers = _validate_trainable_layers(
        weights is not None or weights_backbone is not None, trainable_backbone_layers, 6, 6
    )

    # Enable reduced tail if no pretrained backbone is selected. See Table 6 of MobileNetV3 paper.
    reduce_tail = weights_backbone is None

    if norm_layer is None:
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

    backbone = mobilenet_v3_large()
    if weights_backbone is None:
        # Change the default initialization scheme if not pretrained
        _normal_init(backbone)
    backbone = _mobilenet_extractor(
        backbone,
        trainable_backbone_layers,
        norm_layer,
    )

    size = (320, 320)
    anchor_generator = DefaultBoxGenerator([[2, 3] for _ in range(6)], min_ratio=0.2, max_ratio=0.95)
    out_channels = det_utils.retrieve_out_channels(backbone, size)
    num_anchors = anchor_generator.num_anchors_per_location()
    if len(out_channels) != len(anchor_generator.aspect_ratios):
        raise ValueError(
            f"The length of the output channels from the backbone {len(out_channels)} do not match the length of the anchor generator aspect ratios {len(anchor_generator.aspect_ratios)}"
        )

    defaults = {
        "score_thresh": 0.001,
        "nms_thresh": 0.55,
        "detections_per_img": 300,
        "topk_candidates": 300,
        # Rescale the input in a way compatible to the backbone:
        # The following mean/std rescale the data from [0, 1] to [-1, 1]
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    }
    kwargs: Any = {**defaults, **kwargs}
    model = QuantizableSSD(
        backbone,
        anchor_generator,
        size,
        num_classes,
        head=SSDLiteHead(out_channels, num_anchors, num_classes, norm_layer),
        **kwargs,
    )
    _replace_relu(model)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    if quantize:
        quantize_model(model,'fbgemm',calibrate=True)

    return model

def get_model(device):
    """
    Get the SSDLite320_MobileNet_V3_Large model.
    """
    # load the model 
    weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True,weights=weights)
    # load the model onto the computation device
    model = model.eval().to(device)
    return model

def get_quant_model(device):
    """
    Get the quantizable SSDLite320_MobileNet_V3_Large model.
    """
    weights = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = qssdlite320_mobilenet_v3_large(pretrained=True,weights=weights,quantize=True)
    return model.eval().to(device)