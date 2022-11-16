import warnings
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torchvision
from torch import nn, Tensor
from torchinfo import summary
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models._utils import handle_legacy_interface, _ovewrite_value_param
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights, mobilenet_v3_large
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.backbone_utils import _validate_trainable_layers
from torchvision.models.vgg import VGG, VGG16_Weights, vgg16
from torchvision.models.detection.ssd import SSD, _vgg_extractor, SSD300_VGG16_Weights
from torchvision.models.detection.ssdlite import _normal_init, _mobilenet_extractor, SSDLiteHead, SSDLite320_MobileNet_V3_Large_Weights

class QuantizableSSD(SSD):
    def __init__(self, backbone: nn.Module, anchor_generator: DefaultBoxGenerator, size: Tuple[int, int], num_classes: int, 
            image_mean: Optional[List[float]] = None, image_std: Optional[List[float]] = None, head: Optional[nn.Module] = None, 
            score_thresh: float = 0.01, nms_thresh: float = 0.45, detections_per_img: int = 200, iou_thresh: float = 0.5, 
            topk_candidates: int = 400, positive_fraction: float = 0.25, **kwargs: Any):
        super().__init__(backbone, anchor_generator, size, num_classes, 
                        image_mean, image_std, head, score_thresh, nms_thresh, 
                        detections_per_img, iou_thresh, topk_candidates, positive_fraction, **kwargs)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        images = self.quant(images)
        return self.dequant(super().forward(images, targets))
        
@handle_legacy_interface(
    weights=("pretrained", SSDLite320_MobileNet_V3_Large_Weights.COCO_V1),
    weights_backbone=("pretrained_backbone", MobileNet_V3_Large_Weights.IMAGENET1K_V1),
)
def qssdlite320_mobilenet_v3_large(
    *,
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

    backbone = mobilenet_v3_large(
        weights=weights_backbone, progress=progress, norm_layer=norm_layer, reduced_tail=reduce_tail, **kwargs
    )
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

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

@handle_legacy_interface(
    weights=("pretrained", SSD300_VGG16_Weights.COCO_V1),
    weights_backbone=("pretrained_backbone", VGG16_Weights.IMAGENET1K_FEATURES),
)
def qssd300_vgg16(
    *,
    weights: Optional[SSD300_VGG16_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    weights_backbone: Optional[VGG16_Weights] = VGG16_Weights.IMAGENET1K_FEATURES,
    trainable_backbone_layers: Optional[int] = None,
    **kwargs: Any,
) -> QuantizableSSD:
    """The SSD300 model is based on the `SSD: Single Shot MultiBox Detector
    <https://arxiv.org/abs/1512.02325>`_ paper.

    .. betastatus:: detection module

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes but they will be resized
    to a fixed size before passing it to the backbone.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:

        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows, where ``N`` is the number of detections:

        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each detection
        - scores (Tensor[N]): the scores for each detection

    Example:

        >>> model = torchvision.models.detection.ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 300), torch.rand(3, 500, 400)]
        >>> predictions = model(x)

    Args:
        weights (:class:`~torchvision.models.detection.SSD300_VGG16_Weights`, optional): The pretrained
                weights to use. See
                :class:`~torchvision.models.detection.SSD300_VGG16_Weights`
                below for more details, and possible values. By default, no
                pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr
            Default is True.
        num_classes (int, optional): number of output classes of the model (including the background)
        weights_backbone (:class:`~torchvision.models.VGG16_Weights`, optional): The pretrained weights for the
            backbone
        trainable_backbone_layers (int, optional): number of trainable (not frozen) layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable. If ``None`` is
            passed (the default) this value is set to 4.
        **kwargs: parameters passed to the ``torchvision.models.detection.SSD``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/detection/ssd.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.detection.SSD300_VGG16_Weights
        :members:
    """
    weights = SSD300_VGG16_Weights.verify(weights)
    weights_backbone = VGG16_Weights.verify(weights_backbone)

    if "size" in kwargs:
        warnings.warn("The size of the model is already fixed; ignoring the parameter.")

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param(num_classes, len(weights.meta["categories"]))
    elif num_classes is None:
        num_classes = 91

    trainable_backbone_layers = _validate_trainable_layers(
        weights is not None or weights_backbone is not None, trainable_backbone_layers, 5, 4
    )

    # Use custom backbones more appropriate for SSD
    backbone = vgg16(weights=weights_backbone, progress=progress)
    backbone = _vgg_extractor(backbone, False, trainable_backbone_layers)
    anchor_generator = DefaultBoxGenerator(
        [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
        steps=[8, 16, 32, 64, 100, 300],
    )

    defaults = {
        # Rescale the input in a way compatible to the backbone
        "image_mean": [0.48235, 0.45882, 0.40784],
        "image_std": [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0],  # undo the 0-1 scaling of toTensor
    }
    kwargs: Any = {**defaults, **kwargs}
    model = QuantizableSSD(backbone, anchor_generator, (300, 300), num_classes, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

def get_ssd(device):
    # load the model 
    # model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True,weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    model = torchvision.models.detection.ssd300_vgg16(pretrained=True,weights=SSD300_VGG16_Weights.DEFAULT)
    # load the model onto the computation device
    model = model.eval().to(device)
    return model

def get_qssd(device):
    # qssd = qssdlite320_mobilenet_v3_large(pretrained=True, weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT).to(device)
    qssd = qssd300_vgg16(pretrained=True, weights=SSD300_VGG16_Weights.DEFAULT).to(device)
    qssd.eval()
    return qssd

def get_modules_to_fused(state_keys):
    modules_to_fused = []
    for key in state_keys:
        if "0.weight" in key:
            modules_to_fused.append([key[:-8]+'0', key[:-8]+'1'])
    return modules_to_fused

def get_fused_model(model):
    modules_to_fused = get_modules_to_fused(model.state_dict().keys())
    return torch.quantization.fuse_modules(model, modules_to_fused)

def get_quant_model(fused_model):
    fused_model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    model_fp32_prepared = torch.quantization.prepare(fused_model)
    model_int8 = torch.quantization.convert(model_fp32_prepared)
    return model_int8

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ssd = get_qssd(device)
    print(ssd.state_dict().keys())
    # fssdlite = get_fused_model(ssd)
    # int8ssdlite = get_quant_model(fssdlite)
    # print(int8ssdlite)