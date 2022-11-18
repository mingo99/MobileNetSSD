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
from torchvision.models.quantization.utils import _fuse_modules
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation
from torchvision.models.mobilenetv3 import (
    InvertedResidual,
    InvertedResidualConfig,
    MobileNetV3,
    _mobilenet_v3_conf,
    MobileNet_V3_Large_Weights,
)

class QuantizableSqueezeExcitation(SqueezeExcitation):
    _version = 2

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["scale_activation"] = nn.Hardsigmoid
        super().__init__(*args, **kwargs)
        self.skip_mul = nn.quantized.FloatFunctional()

    def forward(self, input: Tensor) -> Tensor:
        return self.skip_mul.mul(self._scale(input), input)

    def fuse_model(self, is_qat: Optional[bool] = None) -> None:
        _fuse_modules(self, ["fc1", "activation"], is_qat, inplace=True)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if hasattr(self, "qconfig") and (version is None or version < 2):
            default_state_dict = {
                "scale_activation.activation_post_process.scale": torch.tensor([1.0]),
                "scale_activation.activation_post_process.activation_post_process.scale": torch.tensor([1.0]),
                "scale_activation.activation_post_process.zero_point": torch.tensor([0], dtype=torch.int32),
                "scale_activation.activation_post_process.activation_post_process.zero_point": torch.tensor(
                    [0], dtype=torch.int32
                ),
                "scale_activation.activation_post_process.fake_quant_enabled": torch.tensor([1]),
                "scale_activation.activation_post_process.observer_enabled": torch.tensor([1]),
            }
            for k, v in default_state_dict.items():
                full_key = prefix + k
                if full_key not in state_dict:
                    state_dict[full_key] = v

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class QuantizableInvertedResidual(InvertedResidual):
    # TODO https://github.com/pytorch/vision/pull/4232#pullrequestreview-730461659
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(se_layer=QuantizableSqueezeExcitation, *args, **kwargs)  # type: ignore[misc]
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return self.skip_add.add(x, self.block(x))
        else:
            return self.block(x)

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

    def fuse_model(self, is_qat: Optional[bool] = None) -> None:
        for m in self.modules():
            if type(m) is Conv2dNormActivation:
                modules_to_fuse = ["0", "1"]
                if len(m) == 3 and type(m[2]) is nn.ReLU:
                    modules_to_fuse.append("2")
                _fuse_modules(m, modules_to_fuse, is_qat, inplace=True)
            elif type(m) is QuantizableSqueezeExcitation:
                m.fuse_model(is_qat)

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
    # print(ssd)
    # print(ssd.state_dict().keys())
    # print(ssd.backbone.features[1])
    # summary(ssd, (1,3,300,300),  device=device)
    # fssdlite = get_fused_model(ssd)
    int8ssd = get_quant_model(ssd)
    # print(int8ssdlite)