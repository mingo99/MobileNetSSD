import warnings
import torchvision
from functools import partial
from typing import Any, Callable, Optional
from torch import nn
from torchvision.models._utils import handle_legacy_interface, _ovewrite_value_param
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.backbone_utils import _validate_trainable_layers
from torchvision.models.detection.ssdlite import(
    _normal_init, 
    _mobilenet_extractor, 
    SSDLiteHead, 
    SSDLite320_MobileNet_V3_Large_Weights
)
from torchvision.models.quantization.utils import _fuse_modules, _replace_relu
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights

from ._utils import *

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
    # Load weights
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
    # Quantize model
    if quantize:
        quantize_model(model,'fbgemm')
    return model


def get_model(device):
    """
    Get the SSDLite320_MobileNet_V3_Large model.
    """
    # load the model 
    weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True,weights=weights)
    # load the model onto the computation device
    return model.eval().to(device)

def get_quant_model(device):
    """
    Get the quantizable SSDLite320_MobileNet_V3_Large model.
    """
    # load the model 
    weights = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = qssdlite320_mobilenet_v3_large(pretrained=True,weights=weights,quantize=True)
    # load the model onto the computation device
    return model.eval().to(device)

def ssdlite_with_quant_weights(path):
    model = get_quant_model('cpu')
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    return model