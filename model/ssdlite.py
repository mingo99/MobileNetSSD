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
from torchvision.models.detection.ssd import SSD

from .utils import *

@handle_legacy_interface(
    weights=("pretrained", SSDLite320_MobileNet_V3_Large_Weights.COCO_V1),
    weights_backbone=("pretrained_backbone", MobileNet_V3_Large_Weights.IMAGENET1K_V1),
)
def ssdlite320_mobilenet_v3_large(
    *,
    weights: Optional[SSDLite320_MobileNet_V3_Large_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    weights_backbone: Optional[MobileNet_V3_Large_Weights] = MobileNet_V3_Large_Weights.IMAGENET1K_V1,
    trainable_backbone_layers: Optional[int] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
    **kwargs: Any,
) -> SSD:
    """SSDlite model architecture with input size 320x320 and a MobileNetV3 Large backbone, as
    described at `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`__ and
    `MobileNetV2: Inverted Residuals and Linear Bottlenecks <https://arxiv.org/abs/1801.04381>`__.

    .. betastatus:: detection module

    See :func:`~torchvision.models.detection.ssd300_vgg16` for more details.

    Example:

        >>> model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
        >>> model.eval()
        >>> x = [torch.rand(3, 320, 320), torch.rand(3, 500, 400)]
        >>> predictions = model(x)

    Args:
        weights (:class:`~torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model
            (including the background).
        weights_backbone (:class:`~torchvision.models.MobileNet_V3_Large_Weights`, optional): The pretrained
            weights for the backbone.
        trainable_backbone_layers (int, optional): number of trainable (not frozen) layers
            starting from final block. Valid values are between 0 and 6, with 6 meaning all
            backbone layers are trainable. If ``None`` is passed (the default) this value is
            set to 6.
        norm_layer (callable, optional): Module specifying the normalization layer to use.
        **kwargs: parameters passed to the ``torchvision.models.detection.ssd.SSD``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/detection/ssd.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights
        :members:
    """

    weights = SSDLite320_MobileNet_V3_Large_Weights.verify(weights)
    weights_backbone = MobileNet_V3_Large_Weights.verify(weights_backbone)

    if "size" in kwargs:
        warnings.warn("The size of the model is already fixed; ignoring the parameter.")

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param("num_classes", num_classes, len(weights.meta["categories"]))
    elif num_classes is None:
        num_classes = 91

    trainable_backbone_layers = _validate_trainable_layers(
        weights is not None or weights_backbone is not None, trainable_backbone_layers, 6, 6
    )

    # Enable reduced tail if no pretrained backbone is selected. See Table 6 of MobileNetV3 paper.
    reduce_tail = weights_backbone is None

    if norm_layer is None:
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

    backbone = torchvision.models.mobilenet.mobilenet_v3_large(
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
    anchor_generator = DefaultBoxGenerator(
        # [[1.7, 1.5, 1.9],[1.9, 1.7, 2.1],[1.8, 1.6, 2.0],[2.2, 2.0, 2.4],[2.5,2.3, 2.7],[1.7, 1.5, 1.9]], 
        # # scales=[0.03, 0.05, 0.08, 0.15, 0.3, 0.6, 0.9]
        # scales=[0.05, 0.08, 0.15, 0.3, 0.6, 0.9, 1.05]
        [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
    )
    out_channels = det_utils.retrieve_out_channels(backbone, size)
    num_anchors = anchor_generator.num_anchors_per_location()
    if len(out_channels) != len(anchor_generator.aspect_ratios):
        raise ValueError(
            f"The length of the output channels from the backbone {len(out_channels)} do not match the length of the anchor generator aspect ratios {len(anchor_generator.aspect_ratios)}"
        )

    defaults = {
        "score_thresh": 0.001,
        "nms_thresh": 0.50,
        "detections_per_img": 300,
        "topk_candidates": 300,
        # Rescale the input in a way compatible to the backbone:
        # The following mean/std rescale the data from [0, 1] to [-1, 1]
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    }
    kwargs: Any = {**defaults, **kwargs}
    model = SSD(
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

def get_model(device, pretrained: bool=False) -> SSD:
    """
    Get the SSDLite320_MobileNet_V3_Large model.
    """
    # load the model 
    weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    if pretrained:
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights)
        # model = ssdlite320_mobilenet_v3_large(weights=weights)
    else:
        # model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights_backbone=MobileNet_V3_Large_Weights.DEFAULT,num_classes=3)
        # model = torchvision.models.detection.ssd300_vgg16(weights_backbone=torchvision.models.detection.SSD300_VGG16_Weights,num_classes=3)
        model = ssdlite320_mobilenet_v3_large(weights_backbone=MobileNet_V3_Large_Weights.DEFAULT,num_classes=3)
    # if torch.cuda.is_available():
    #     if torch.cuda.device_count() > 1:
    #         print("Let's use", torch.cuda.device_count(), "GPUs!")
    #         # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #         model = nn.DataParallel(model)
    # load the model onto the computation device
    return model.to(device)

def ssdlite_with_weights(path, device, custom=True):
    if custom:
        model = get_model(device,False)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model"])
    else:
        model = get_model(device, True)
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
    model.eval()
    return model