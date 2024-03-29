import os
import warnings
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union
from torchvision.ops.misc import Conv2dNormActivation
from torchvision.ops import boxes as box_ops
from torchvision.models._utils import handle_legacy_interface
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.ssd import SSD
from torchvision.models.quantization.utils import _fuse_modules
from torchvision.models.quantization.mobilenetv3 import (
    QuantizableSqueezeExcitation,
    QuantizableMobileNetV3, 
    _mobilenet_v3_model, 
    MobileNet_V3_Large_QuantizedWeights,
)
from torchvision.models.mobilenetv3 import (
    _mobilenet_v3_conf,
    MobileNet_V3_Large_Weights
)

from datasets import get_dataloader

class CustomSSD(SSD):
    def compute_loss(
        self,
        targets: List[Dict[str, Tensor]],
        head_outputs: Dict[str, Tensor],
        anchors: List[Tensor],
        matched_idxs: List[Tensor],
    ) -> Dict[str, Tensor]:
        bbox_regression = head_outputs["bbox_regression"]
        cls_logits = head_outputs["cls_logits"]

        # Match original targets with default boxes
        num_foreground = 0
        bbox_loss = []
        cls_targets = []
        for (
            targets_per_image,
            bbox_regression_per_image,
            cls_logits_per_image,
            anchors_per_image,
            matched_idxs_per_image,
        ) in zip(targets, bbox_regression, cls_logits, anchors, matched_idxs):
            # produce the matching between boxes and targets
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            foreground_matched_idxs_per_image = matched_idxs_per_image[foreground_idxs_per_image]
            num_foreground += foreground_matched_idxs_per_image.numel()

            # Calculate regression loss
            matched_gt_boxes_per_image = targets_per_image["boxes"][foreground_matched_idxs_per_image]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]
            target_regression = self.box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
            bbox_loss.append(
                torch.nn.functional.smooth_l1_loss(bbox_regression_per_image, target_regression, reduction="sum")
            )

            # Estimate ground truth for class targets
            gt_classes_target = torch.zeros(
                (cls_logits_per_image.size(0),),
                dtype=targets_per_image["labels"].dtype,
                device=targets_per_image["labels"].device,
            )
            gt_classes_target[foreground_idxs_per_image] = targets_per_image["labels"][
                foreground_matched_idxs_per_image
            ]
            cls_targets.append(gt_classes_target)

        bbox_loss = torch.stack(bbox_loss)
        cls_targets = torch.stack(cls_targets)

        # Calculate classification loss
        num_classes = cls_logits.size(-1)
        class_weights = torch.FloatTensor([0.1, 1, 2]).to(cls_logits.device)
        cls_loss = F.cross_entropy(cls_logits.view(-1, num_classes), cls_targets.view(-1), reduction="none", weight=class_weights).view(
            cls_targets.size()
        )

        # Hard Negative Sampling
        foreground_idxs = cls_targets > 0
        num_negative = self.neg_to_pos_ratio * foreground_idxs.sum(1, keepdim=True)
        # num_negative[num_negative < self.neg_to_pos_ratio] = self.neg_to_pos_ratio
        negative_loss = cls_loss.clone()
        negative_loss[foreground_idxs] = -float("inf")  # use -inf to detect positive values that creeped in the sample
        values, idx = negative_loss.sort(1, descending=True)
        # background_idxs = torch.logical_and(idx.sort(1)[1] < num_negative, torch.isfinite(values))
        background_idxs = idx.sort(1)[1] < num_negative

        N = max(1, num_foreground)
        return {
            "bbox_regression": bbox_loss.sum() / N,
            "classification": (cls_loss[foreground_idxs].sum() + cls_loss[background_idxs].sum()) / N,
        }

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

    def compute_loss(
        self,
        targets: List[Dict[str, Tensor]],
        head_outputs: Dict[str, Tensor],
        anchors: List[Tensor],
        matched_idxs: List[Tensor],
    ) -> Dict[str, Tensor]:
        bbox_regression = head_outputs["bbox_regression"]
        cls_logits = head_outputs["cls_logits"]

        # Match original targets with default boxes
        num_foreground = 0
        bbox_loss = []
        cls_targets = []
        for (
            targets_per_image,
            bbox_regression_per_image,
            cls_logits_per_image,
            anchors_per_image,
            matched_idxs_per_image,
        ) in zip(targets, bbox_regression, cls_logits, anchors, matched_idxs):
            # produce the matching between boxes and targets
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            foreground_matched_idxs_per_image = matched_idxs_per_image[foreground_idxs_per_image]
            num_foreground += foreground_matched_idxs_per_image.numel()

            # Calculate regression loss
            matched_gt_boxes_per_image = targets_per_image["boxes"][foreground_matched_idxs_per_image]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]
            target_regression = self.box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
            bbox_loss.append(
                torch.nn.functional.smooth_l1_loss(bbox_regression_per_image, target_regression, reduction="sum")
            )

            # Estimate ground truth for class targets
            gt_classes_target = torch.zeros(
                (cls_logits_per_image.size(0),),
                dtype=targets_per_image["labels"].dtype,
                device=targets_per_image["labels"].device,
            )
            gt_classes_target[foreground_idxs_per_image] = targets_per_image["labels"][
                foreground_matched_idxs_per_image
            ]
            cls_targets.append(gt_classes_target)

        bbox_loss = torch.stack(bbox_loss)
        cls_targets = torch.stack(cls_targets)

        # Calculate classification loss
        num_classes = cls_logits.size(-1)
        cls_loss = F.cross_entropy(cls_logits.view(-1, num_classes), cls_targets.view(-1), reduction="none").view(
            cls_targets.size()
        )

        # Hard Negative Sampling
        foreground_idxs = cls_targets > 0
        num_negative = self.neg_to_pos_ratio * foreground_idxs.sum(1, keepdim=True)
        # num_negative[num_negative < self.neg_to_pos_ratio] = self.neg_to_pos_ratio
        negative_loss = cls_loss.clone()
        negative_loss[foreground_idxs] = -float("inf")  # use -inf to detect positive values that creeped in the sample
        values, idx = negative_loss.sort(1, descending=True)
        # background_idxs = torch.logical_and(idx.sort(1)[1] < num_negative, torch.isfinite(values))
        background_idxs = idx.sort(1)[1] < num_negative

        # N = max(1, num_foreground)
        # return {
        #     "bbox_regression": bbox_loss.sum() / N,
        #     "classification": (cls_loss[foreground_idxs].sum() + cls_loss[background_idxs].sum()) / N,
        # }
        # 22/12/20 modified
        N = max(0, num_foreground)
        loss_b = 0.0 if N==0 else bbox_loss.sum() / N
        loss_c = 0.0 if N==0 else (cls_loss[foreground_idxs].sum() + cls_loss[background_idxs].sum()) / N
        return {
            "bbox_regression": loss_b,
            "classification": loss_c,
        }

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


def quantize_model(
        model: nn.Module, 
        backend: str,
        path: str="./weights/ssdlite320_mobilenet_v3_large_calibrated_model_pre.pth",
        calibrate: bool=False, 
        batch_size: Optional[int]=1, 
        epochs: Optional[int]=1
    ) -> None:
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
        model.qconfig = torch.quantization.QConfig(  # type: ignore[assignment]
            activation=torch.quantization.default_observer,
            weight=torch.quantization.default_per_channel_weight_observer,
        )
    elif backend == "qnnpack":
        model.qconfig = torch.quantization.QConfig(  # type: ignore[assignment]
            activation=torch.quantization.default_observer, 
            weight=torch.quantization.default_weight_observer
        )
    model.fuse_model()  # type: ignore[operator]
    torch.quantization.prepare(model, inplace=True)
    if calibrate:
        device = torch.device('cpu')
        print(f"Calibrate is enable, open {device} as computation device.")
        print("Calibrating...")
        for epoch in range(epochs):
            dir = f"./weights/epoch{epoch}"
            if not os.path.exists(dir):
                os.makedirs(dir)
            model.to(device)
            _, data_loader, _ = get_dataloader("../../data/coco/", 1, False)
            for i, data in enumerate(data_loader):
                print(f"Epoch: {epoch} | Iter: {i}")
                image = data[0].to(device)
                model(image)
            torch.save(model.state_dict(),f"./weights/epoch{epoch}/ssdlite320_mobilenet_v3_large_calibrated_model_pre.pth")
        torch.save(model.state_dict(),path)
        print("Calibrate done.")
    else:
        for i in range(1024):
            print(f"The {i}th calibration.")
            _dummy_input_data = torch.rand(1, 3, 320, 320)
            model(_dummy_input_data)
        torch.save(model.state_dict(),"./weights/ssdlite320_mobilenet_v3_large_calibrated_model_pre_dummy.pth")
    torch.quantization.convert(model, inplace=True)


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