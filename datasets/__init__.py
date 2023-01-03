from .coco import CocoDetection, get_coco_dataloader, get_coco_calibrate_dataloader, coco_eval
from .coco_names import COCO_INSTANCE_CATEGORY_NAMES

__all__ = [
    "CocoDetection",
    "get_coco_dataloader",
    "get_coco_calibrate_dataloader",
    "coco_eval",
    "COCO_INSTANCE_CATEGORY_NAMES"
]