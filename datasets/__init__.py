from .coco import get_coco_datasets, get_coco_calibrate_datasets, coco_eval
from .coco_names import COCO_INSTANCE_CATEGORY_NAMES

__all__ = [
    "get_coco_datasets",
    "get_coco_calibrate_datasets",
    "coco_eval",
    "COCO_INSTANCE_CATEGORY_NAMES"
]