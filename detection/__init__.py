from .detect import detect_image, detect_video
from ._utils import predict, draw_boxes

__all__ = [
    "detect_image",
    "detect_video",
    "draw_boxes",
    "predict"
]