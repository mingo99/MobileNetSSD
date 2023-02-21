from datasets import CocoDetection
from detection import draw_boxes
from datasets import COCO_INSTANCE_CATEGORY_NAMES as coco_names
from model import get_model

import torchvision.transforms as transforms
import numpy as np
import cv2
import json

COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((320,320))
])

coco_root = "../../data/coco/"
coco_img_train = coco_root+"images/train2014/"
coco_img_val = coco_root+"images/val2014/"
coco_ann_train = coco_root+"annotations/instances_train2014.json"
coco_ann_val = coco_root+"annotations/instances_val2014.json"

if __name__ == '__main__':
    dataset = json.load(open(coco_ann_val, 'r'))
    anns = []
    for ann in dataset['annotations']:
        anns.append({
            "image_id": ann['image_id'],
            "bbox": ann['bbox'],
            "category_id": ann['category_id'],
            "id": ann['id']
        })
    anns.sort(key=lambda ann: ann['image_id'])
    with open("./coco_val_anns.json", 'w') as f:
        json.dump(anns, f, indent=4)