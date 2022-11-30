import random
import torch
import torchvision.transforms as transforms
from typing import Optional
from torchvision.datasets.coco import CocoDetection
from torch.utils.data.dataloader import DataLoader, RandomSampler
from collections import defaultdict

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((320,320))
])

coco_root = "../../data/coco/"
coco_img_train = coco_root+"images/train2014/"
coco_img_val = coco_root+"images/val2014/"
coco_ann_train = coco_root+"annotations/instances_train2014.json"
coco_ann_val = coco_root+"annotations/instances_val2014.json"

def coco_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch(Tuple): A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            images(Tensor): batch of images stacked on their 0 dim
            targets(List[Dict[str, Tensor]]): annotations for a given image are stacked on 0 dim
    """
    targets = []
    images = []
    for sample in batch:
        images.append(sample[0])
        target = defaultdict(list)
        for ann in sample[1]:
            target['boxes'].append(torch.FloatTensor(ann['bbox']))
            target['labels'].append(ann['category_id'])
        if target['boxes']:
            target['boxes'] = torch.stack(target['boxes'])
        target['labels'] = torch.tensor(target['labels'],dtype=torch.long)
        targets.append(target)
    return torch.stack(images, 0), targets

def get_coco_datasets(batch_size,train=True):
    if train:
        print("Loading train set...")
        dataset = CocoDetection(coco_img_train,coco_ann_train,transform=transform)
    else:
        print("Loading validation set...")
        dataset = CocoDetection(coco_img_val,coco_ann_val,transform=transform)
    return DataLoader(dataset,batch_size=batch_size,collate_fn=coco_collate)

def get_coco_calibrate_datasets(batch_size: Optional[int] = 1):
    """
    Select 5% of validation set as calibration set for quantization.
    """
    # random.seed(12345)
    dataset = CocoDetection(coco_img_val,coco_ann_val,transform=transform)
    calibImgIds = []
    for catId in dataset.coco.getCatIds():
        imgIds = dataset.coco.getImgIds(catIds=catId)
        slice_len = (len(imgIds)+19)//20
        imgIds_calib = random.sample(imgIds,slice_len)
        calibImgIds.extend(imgIds_calib)
    calibImgIds = list(set(calibImgIds))
    calibImgIds = sorted(calibImgIds)
    dataset.ids = calibImgIds
    print(f"Batch size: {batch_size}")
    return DataLoader(dataset,batch_size=batch_size,collate_fn=coco_collate)


if __name__ == "__main__":
    calib_data = get_coco_calibrate_datasets(64)
    print(len(calib_data))
    for i, data in enumerate(calib_data):
        print(data[0].shape)
        print(i)