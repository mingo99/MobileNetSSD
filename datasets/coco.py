import random
import torch
import torchvision.transforms as transforms
from torchvision.datasets.coco import CocoDetection
from torch.utils.data.dataloader import DataLoader, RandomSampler
from collections import defaultdict

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((320,320))
])

coco_root = "../../../data/coco/"
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
        dataset = CocoDetection(coco_img_train,coco_ann_train,transform=transform)
    else:
        dataset = CocoDetection(coco_img_val,coco_ann_val,transform=transform)
    return DataLoader(dataset,batch_size=batch_size,collate_fn=coco_collate)

def get_coco_calibrate_datasets(batch_size):
    dataset = CocoDetection(coco_img_val,coco_ann_val,transform=transform)
    print(dataset.coco.getAnnIds())
    path = dataset.coco.loadImgs(id)[0]["file_name"]


if __name__ == "__main__":
    dataset = CocoDetection(coco_img_val,coco_ann_val,transform=transform)
    calib_ds_len = 0
    ds_len = 0
    for catId in dataset.coco.getCatIds():
        imgIds = dataset.coco.getImgIds(catIds=catId)
        idx_max = (len(imgIds)+19)//20
        # print(len(imgIds))
        imgIds_part = random.sample(imgIds,idx_max)
        # print(len(imgIds_part))
        ds_len += len(imgIds)
        calib_ds_len += len(imgIds_part)
    print(imgIds)
    print(ds_len)
    print(calib_ds_len)
    # print(len(dataset.ids))