import os.path
import random
import torch
import torchvision.transforms as transforms
from typing import Any, Callable, Optional, Tuple, List
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
# from torchvision.datasets.coco import CocoDetection
from torch.utils.data.dataloader import DataLoader, RandomSampler
from collections import defaultdict
from PIL import Image
from torchvision.datasets.vision import VisionDataset

class CocoDetection(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        image_id = self.coco.loadImgs(id)[0]["id"]
        return Image.open(os.path.join(self.root, path)).convert("RGB"), image_id

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image, image_id = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, image_id

    def __len__(self) -> int:
        return len(self.ids)

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
            ann['bbox'][2] = ann['bbox'][0]+ann['bbox'][2]
            ann['bbox'][3] = ann['bbox'][1]+ann['bbox'][3]
            target['boxes'].append(torch.FloatTensor(ann['bbox']))
            target['labels'].append(ann['category_id'])
        
        if target['boxes']:
            target['boxes'] = torch.stack(target['boxes'])
        else:
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
        target['labels'] = torch.tensor(target['labels'],dtype=torch.long)
        target['image_id'] = torch.tensor(sample[2],dtype=torch.long)
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


def coco_eval(dt_path, iou_type):
    """
    Args:
        iou_type: `segm`, `bbox`, `keypoints`
    """
    cocoGt = COCO(coco_ann_val)
    cocoDt = cocoGt.loadRes(dt_path)
    cocoEval = COCOeval(cocoGt,cocoDt,iou_type)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

if __name__ == "__main__":
    dataset = CocoDetection(coco_img_train,coco_ann_train,transform=transform)
    id = dataset.coco.getAnnIds(imgIds=550395)
    anns = dataset.coco.loadAnns(id)
    print(len(anns))
    for ann in anns:
        print(ann['bbox'])