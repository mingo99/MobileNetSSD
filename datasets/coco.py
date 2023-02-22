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

class CocoDataset():
    """ 
    Get the images and annotations of COCO dataset.
    Params:
        dataset_name: `coco` or `cocofb`
        version: `2014` and `2017` of coco, `2023` of cocofb
        set_name: `train` or `val`
    """
    def __init__(self, root:str, version: int, set_name: str) -> None:
        self.coco_root = root
        self.version = version
        self.coco_img_path = self.coco_root+f"images/{set_name}{self.version}/"
        self.coco_ann_path = self.coco_root+f"annotations/instances_{set_name}{self.version}.json"

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((320,320))
        ])
        self.dataset = CocoDetection(self.coco_img_path,self.coco_ann_path,transform=self.transform)
        self.coco = self.dataset.coco

    def coco_collate(self, batch):
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
            target = {'boxes':[],'labels':[]}
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

    def get_coco_dataset(self) -> CocoDetection:
        return self.dataset

    def get_coco_dataloader(self, batch_size) -> DataLoader:
        return DataLoader(self.dataset,batch_size=batch_size,collate_fn=self.coco_collate, drop_last=True)

    def get_coco_calibrate_dataloader(self, batch_size: Optional[int] = 1) -> DataLoader:
        """
        Select 5% of validation set as calibration set for quantization.
        The argument `set_name` must be `val`.
        """
        # random.seed(12345)
        calibImgIds = []
        for catId in self.coco.getCatIds():
            imgIds = self.coco.getImgIds(catIds=catId)
            slice_len = (len(imgIds)+19)//20
            imgIds_calib = random.sample(imgIds,slice_len)
            calibImgIds.extend(imgIds_calib)
        calibImgIds = list(set(calibImgIds))
        calibImgIds = sorted(calibImgIds)
        self.dataset.ids = calibImgIds
        print(f"Batch size: {batch_size}")
        return DataLoader(self.dataset,batch_size=batch_size,collate_fn=self.coco_collate, drop_last=True)

    def coco_eval(self, dt_path, iou_type):
        """
        Args:
            iou_type: `segm`, `bbox`, `keypoints`
        """
        cocoGt = COCO(self.coco_ann_path)
        cocoDt = cocoGt.loadRes(dt_path)
        cocoEval = COCOeval(cocoGt,cocoDt,iou_type)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

if __name__ == "__main__":
    dataset = CocoDataset('../../data/cocofb/', 2023, 'train')
    data = dataset.get_coco_dataloader(16)
    for i in data:
        print(type(i[1]))
        break
    # id = dataset.coco.getAnnIds(imgIds=262145)
    # anns = dataset.coco.loadAnns(id)
    # print(len(anns))
    # for ann in anns:
    #     print(ann['bbox'])