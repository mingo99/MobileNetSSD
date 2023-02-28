import torch
from torch.utils.data.dataloader import DataLoader, RandomSampler, SequentialSampler, BatchSampler
from torch.utils.data.distributed import DistributedSampler

from . import transforms as T
from .coco_utils import get_coco, get_coco_kp
from .group_by_aspect_ratio import create_aspect_ratio_groups, GroupedBatchSampler

def collate_fn(batch):
    return tuple(zip(*batch))

def get_dataset(name, image_set, transforms, data_path):
    paths = {"coco": (data_path, get_coco, 3), "coco_kp": (data_path, get_coco_kp, 2)}
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transforms)
    return ds, num_classes

def get_dataloader(ds_path, batch_size, num_workers, distributed, aspect_ratio_group_factor):
    transforms = T.Compose([
        T.RandomIoUCrop(),
        T.RandomHorizontalFlip(),
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float),
        T.Resize((320,320))
    ])
    dataset, _ = get_dataset("coco", "train", transforms, ds_path)
    dataset_test, _ = get_dataset("coco", "val", transforms, ds_path)
    if distributed:
        train_sampler = DistributedSampler(dataset)
        test_sampler = DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = RandomSampler(dataset)
        test_sampler = SequentialSampler(dataset_test)

    if aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, batch_size)
    else:
        train_batch_sampler = BatchSampler(train_sampler, batch_size, drop_last=True)

    data_loader = DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn
    )
    data_loader_test = DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn
    )
    return data_loader, data_loader_test, train_sampler