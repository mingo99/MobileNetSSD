from datasets import get_dataloader
from model import get_model
import utils
from engine import evaluate, train_one_epoch
from typing import List
from tqdm import tqdm

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import copy

import datetime
import os
import time

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default=100, type=int, 
                    help='Indicate number of total train epochs')
parser.add_argument('-b', '--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('-n', '--workers', default=1, type=int,
                    help='Number of dataloader workers.')
parser.add_argument('-lr', '--learning_rate', default=0.01, type=float,
                    help='Learning rate')
parser.add_argument('-ds', '--ds_root', default='../../data/cocofb/', type=str,
                    help='The root path of dataset.')
parser.add_argument('--world-size', default=4, type=int, help='number of distributed processes')
parser.add_argument('--local_rank', type=int, help='rank of distributed processes')
parser.add_argument("--print_freq", default=100, type=int, help="print frequency")
parser.add_argument("--output_dir", default="./checkpoint/normal/", type=str, help="path to save outputs")
parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
# Mixed precision training parameters
parser.add_argument("--amp", default=None, action="store_true", help="Use torch.cuda.amp for mixed precision training")
parser.add_argument(
        "--test-only",
        default=False,
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
args = parser.parse_args()

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.learning_rate

def main():
    utils.init_distributed_mode(args)
    print('Training SSD on: coco')
    print('Using the specified args:')
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LR}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(device)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    train_loader, val_loader, train_sampler = get_dataloader(args.ds_root, args.batch_size, args.workers, args.distributed, 3)

    parameters = [p for p in model.parameters() if p.requires_grad]

    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    optimizer = optim.SGD(parameters, lr=LR, momentum=0.9, weight_decay=4e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        torch.backends.cudnn.deterministic = True
        evaluate(model, val_loader, device=device)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, train_loader, device, epoch, 20, None)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": args,
                "epoch": epoch,
            }
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

        # evaluate after every 10 epoch
        if (epoch+1) % 10 ==0:
            evaluate(model, train_loader, device=device)
            evaluate(model, val_loader, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

if __name__ == "__main__":
    main()