from datasets import get_dataloader, coco_eval
from model import get_model, get_quant_model, qssdlite320_mobilenet_v3_large
from _utils import model_save, model_load, postprocess_as_ann, anns_to_json
from detection import predict, draw_boxes
from typing import List
from tqdm import tqdm

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import copy

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epoch_num', default=100, type=int, 
                    help='Indicate number of total train epochs')
parser.add_argument('-b', '--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('-n', '--workers', default=1, type=int,
                    help='Number of dataloader workers.')
parser.add_argument('-lr', '--learning_rate', default=0.01, type=float,
                    help='Learning rate')
parser.add_argument('-ds', '--ds_root', default='../../data/cocofb/', type=str,
                    help='The root path of dataset.')
args = parser.parse_args()

EPOCHS = args.epoch_num
BATCH_SIZE = args.batch_size
LR = args.learning_rate

def train_one_epoch(epoch, model, optimizer, train_loader, device, ITERS_ONE_EPOCH):
    model.train()
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(train_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )
    with open(f"./log/log_net{epoch:02d}.log", "w")as f:
        print(f'Epoch:{epoch}')
        # model.train()
        # model = model.to(device)
        total_loss_b = 0.0
        total_loss_c = 0.0
        for i, data in enumerate(train_loader):
            # 数据读取
            length = len(train_loader)
            images, targets = data
            # images = images.to(device)
            # for target in targets:
            #     target["boxes"] = target["boxes"].to(device)
            #     target["labels"] = target["labels"].to(device)
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # forward
            losses = model(images, targets)
            # backward
            loss_b = losses['bbox_regression'] 
            loss_c = losses['classification']
            loss = loss_b + loss_c
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            if lr_scheduler is not None:
                lr_scheduler.step()              
            # 统计总损失
            total_loss_b += loss_b.item()
            total_loss_c += loss_c.item()
            # 终端打印训练关键信息并保存为log文件
            if (i+1)%10 == 0:
                print(f"LR:{(optimizer.state_dict()['param_groups'][0]['lr']):.10f} | total_iter:{(i+1+epoch*length)} [iter:{i+1}/{ITERS_ONE_EPOCH} in epoch:{epoch}] | Loss_b: {(total_loss_b/(i + 1)):.03f} | Loss_c: {(total_loss_c/(i + 1)):.03f}")
                f.write(f"LR:{(optimizer.state_dict()['param_groups'][0]['lr']):.10f} | total_iter:{(i+1+epoch*length)} [iter:{i+1}/{ITERS_ONE_EPOCH} in epoch:{epoch}] | Loss_b: {(total_loss_b/(i + 1)):.03f} | Loss_c: {(total_loss_c/(i + 1)):.03f}")
                # print(f"LR:{(optimizer.state_dict()['param_groups'][0]['lr']):.10f} | total_iter:{(i+1+epoch*length)} [iter:{i+1}/{ITERS_ONE_EPOCH} in epoch:{epoch}] | Loss_b: {loss_b:.03f} | Loss_c: {loss_c:.03f}")
                # f.write(f"LR:{(optimizer.state_dict()['param_groups'][0]['lr']):.10f} | total_iter:{(i+1+epoch*length)} [iter:{i+1}/{ITERS_ONE_EPOCH} in epoch:{epoch}] | Loss_b: {loss_b:.03f} | Loss_c: {loss_c:.03f}")
                f.write('\n')
                f.flush()
    print(f"Finish {epoch}th epoch training.")

def test_in_train(epoch, model, val_loader, device):
    dt_path = f"./eval_res/dt_anns_{epoch:03d}.json"
    # val_loader = valset.get_coco_dataloader(BATCH_SIZE)
    model.eval()
    res_anns = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(val_loader, desc="Evaluation")):
            # 数据读取
            length = len(val_loader)
            images, targets = data
            # images = images.to(device)
            images = list(image.to(device) for image in images)
            outputs = model(images)
            postprocess_as_ann(res_anns,targets,outputs,0.3)
    anns_to_json(res_anns,dt_path)
    coco_eval(dt_path,'bbox')

# def qat_test_in_train(epoch, net, valset: CocoDataset, device):
#     dt_path = f"./eval_res/dt_anns_{epoch:03d}.json"
#     val_loader = valset.get_coco_dataloader(BATCH_SIZE)
#     model = net.to(device)
#     model = torch.quantization.convert(net)
#     model.eval()
#     res_anns = []
#     with torch.no_grad():
#         for i, data in enumerate(val_loader):
#             print(f"Test {i}th batch...")
#             # 数据读取
#             length = len(val_loader)
#             images, targets = data
#             images = images.to(device)
#             outputs = model(images)
#             postprocess_as_ann(res_anns,targets,outputs,0.3)
#         print("Test done!")
#     anns_to_json(res_anns,dt_path)
#     valset.coco_eval(dt_path,'bbox')

def train():
    print('Training SSD on: coco')
    print('Using the specified args:')
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LR}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(device, False)
    model.train()

    # trainset = get_dataset()
    # dataset, num_classes = get_dataset("coco", "train", get_transform(True, args), args.data_path)

    # trainset = CocoDataset(args.ds_root, 2023, 'train')
    # valset = CocoDataset(args.ds_root, 2023, 'val')
    train_loader, val_loader = get_dataloader(args.ds_root, args.workers)
    ITERS_ONE_EPOCH = len(train_loader)
    # num_steps = len(train_loader)*EPOCHS

    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=4e-5)
    # scheduler = optim.lr_scheduler.LinearLR(optimizer, gamma=0.9)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    start_epoch = model_load(model, optimizer, "./checkpoint/normal/")
    for epoch in range(start_epoch, EPOCHS):
        train_one_epoch(epoch,model,optimizer,train_loader,device,ITERS_ONE_EPOCH)
        if (epoch+1)%10 == 0:
            test_in_train(epoch,model,val_loader,device)
        scheduler.step()
        model_save(epoch, model.state_dict(), optimizer.state_dict(), f'./checkpoint/normal/ckp_net{epoch:02d}.pth')
    # test_in_train(EPOCHS,model,val_loader,device)

# def qat_train():
#     print('Quantization Aware Training SSD on: coco')
#     print('Using the specified args:')
#     print(f"Epochs: {EPOCHS}")
#     print(f"Batch Size: {BATCH_SIZE}")
#     print(f"Learning Rate: {LR}")

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # model = qssdlite320_mobilenet_v3_large()
#     weights = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
#     model = qssdlite320_mobilenet_v3_large(pretrained=True,weights=weights)
#     model.train()
#     # Settings of QAT
#     model.fuse_model(is_qat=True)
#     model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
#     torch.quantization.prepare_qat(model, inplace=True)

#     train_loader = get_coco_dataloader(BATCH_SIZE, True)
#     test_loader = get_coco_dataloader(BATCH_SIZE, False)
#     # num_steps = len(train_loader)*EPOCHS

#     optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
#     # scheduler = optim.lr_scheduler.LinearLR(optimizer, gamma=0.9)
#     scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
#     # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

#     start_epoch = model_load(model, optimizer, "./checkpoint/qat/")
#     for epoch in range(start_epoch, EPOCHS):
#         model.train()
#         _ = model_load(model, optimizer, "./checkpoint/qat/")
#         train_one_epoch(epoch,model,optimizer,train_loader,device)
#         if (epoch+1)%10 == 0:
#             test_in_train(epoch,model,test_loader,"cpu")
#         # scheduler.step()
#         model_save(epoch, model.state_dict(), optimizer.state_dict(), f'./checkpoint/qat/ckp_net{epoch:02d}.pth')
#     model.to('cpu')
#     torch.quantization.convert(model, inplace=True)
#     test_in_train(10,model,test_loader,device)

if __name__ == "__main__":
    train()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = get_model(device,True)
    # test_loader = get_coco_datasets(BATCH_SIZE, False)
    # test_per_ten_epoch(9,model,test_loader,device)