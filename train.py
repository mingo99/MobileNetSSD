from datasets import get_coco_datasets, coco_eval
from model import get_model, get_quant_model, qssdlite320_mobilenet_v3_large
from _utils import model_save, model_load, postprocess_as_ann, anns_to_json
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epoch_num', default=100, type=int, 
                    help='Indicate number of total train epochs')
parser.add_argument('-b', '--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('-lr', '--learning_rate', default=0.01, type=float,
                    help='Learning rate')
args = parser.parse_args()

EPOCHS = args.epoch_num
BATCH_SIZE = args.batch_size
LR = args.learning_rate
ITERS_ONE_EPOCH = (82782+BATCH_SIZE)//BATCH_SIZE

def train_one_epoch(epoch, model, optimizer, train_loader, device):
    with open(f"./log/log_net{epoch+1:02d}.log", "w")as f:
        print(f'Epoch:{epoch+1}')
        model.train()
        model = model.to(device)
        total_loss_b = 0.0
        total_loss_c = 0.0
        for i, data in enumerate(train_loader):
            # 数据读取
            length = len(train_loader)
            images, targets = data
            images = images.to(device)
            for target in targets:
                target["boxes"] = target["boxes"].to(device)
                target["labels"] = target["labels"].to(device)
            # 梯度清零
            optimizer.zero_grad()
            # forward + backward
            losses = model(images, targets)
            loss_b = losses['bbox_regression'] 
            loss_c = losses['classification']
            loss = loss_b + loss_c
            loss.backward()
            optimizer.step()
            # 统计总损失
            total_loss_b += loss_b
            total_loss_c += loss_c
            # 终端打印训练关键信息并保存为log文件
            print(f"LR:{(optimizer.state_dict()['param_groups'][0]['lr']):.4f} | total_iter:{(i+1+epoch*length)} [iter:{i+1}/{ITERS_ONE_EPOCH} in epoch:{epoch}] | Loss_b: {(total_loss_b/(i + 1)):.03f} | Loss_c: {(total_loss_c/(i + 1)):.03f}")
            f.write(f"LR:{(optimizer.state_dict()['param_groups'][0]['lr']):.4f} | total_iter:{(i+1+epoch*length)} [iter:{i+1}/{ITERS_ONE_EPOCH} in epoch:{epoch}] | Loss_b: {(total_loss_b/(i + 1)):.03f} | Loss_c: {(total_loss_c/(i + 1)):.03f}")
            f.write('\n')
            f.flush()
    print(f"Finish {epoch+1}th epoch training.")

def train():
    print('Quantization Aware Training SSD on: coco')
    print('Using the specified args:')
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LR}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = qssdlite320_mobilenet_v3_large()
    model.train()
    # Settings of QAT
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    model.fuse_model()
    torch.quantization.prepare_qat(model, inplace=True)

    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    train_loader = get_coco_datasets(BATCH_SIZE, True)
    # test_loader = get_coco_datasets(BATCH_SIZE, False)

    start_epoch = model_load(model, optimizer, "./checkpoint/")
    for epoch in range(start_epoch, EPOCHS):
        train_one_epoch(epoch,model,optimizer,train_loader,device)
        # if (epoch+1)%10 == 0:
        # test_per_ten_epoch(epoch,model,test_loader,device)
        model_save(epoch, model.state_dict(), optimizer.state_dict(), f'./checkpoint/ckp_net{(epoch+1):02d}.pth')
    torch.quantization.convert(model, inplace=True)
    torch.save(model, "./checkpoint/ckp_net_int8.pth")

if __name__ == "__main__":
    train()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = get_model(device,True)
    # test_loader = get_coco_datasets(BATCH_SIZE, False)
    # test_per_ten_epoch(9,model,test_loader,device)