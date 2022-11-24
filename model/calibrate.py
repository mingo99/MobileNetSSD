import os
import torch
import torch.nn as nn
from datasets import get_coco_datasets

def ssdlite_calibrate(model):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    print(f"Calibrate is enable, open {device} as computation device.")
    with torch.no_grad():
        for epoch in range(10):
            model = nn.DataParallel(model)
            model = model.to(device)
            dir = f"./weights/epoch{epoch}"
            if not os.path.exists(dir):
                os.makedirs(dir)
            for i, data in enumerate(get_coco_datasets(32)):
                print(f"Epoch:{epoch} | Batch:{i}")
                # print(data[0].shape)
                image = data[0].to(device)
                model(image)
                break
            torch.cuda.empty_cache()
            model = model.to('cpu')
            torch.save(model.state_dict(),f"./weights/epoch{epoch}/ssdlite320_mobilenet_v3_large_float32.pth")
            torch.ao.quantization.convert(model, inplace=True)
            torch.save(model.state_dict(),f"./weights/epoch{epoch}/ssdlite320_mobilenet_v3_large_int8.pth")
    return model