import torch
import torchvision
from datasets import CocoDataset
from model import get_model, get_quant_model, qssdlite320_mobilenet_v3_large, ssdlite_with_quant_weights, ssdlite_with_weights
from _utils import model_save, model_load, postprocess_as_ann, anns_to_json
import json

def eval_model(model, valset:CocoDataset, device):
    dt_path = f"./eval_res/ssdlite_dt_anns.json"
    val_loader = valset.get_coco_dataloader(32)
    model.eval()
    model = model.to(device)
    res_anns = []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            print(f"Test {i}th batch...")
            # 数据读取
            length = len(val_loader)
            images, targets = data
            images = images.to(device)
            outputs = model(images)
            postprocess_as_ann(res_anns,targets,outputs,0.1)
        print("Test done!")
    anns_to_json(res_anns,dt_path)
    valset.coco_eval(dt_path,'bbox')

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = get_model(device, True)
    # weights=torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT
    # model = torchvision.models.detection.ssd300_vgg16(pretrained=True,weights=weights)
    model = get_model(device)
    valset = CocoDataset('../../data/cocofb/', 2023, 'val')
    eval_model(model, valset, device)
    # dt_path = f"./eval_res/ssdlite_dt_anns.json"
    # dt_anns = json.load(open(dt_path, 'r'))
    # for ann in dt_anns:
    #     for i, loc in enumerate(ann['bbox']):
    #         ann['bbox'][i] = round(loc,2)
    #     ann['score'] = round(ann['score'],2)
    # with open(f"./eval_res/ssdlite_dt_anns_new.json", 'w') as f:
    #     json.dump(dt_anns, f, indent=4)
    # dt_path = f"./eval_res/result.json"
    # coco_eval(dt_path,'bbox')
