import torch
import numpy as np
from datasets import get_dataloader
from model import get_model, ssdlite_with_weights
from utils import postprocess_as_ann, anns_to_json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

def coco_eval(dt_path):
    # 加载ground truth数据
    annFile = "../../data/cocofb/annotations/instances_val2017.json"
    cocoGt = COCO(annFile)
    cocoDt = cocoGt.loadRes(dt_path)

    # 创建评估器
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')

    # 运行评估
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    # print(cocoEval.eval['precision'][:,0,1,0,2])
    for catId in cocoEval.params.catIds:
        s = cocoEval.eval['precision'][0,:,catId-1,0,2]
        if len(s[s>-1])==0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s>-1])
        print(mean_s)

def eval_model(model, device):
    dt_path = f"./eval_res/ssdlite_dt_anns.json"
    _, val_loader, _ = get_dataloader("../../data/cocofb", 2017, 24, 8, False, -1)
    model = ssdlite_with_weights("./checkpoint/normal/checkpoint.pth", device)
    model = model.to(device)
    res_anns = []
    with torch.no_grad():
        for data in tqdm(val_loader, desc="Test Image"):
            images, targets = data
            images = list(img.to(device) for img in images)
            outputs = model(images)
            postprocess_as_ann(res_anns,targets,outputs,0.3)
        print("Test done!")
    anns_to_json(res_anns,dt_path)
    coco_eval(dt_path)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(device)
    eval_model(model, device)
    # dt_path = f"./eval_res/ssdlite_dt_anns.json"
    # coco_eval(dt_path)
