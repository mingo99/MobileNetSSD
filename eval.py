import torch
from datasets import get_coco_datasets, coco_eval
from model import get_model, get_quant_model, qssdlite320_mobilenet_v3_large
from _utils import model_save, model_load, postprocess_as_ann, anns_to_json
import torchvision

def eval_model(model, test_loader, device):
    dt_path = f"./eval_res/dt_anns.json"
    model.eval()
    model = model.to(device)
    res_anns = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            print(f"Test {i}th batch...")
            # 数据读取
            length = len(test_loader)
            images, targets = data
            images = images.to(device)
            outputs = model(images)
            postprocess_as_ann(res_anns,targets,outputs,0.5)
        print("Test done!")
    anns_to_json(res_anns,dt_path)
    coco_eval(dt_path,'bbox')

if __name__ == "__main__":
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # model = get_model(device, True)
    # weights=torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT
    # model = torchvision.models.detection.ssd300_vgg16(pretrained=True,weights=weights)
    # test_loader = get_coco_datasets(32,False)
    # eval_model(model, test_loader, device)
    dt_path = f"./eval_res/dt_anns.json"
    coco_eval(dt_path,'bbox')