import torch
from datasets import get_coco_datasets, coco_eval
from model import get_model, get_quant_model, qssdlite320_mobilenet_v3_large
from _utils import model_save, model_load, postprocess_as_ann, anns_to_json

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
    # model = get_model(device, True)
    # test_loader = get_coco_datasets(32,False)
    # eval_model(model, test_loader, device)
    a = [
        {
            "image_id": 558840,
            "bbox": [
                199.84,
                200.46,
                77.71,
                70.88
            ],
            "category_id": 58,
            "id": 156
        },
        {
            "image_id": 200365,
            "bbox": [
                234.22,
                317.11,
                149.39,
                38.55
            ],
            "category_id": 58,
            "id": 509
        }
    ]
    print(a)
    print(a.sort())