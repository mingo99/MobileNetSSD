import os
import json
import torch
import numpy as np
from numpyencoder import NumpyEncoder

def model_save(epoch, model_state_dict, optimizer_state_dict, PATH):
    """To save checkpoint"""
    print("Start saving model...")
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            }, PATH)
    print("Finish saving model...")

def model_load(model, optimizer, PATH):
    """To load checkpoint"""
    print("Start loading checkpoint!!!\n...")
    if os.path.isdir(PATH):
        try:
            if len(os.listdir(PATH)) > 0:
                path_list = os.listdir(PATH)
                path_list.sort(key=lambda x:int(x.split('ckp_net')[1].split('.pth')[0]))
                checkpoint = torch.load(PATH + path_list[-1])
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if torch.cuda.is_available():
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()
                start_epoch = checkpoint['epoch'] + 1
                print('Load checkpoint successfully!!!')
            else:
                print('Can\'t found the .pth file in checkpoint and start from scratch!')
                start_epoch = 0
        except FileNotFoundError:
            print(f'Can\'t found the directory `{PATH}` and start from scratch!')
            start_epoch = 0
    else:
        start_epoch = 0
        print("Can\'t find folder 'checkpoint' and Start from scratch")
    return start_epoch

def get_fixed_point(float_x, m):
        x = float_x
        fixed_x = 0
        f = 0
        while(int(x)<(2**m)):
            if(int(x*2)<(2**m)):
                x = x*2
                f += 1
                continue
            else:
                fixed_x = int(x)
                break
        return fixed_x, f

def postprocess_as_ann(res_anns, targets, outputs, detection_threshold):
    for i, output in enumerate(outputs):
        pred_scores = output['scores'].detach().cpu().numpy()
        pred_bboxes = output['boxes'].detach().cpu().numpy()
        pred_labels = output['labels'].detach().cpu().numpy()
        # get boxes above the threshold score
        idxs = pred_scores >= detection_threshold
        boxes = pred_bboxes[idxs]
        scores = pred_scores[idxs]
        labels = pred_labels[idxs]
        for box, score, label in zip(boxes,scores,labels):
            box[2] = box[2] - box[0]
            box[3] = box[3] - box[1]
            ann = {"image_id": targets[i]['image_id'].item(),
                    "category_id": label,
                    "bbox": box.tolist(),
                    "score": score
            }
            res_anns.append(ann)

def anns_to_json(anns, path):
    with open(path, 'w') as f:
        json.dump(anns, f, indent=4, cls=NumpyEncoder)

if __name__ == "__main__":
    pass