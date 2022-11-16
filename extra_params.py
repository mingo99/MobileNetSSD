import torch
import torchvision

def get_ssd(device):
    # load the model 
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
        pretrained=True,weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    # load the model onto the computation device
    return model.eval().to(device)

def get_modules_to_fused(state_keys):
    modules_to_fused = []
    for key in state_keys:
        if "0.weight" in key:
            modules_to_fused.append([key[:-8]+'0', key[:-8]+'1'])
    return modules_to_fused

def get_fused_model(model):
    modules_to_fused = get_modules_to_fused(model.state_dict().keys())
    return torch.quantization.fuse_modules(model, modules_to_fused)

def extra_params(device):
    ssd = get_ssd(device)
    fssd = get_fused_model(ssd)
    print(fssd)
    print(fssd.state_dict()['backbone.features.0.1.block.0.0.weight'].shape)

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

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extra_params(device)