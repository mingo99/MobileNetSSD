import torch
import torchvision
from torchinfo import summary

def get_model(device):
    # load the model 
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True,weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    # load the model onto the computation device
    model = model.eval().to(device)
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ssdlite = get_model(device)
    ssdlite.eval()
    # summary(ssdlite,(1,3,320,320), device=device)
    print(ssdlite)