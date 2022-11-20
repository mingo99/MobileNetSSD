import torch
import torchvision
from torchinfo import summary
from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names
from torchvision.models.quantization import *
from quantization import *
from PIL import Image

def get_model(device):
    # load the model 
    weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True,weights=weights)
    # model = torchvision.models.detection.ssd300_vgg16(pretrained=True,weights=torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT)
    # load the model onto the computation device
    model = model.eval().to(device)
    return model

def get_quant_model():
    weights = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = qssdlite320_mobilenet_v3_large(pretrained=True,weights=weights,quantize=True)
    return model.eval()

def get_mobilenetv3(device):
    weights=torchvision.models.quantization.MobileNet_V3_Large_QuantizedWeights.DEFAULT
    model = torchvision.models.quantization.mobilenet_v3_large(pretrained=True, weights=weights,quantize=True)
    return model.eval().to(device)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ssdlite = get_model(device)
    # # summary(ssdlite,(1,3,320,320), device=device)
    # print(ssdlite)
    # qmnetv3 = get_quant_model(device)
    # print(qmnetv3.state_dict().keys())
    # summary(ssdlite,(1,3,320,320), device=device)
    # print(qmnetv3)

    # Test quant_mobilenetv3
    transform = torchvision.transforms.ToTensor()
    model = get_mobilenetv3(device)
    image = Image.open('input/image_1.jpg')
    image = transform(image).to(device)
    image = image.unsqueeze(0)
    print(model(image))