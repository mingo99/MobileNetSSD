import torch
import torchvision

model = torchvision.models.detection.ssdlite320_mobilenet_v3_large()
model = model.eval().cuda()
dummy_input = torch.rand((1, 3, 320, 320), device="cuda")
features = model.backbone(dummy_input)
for key in features.keys():
    print(features[key].shape)