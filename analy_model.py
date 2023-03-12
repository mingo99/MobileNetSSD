import torch
import torchvision
from torchinfo import summary

ssdlite = torchvision.models.detection.ssdlite320_mobilenet_v3_large()
ssd = torchvision.models.detection.ssd300_vgg16()
ssdlite = ssdlite.cuda()
ssd = ssd.cuda()
# dummy_input = torch.rand((1, 3, 320, 320), device="cuda")
# features1 = ssdlite.backbone(dummy_input)
# _ = ssdlite(dummy_input)
# features2 = ssd.backbone(dummy_input)
# for key in features1.keys():
#     print(features1[key].shape)
# for anchor in ssdlite.anchors[0]:
#     print(anchor)
summary(ssdlite,(1, 3, 320, 320))
# summary(ssd,(1, 3, 300, 300))