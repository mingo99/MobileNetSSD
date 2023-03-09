import torch
import torchvision

ssdlite = torchvision.models.detection.ssdlite320_mobilenet_v3_large()
wh_pairs = ssdlite.anchor_generator._wh_pairs
print(ssdlite.anchor_generator.scales)
print(ssdlite.anchor_generator.steps)
ssdlite = ssdlite.eval().cuda()
dummy_input = torch.rand((1, 3, 320, 320), device="cuda")
features1 = ssdlite.backbone(dummy_input)
_ = ssdlite(dummy_input)
# features2 = ssd.backbone(dummy_input)
# for key in features1.keys():
#     print(features1[key].shape)
# for anchor in ssdlite.anchors[0]:
#     print(anchor)