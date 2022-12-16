import torch
import torchvision
import torchextractor as tx
from torchvision.models.resnet import ResNet18_Weights
from torchvision.models.vgg import VGG16_Weights
from torchinfo import summary

model = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
model = model.eval().to('cuda')
# model = tx.Extractor(model, ["conv1","layer1", "layer2", "layer3", "layer4"])
dummy_input = torch.rand(1, 3, 224, 224).to('cuda')
model_output = model(dummy_input)
# print(features['conv1'])
# feature_shapes = {name: f.shape for name, f in features.items()}
# print(feature_shapes)
summary(model,dummy_input.shape,device='cuda')
# {
#   'layer1': torch.Size([1, 64, 56, 56]),
#   'layer2': torch.Size([1, 128, 28, 28]),
#   'layer3': torch.Size([1, 256, 14, 14]),
#   'layer4': torch.Size([1, 512, 7, 7]),
# }