import os
import torch
import numpy as np
from model import get_quant_model, ssdlite_calibrate, ssdlite_with_quant_weights
from detection import detect_image

def ssdlite_quantize():
    """
    Build quantizable ssdlite and calibrate with coco dataset.

    Return:
        model(QuantizableSSD): Quantized and calibrated SSDLite 
    """
    model = get_quant_model('cpu')
    model = ssdlite_calibrate(model,1,128)
    return model

if __name__ == "__main__":
    # model = get_quant_model('cpu')
    # for key in model.state_dict().keys():
    #     print(key)
    model = get_quant_model('cpu',True)
    # model = ssdlite_with_quant_weights("./weights/ssdlite320_mobilenet_v3_large_calibrated_model.pth")
    state_dict0 = torch.load("./weights/epoch0/ssdlite320_mobilenet_v3_large_int8.pth")
    state_dict1 = torch.load("./weights/epoch1/ssdlite320_mobilenet_v3_large_float32.pth")
    state_dict2 = torch.load("./weights/epoch2/ssdlite320_mobilenet_v3_large_float32.pth")
    state_dict3 = torch.load("./weights/ssdlite320_mobilenet_v3_large_calibrated_model.pth")
    print(state_dict0.keys())
    print(state_dict0['backbone.features.0.3.block.2.0.scale'],state_dict0['backbone.features.0.3.block.2.0.zero_point'])
    print(state_dict1['backbone.features.0.3.block.2.0.scale'],state_dict1['backbone.features.0.3.block.2.0.zero_point'])
    print(state_dict3['backbone.features.0.3.block.2.0.scale'],state_dict3['backbone.features.0.3.block.2.0.zero_point'])
    print(state_dict0['backbone.features.0.4.block.2.skip_mul.scale'],state_dict3['backbone.features.0.4.block.2.skip_mul.zero_point'])
    print(state_dict1['backbone.features.0.4.block.2.skip_mul.scale'],state_dict3['backbone.features.0.4.block.2.skip_mul.zero_point'])
    print(state_dict3['backbone.features.0.4.block.2.skip_mul.scale'],state_dict3['backbone.features.0.4.block.2.skip_mul.zero_point'])
    