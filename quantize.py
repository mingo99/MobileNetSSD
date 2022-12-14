import os
import torch
import numpy as np
from model import get_quant_model, ssdlite_with_quant_weights
from detection import detect_image

def ssdlite_quant_calib():
    """
    Build quantizable ssdlite and calibrate with coco dataset.

    Return:
        model(QuantizableSSD): Quantized and calibrated SSDLite 
    """
    return get_quant_model('cpu', True, 64, 1)

def ssdlite_quant():
    """
    Build quantizable ssdlite without calibration.

    Return:
        model(QuantizableSSD): Only quantized SSDLite 
    """
    return get_quant_model('cpu')

if __name__ == "__main__":
    # model = get_quant_model('cpu')
    # for key in model.state_dict().keys():
    #     print(key)
    path = "./weights/ssdlite320_mobilenet_v3_large_calibrated_pre_model.pth"
    model = get_quant_model('cpu',path,True,128,5)
    # model = ssdlite_with_quant_weights("./weights/ssdlite320_mobilenet_v3_large_calibrated_model.pth")
    # state_dict0 = torch.load(path)
    # print(state_dict0.keys())
    # print(state_dict0['backbone.features.0.3.block.2.0.scale'],state_dict0['backbone.features.0.3.block.2.0.zero_point'])
    # print(state_dict0['backbone.features.0.4.block.2.skip_mul.scale'],state_dict3['backbone.features.0.4.block.2.skip_mul.zero_point'])