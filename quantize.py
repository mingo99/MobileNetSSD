import os
import torch
import numpy as np
from model import get_quant_model, ssdlite_calibrate
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
    model = ssdlite_quantize()
    detect_image("samples/image_1.jpg",0.5,True,False,"./weights/epoch1/ssdlite320_mobilenet_v3_large_int8.pth")