import os
import torch
import numpy as np
from model import get_quant_model, ssdlite_calibrate

def ssdlite_quantize():
    """
    Build quantizable ssdlite and calibrate with coco dataset.

    Return:
        model(QuantizableSSD): Quantized and calibrated SSDLite 
    """
    model = get_quant_model('cpu')
    model = ssdlite_calibrate(model)
    return model

if __name__ == "__main__":
    # model = get_quant_model('cpu')
    # for key in model.state_dict().keys():
    #     print(key)
    ssdlite_quantize()