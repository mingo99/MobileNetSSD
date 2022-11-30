import os
import torch
import numpy as np

from _utils import get_fixed_point

def get_quant_factors(state_dict):
    iact_scales      = []
    iact_zero_points = []
    wgt_scales       = []
    bias_scales      = []
    """Collect quantization scales"""
    for num,item in enumerate(state_dict.items()):
        """Collect activation post process scale and zero_point"""
        if 'scale' in item[0]:
            scale = item[1].item()
            scale_int, f = get_fixed_point(scale,10)
            scale_float = scale_int/(2**f)
            iact_scales.append([scale,scale_float,scale_int,f])
        if 'zero_point' in item[0]:
            zero_point = item[1].item()
            iact_zero_points.append(zero_point)
        """Collect activation post process scale and zero_point"""
        if 'weight' in item[0]:
            scales = []
            for scale in item[1].q_per_channel_scales():
                scale = scale.item()
                scale_int, f = get_fixed_point(scale, 10) 
                scale_float = scale_int/(2**f)
                scales.append([scale,scale_float,scale_int,f])
            wgt_scales.append(scales)
        if "fc._packed_params._packed_params" in item[0]:
            scales = []
            scale = item[1][0].q_per_channel_scales().item()
            scale_int, f = get_fixed_point(scale, 10)
            scale_float = scale_int/(2**f)
            scales.append([scale,scale_float,scale_int,f])
            wgt_scales.append(scales)
    for lay in range(len(wgt_scales)):
        """Get bias scale and zero_point"""
        scales = []
        for oc in range(len(wgt_scales[lay])):
            scale = wgt_scales[lay][oc][0]*iact_scales[lay][0]
            scale_int, f = get_fixed_point(scale, 10)
            scale_float = scale_int/(2**f)
            scales.append([scale,scale_float,scale_int,f])
        bias_scales.append(scales)
    return iact_scales, iact_zero_points, wgt_scales, bias_scales

def get_weight_bias(state_dict,Sb):
    weights = []
    biases = []
    bias_num = 0
    for num,item in enumerate(state_dict.items()):
        if "weight" in item[0]:
            weight_int = item[1].int_repr().numpy()
            weight_scale = item[1].q_per_channel_scales().numpy()
            weights.append([weight_int,weight_scale])
        if "bias" in item[0]:
            bias_ints = []
            bias_scales = []
            for oc,bias in enumerate(item[1]):
                bias = bias.item()
                bias_int = int(bias//Sb[bias_num][oc][0])
                bias_scale = Sb[bias_num][oc][0]
                bias_ints.append(bias_int)
                bias_scales.append(bias_scale)
            biases.append([np.array(bias_ints),np.array(bias_scales)])
            bias_num += 1
        if "fc._packed_params._packed_params" in item[0]:
            bias_ints = []
            bias_scales = []
            weight_int = item[1][0].int_repr().numpy()
            weight_scale = item[1][0].q_per_channel_scales().numpy()
            weights.append([weight_int, weight_scale])
            bias = item[1][1].item()
            bias_int = int(bias//Sb[bias_num][0][0])
            bias_scale = Sb[bias_num][0][0]
            bias_ints.append(bias_int)
            bias_ints = np.array(bias_ints)
            bias_scales.append(bias_scale)
            bias_scales = np.array(bias_scales)
            biases.append([bias_ints,bias_scales])
    return np.array(weights), np.array(biases)

def write_params_to_file(state_dict,Sb):
    """Write wieght in file"""
    wgt_num = 0
    bias_num = 0
    for num,item in enumerate(state_dict.items()):
        if "weight" in item[0]:
            with open(f"./params/weight_{wgt_num}.txt", "w") as f:
                t_shape = item[1].shape
                f.write(f"{item[0]}\tShape:{t_shape}\n")
                for oc in range(t_shape[0]):
                    for ic in range(t_shape[1]):
                        for row in range(t_shape[2]):
                            for col in range(t_shape[3]):
                                f.write(f"{hex(item[1].int_repr()[oc][ic][row][col].item()&0xFF).replace('0x','')}\t")
                            f.write(f"\n")
                f.write(f"\n")
            wgt_num += 1
        if "bias" in item[0]:
            with open(f"./params/bias_{bias_num}.txt", "w") as f:
                t_shape = item[1].shape
                f.write(f"{item[0]}\tShape:{t_shape}\n")
                for oc in range(t_shape[0]):
                    bias = torch.quantize_per_tensor(item[1][oc],Sb[bias_num][oc][0], 0, torch.qint8)
                    f.write(f"{hex(bias.int_repr().item()&0xFF).replace('0x','')}\n")
            bias_num += 1
        if "fc._packed_params._packed_params" in item[0]:
            with open(f"./params/weight_{wgt_num}.txt", "w") as f:
                t_shape = item[1][0].shape
                f.write(f"{item[0]}\tShape:{t_shape}\n")
                for oc in range(t_shape[0]):
                    for ic in range(t_shape[1]):
                        f.write(f"{hex(item[1][0].int_repr()[oc][ic].item()&0xFF).replace('0x','')}\t")
                    f.write(f"\n")
            with open(f"./params/bias_{bias_num}.txt", "w") as f:
                bias = torch.quantize_per_tensor(item[1][1][0],Sb[bias_num][0][0], 0, torch.qint32)
                f.write(f"{hex(bias.int_repr().item()&0xFF).replace('0x','')}\n")

def get_params():
    torch.set_printoptions(precision=5)
    quant_model_state_dict = torch.load('./model/quant_model_selected/quant_model_0.pt')
    if not os.path.exists('./params/0_state_dict_name.txt'):
        with open("./params/0_state_dict_name.txt", "w") as f:
            for key in quant_model_state_dict.keys():
                f.write(f"{key}\n")
    Sx, Zx, Sw, Sb = get_quant_factors(quant_model_state_dict)
    # print(quant_model_state_dict['volume.0.0.bias'][1].item())
    # print(hex(int(quant_model_state_dict['volume.0.0.bias'][1].item()//Sb[0][1][0])&0xFFFF))
    # print((quant_model_state_dict['volume.0.0.bias'][1].item()//Sb[0][1][0])*Sb[0][1][0])
    # print(Sb[0][1][0])
    # # self.write_params_to_file(quant_model_state_dict,Sb)
    w,b = get_weight_bias(quant_model_state_dict,Sb)
    print(w)
    print(b)
    print(len(w))
    print(len(b))


if __name__ == "__main__":
    state_dict0 = torch.load("../weights/epoch0/ssdlite320_mobilenet_v3_large_int8.pth")
    state_dict1 = torch.load("../weights/epoch1/ssdlite320_mobilenet_v3_large_float32.pth")
    state_dict2 = torch.load("../weights/epoch2/ssdlite320_mobilenet_v3_large_float32.pth")
    state_dict3 = torch.load("../weights/ssdlite320_mobilenet_v3_large_calibrated_model.pth")
    print(state_dict0['backbone.features.0.5.block.2.skip_mul.scale'])
    print(state_dict1['backbone.features.0.5.block.2.skip_mul.scale'])
    print(state_dict3['backbone.features.0.5.block.2.skip_mul.scale'])
    # for key in state_dict0.keys():
    #     print(state_dict0[key].int_repr())
    #     break
    # print(state_dict0['backbone.features.0.0.0.scale'])
    # print(state_dict1['backbone.features.0.0.0.scale'])