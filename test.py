import argparse
import torch
from detection import detect_video, detect_image

def test():
        # construct the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', default=False, type=bool, 
                        help='indicate test whether video or image')
    parser.add_argument('-i', '--input', default='samples/image_1.jpg', 
                        help='path to input input image')
    parser.add_argument('-t', '--threshold', default=0.5, type=float,
                        help='detection threshold')
    parser.add_argument('-q', '--quantize', default=False, type=bool,
                        help='whether to quantize model')
    args = vars(parser.parse_args())

    if args['video']:
        detect_video(args['input'],args['threshold'],args['quantize'])
    else:
        detect_image(args['input'],args['threshold'],args['quantize'],True,"./weights/epoch0/ssdlite320_mobilenet_v3_large_calibrated_model.pth")

if __name__ == "__main__":
    test()
    # state_dict0 = torch.load("./weights/epoch0/ssdlite320_mobilenet_v3_large_calibrated_model.pth")
    # state_dict1 = torch.load("./weights/epoch1/ssdlite320_mobilenet_v3_large_calibrated_model.pth")
    # state_dict2 = torch.load("./weights/epoch2/ssdlite320_mobilenet_v3_large_calibrated_model.pth")
    # state_dict3 = torch.load("./weights/ssdlite320_mobilenet_v3_large_calibrated_model_0.pth")
    # print(state_dict0.keys())
    # print(state_dict0['backbone.features.0.3.block.2.0.scale'],state_dict0['backbone.features.0.3.block.2.0.zero_point'])
    # print(state_dict1['backbone.features.0.3.block.2.0.scale'],state_dict1['backbone.features.0.3.block.2.0.zero_point'])
    # print(state_dict2['backbone.features.0.3.block.2.0.scale'],state_dict1['backbone.features.0.3.block.2.0.zero_point'])
    # print(state_dict3['backbone.features.0.3.block.2.0.scale'],state_dict3['backbone.features.0.3.block.2.0.zero_point'])
    # print(state_dict0['backbone.features.0.4.block.2.skip_mul.scale'],state_dict3['backbone.features.0.4.block.2.skip_mul.zero_point'])
    # print(state_dict1['backbone.features.0.4.block.2.skip_mul.scale'],state_dict3['backbone.features.0.4.block.2.skip_mul.zero_point'])
    # print(state_dict2['backbone.features.0.4.block.2.skip_mul.scale'],state_dict3['backbone.features.0.4.block.2.skip_mul.zero_point'])
    # print(state_dict3['backbone.features.0.4.block.2.skip_mul.scale'],state_dict3['backbone.features.0.4.block.2.skip_mul.zero_point'])
    