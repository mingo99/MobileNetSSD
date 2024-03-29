import argparse
import torch
from detection import detect_video, detect_image

def simple_test():
        # construct the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', default=False, type=bool, 
                        help='indicate test whether video or image')
    parser.add_argument('-i', '--input', default='samples/image_1.jpg', 
                        help='path to input input image')
    parser.add_argument('-t', '--threshold', default=0.3, type=float,
                        help='detection threshold')
    parser.add_argument('-q', '--quantize', default=False, type=bool,
                        help='whether to quantize model')
    parser.add_argument('-p', '--path', default="./checkpoint/normal/best.pth", type=str,
                        help='Path of trained model.')
    args = vars(parser.parse_args())

    if args['video']:
        detect_video(args['input'],args['threshold'],args['quantize'])
    else:
        # detect_image(args['input'],args['threshold'],args['quantize'],True,"./checkpoint/ckp_net19.pth")
        # detect_image(args['input'],args['threshold'],args['quantize'],True,"./weights/ssdlite320_mobilenet_v3_large_coco-a79551df.pth")
        detect_image(args['input'],args['threshold'],args['quantize'],True,args["path"])

if __name__ == "__main__":
    simple_test()