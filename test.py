import argparse
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
        detect_image(args['input'],args['threshold'],args['quantize'],True,"./weights/ssdlite320_mobilenet_v3_large_calibrated_model.pth")

if __name__ == "__main__":
    test()