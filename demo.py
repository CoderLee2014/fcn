import argparse
import mxnet as mx
import os
import sys
from symbol.rfcn import get_symbol
from symbol.rfcn import rfcn
from detect.detector import Detector

CLASSES = ('text', )
           #'bottle', 'bus', 'car', 'cat', 'chair',
           #'cow', 'diningtable', 'dog', 'horse',
           #'motorbike', 'person', 'pottedplant',
            #'sheep', 'sofa', 'train', 'tvmonitor')

def get_detector(prefix, epoch, data_shape, mean_pixels, ctx,
                 nms_thresh=0.5, force_nms=True):
    """
    wrapper for initialize a detector

    Parameters:
    ----------
    net : str
        test network name
    prefix : str
        load model prefix
    epoch : int
        load model epoch
    data_shape : int
        resize image shape
    mean_pixels : tuple (float, float, float)
        mean pixel values (R, G, B)
    ctx : mx.ctx
        running context, mx.cpu() or mx.gpu(?)
    force_nms : bool
        force suppress different categories
    """
    #net = get_symbol(nms_thresh, force_nms)
    net = rfcn(mode="test")
    detector = Detector(net, prefix, epoch, \
        data_shape, mean_pixels, ctx=ctx)
    return detector

def parse_args():
    parser = argparse.ArgumentParser(description='DRMO detection network demo')
    parser.add_argument('--network', dest='network', type=str, default='text-location',
                        choices=['text-location'], help='which network to use')
    parser.add_argument('--images', dest='images', type=str, default='./data/demo/img_901.jpg',
                        help='run demo with images, use comma(without space) to seperate multiple images')
    parser.add_argument('--dir', dest='dir', nargs='?',
                        help='demo image directory, optional', type=str)
    parser.add_argument('--ext', dest='extension', help='image extension, optional',
                        type=str, nargs='?')
    parser.add_argument('--epoch', dest='epoch', help='epoch of trained model',
                        default=67, type=int)
    parser.add_argument('--prefix', dest='prefix', help='trained model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'text-location'), type=str)
    parser.add_argument('--cpu', dest='cpu', help='(override GPU) use CPU to detect',
                        action='store_true', default=False)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0,
                        help='GPU device id to detect with')
    parser.add_argument('--data-shape', dest='data_shape', type=int, default=320,
                        help='set image shape')
    parser.add_argument('--mean-r', dest='mean_r', type=float, default=123,
                        help='red mean value')
    parser.add_argument('--mean-g', dest='mean_g', type=float, default=117,
                        help='green mean value')
    parser.add_argument('--mean-b', dest='mean_b', type=float, default=104,
                        help='blue mean value')
    parser.add_argument('--thresh', dest='thresh', type=float, default=0.7,
                        help='object visualize score threshold, default 0.7')
    parser.add_argument('--nms', dest='nms_thresh', type=float, default=0.5,
                        help='non-maximum suppression threshold, default 0.5')
    parser.add_argument('--force', dest='force_nms', type=bool, default=True,
                        help='force non-maximum suppression on different class')
    parser.add_argument('--timer', dest='show_timer', type=bool, default=True,
                        help='show detection time')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.cpu:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu_id)

    # parse image list
    image_list = [i.strip() for i in args.images.split(',')]
    assert len(image_list) > 0, "No valid image specified to detect"

    detector = get_detector(args.prefix, args.epoch,
                            args.data_shape,
                            (args.mean_r, args.mean_g, args.mean_b),
                            ctx, args.nms_thresh, args.force_nms)
    # run detection
    detector.detect_and_visualize(image_list, args.dir, args.extension,
                                  CLASSES, args.thresh, args.show_timer)
