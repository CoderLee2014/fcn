import mxnet as mx
from symbol.rfcn import rfcn
import os
import argparse
import Queue
import logging
from multiprocessing import Process, Manager, Value
from dataset.iterator import DetIter, DetRecordIter
from dataset.icdar import ICDAR
from config.config import cfg
from metric import Metrics

WIDTH = 320
HEIGHT = 320
NUM_LABEL = 2

def init_conv(ctx, sym, train_iter):
    """
    use zero initialization for better convergence, because it tends to oputut 0,
        and the label 0 stands for background(non-text), which may occupy most size of one image.
    params:
        ctx: [mx.cpu()] or [mx.gpu()]
            list of mxnet contexts
        sym: mx.symbol
            the network symbol
        train_iter: mx.io.DataIter
            the training data iterator
    """
    data_shape_dict = dict(train_iter.provide_data + train_iter.provide_label)
    arg_shape, _, aux_shape = sym.infer_shape(**data_shape_dict)
    arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))
    aux_shape_dict = dict(zip(sym.list_auxiliary_states(), aux_shape))
    init = mx.init.Xavier(factor_type="in", rnd_type='gaussian', magnitude=2.34)
    args = dict()
    auxs = dict()

    for k in sym.list_arguments():
        if k in data_shape_dict:
            continue

        args[k] = mx.nd.zeros(arg_shape_dict[k])
        init(k, args[k])
        if k.startswith('reg'):
            args[k] = mx.nd.zeros(shape=arg_shape_dict[k])
    for k in sym.list_auxiliary_states():
        auxs[k] = mx.nd.zeros(shape=aux_shape_dict[k])
        init(k, auxs[k])
    return args, auxs

def parse_args():
    parser = argparse.ArgumentParser(description='Train a direct regression multi-oriented neural network')
    parser.add_argument('--gpus', dest='gpus', default='4, 5, 6, 7', type=str)
    parser.add_argument('--image-set', dest='image_set', default='ICDAR', type=str)
    parser.add_argument('--year', dest='year', default='2015', type=str)
    parser.add_argument('--val-set', dest='val_set', default='icdar2015_val', type=str)
    parser.add_argument('--data-shape', dest='data_shape', default=320, type=int)
    parser.add_argument('--log', dest='log_file', type=str, default="train_icdar.log")
    parser.add_argument('--devkit-path', dest='devkit_path', type=str, default=os.path.join(os.getcwd(), 'data'))
    #parser.add_argument('--train-path', dest='train_path', type=str, default=os.path.join(os.getcwd(), 'data', 'ICDAR', 'train.rec'))
   # parser.add_argument('--val-path', dest='val_path', type=str, default=os.path.join(os.getcwd(), 'data', 'ICDAR', 'val.rec'))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    num_epoch = 2000
    begin_epoch = 51
    batch_size = 20
    train_from_scratch = False
    prefix_out = './model/text-location-hinge'

    # context list
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    #ctx = mx.cpu()

    # prepare data
    print "start to load data..."
    #imdb_train = load_icdar(args.image_set, args.devkit_path, shuffle=True) 
    #train_iter = DetIter(imdb_train, batch_size, data_shape=args.data_shape, shuffle=cfg.TRAIN.EPOCH_SHUFFLE, is_train=True) 
    train_path = os.path.join(args.devkit_path, args.image_set, 'train.rec')
    path_imglist = os.path.join(args.devkit_path, args.image_set, 'train.lst')
    train_iter = DetRecordIter(train_path, batch_size, data_shape=args.data_shape, path_imglist=path_imglist)

    #imdb_val = load_icdar(args.val_set, args.devkit_path)
    #val_iter = DetIter(imdb_val, batch_size, data_shape=args.data_shape, is_train=True)
    #val_iter = mx.io.PrefetchingIter(val_iter)
    val_path = os.path.join(args.devkit_path, args.image_set, 'val.rec')
    path_imglist = os.path.join(args.devkit_path, args.image_set, 'val.lst')
    val_iter = DetRecordIter(val_path, batch_size, data_shape=args.data_shape, path_imglist=path_imglist)

    train_iter = mx.io.PrefetchingIter(train_iter)

    # build a NN
    print "start to build rfcn..."
    sym = rfcn()
    print "Building NN finished."

    # set up logger
    head = "%(asctime)-15s %(message)s"
    log_dir = './log'
    log_file = 'log_lr0.01_test.txt'
    log_file_full_name = os.path.join(log_dir, log_file)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logger = logging.getLogger()
    handler = logging.FileHandler(log_file_full_name)
    formatter = logging.Formatter(head)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    # init optimizer params
    optimizer_params = {'momentum': 0.9,
                        'wd': 4e-4,
                        'learning_rate': 0.0001,
                        'rescale_grad': 1.0}
    logging.basicConfig(level=logging.DEBUG, format=head)

    if train_from_scratch:
        arg_names = sym.list_arguments()
        print arg_names
        arg_params, aux_params = init_conv(ctx, sym, train_iter)
        begin_epoch = 0
    else:
        _, arg_params, aux_params = mx.model.load_checkpoint(prefix_out, begin_epoch)
    
    #init training module
    model = mx.mod.Module(sym, label_names=("label", ), logger=logger, context=ctx)

    model.fit(train_iter,
              eval_data=val_iter,
              arg_params=arg_params,
              aux_params=aux_params,
              eval_metric=Metrics(),
              optimizer='sgd', optimizer_params=optimizer_params,
              begin_epoch=begin_epoch,
              num_epoch=num_epoch,
              batch_end_callback=mx.callback.Speedometer(batch_size, 10),
              epoch_end_callback=mx.callback.do_checkpoint(prefix_out))
    print "traininig finished."
