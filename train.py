import mxnet as mx
from rfcn import rfcn
import numpy as np
import random
import argparse
import Queue
from multiprocessing import Process, Manager, Value
from gen.gendata import GenImage

WIDTH = 320
HEIGHT = 320
NUM_LABEL = 2

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label, bbox_names, bbox):
        self.data = data
        self.label = label
        self.bbox_target = bbox
        self.data_names = data_names
        self.label_names = label_names
        self.bbox_names = bbox_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]
    
    @property
    def provide_bbox(self):
        return [(n, x.shape) for n, x in zip(self.bbox_names, self.bbox_target)]

class Prefetch:
    def __init__(self, number, worker, f):
        self.m = Manager()
        self.q = self.m.Queue(worker * 2)
        self.number = number
        self.worker = worker
        self.f = f
        self.stop = Value('b', False)
        self._produce()

    def _reset(self):
        if self.stop.value():
            self.stop.value = False
            self._produce()

    def _produce(self):
        def add_queue():    
            random.seed()
            np.random.seed()
            while not self.stop.value:
                self.q.put(self.f())
        self.processes = [Process(target=add_queue) for _ in range(self.worker)]
        for p in self.processes:
            p.start()
    
    def _flush_queue(self):
        while 1:
            try: 
                self.q.get_nowait()
            except Queue.empty: 
                break

    def _join(self):
        if not self.stop.value:
            self.stop.value = True
            self._flush_queue()

            for p in self.processes:    
                p.join()
        
            self._flush_queue()

    def __iter__(self):
        for _ in range(self.number):    
            yield self.q.get()
        self._join()

    
        

class TextIter(mx.io.DataIter):
    def __init__(self, number, batch_size, threads):
        super(TextIter, self).__init__()
        self.gen = GenImage()
        self.batch_size = batch_size
        self.provide_data = [('data', (self.batch_size, 1, HEIGHT, WIDTH))]
        self.provide_label =  [('label', (self.batch_size, NUM_LABEL))]
        self.provide_bbox = [('bbox_target', (self.batch_size, 1, 1, 8))]
        
        def gen_batch_helper():
            return self._gen_batch()
    
        self.prefetch = Prefetch(number, threads, gen_batch_helper)    
    
    def _gen_batch(self):
        data = []
        label = []
        bbox = []
        for i in range(self.batch_size):    
            img, lb = self.gen.generate()
            box = []
            img = img[np.newaxis, :, :]
            img = np.multiply(img, 1/255.0)
            data.append(img)
            label.append(lb)
            bbox.append(box)
        data_all = [np.array(data)]
        label_all = [np.array(label)]
        bbox_all = [np.array(bbox)]
        data_names = ['data']
        label_names = ['label']
        bbox_names = ['bbox_target']
        data_batch = SimpleBatch(data_names, data_all, label_names, label_all, bbox_names, bbox_all)
        return data_batch
            
def Accuracy(label, cls_preds, bbox_target, loc_preds):
    pass

def parse_args():
    parser = argparse.ArgumentParser(description='Train a direct regression multi-oriented neural network')
    parser.add_argument('--gpus', dest='gpus', default='0', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    num_epoch = 0
    batch_size = 128
    train_from_scratch = True
    learning_rate = 0.01
    weight_decay = 4e-4
    momentum = 0.9
    prefix_out = './model/text-location'

    prefetch_thread = 1

    #context list
    #ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]  
    #ctx = [mx.cpu()] if not ctx else ctx
    ctx = [mx.cpu()]

    
    #prepare data
    data_train = TextIter(1000, batch_size, prefetch_thread)
    data_val = TextIter(50, batch_size, prefetch_thread)
    
    print 'generating !'
    #build a NN
    sym = rfcn()
    print sym
    
    if train_from_scratch:
        model = mx.model.FeedForward(ctx=ctx,
                                     symbol=sym,
                                     num_epoch=num_epoch,
                                     learning_rate=learning_rate,
                                     momentum=momentum,
                                     wd=weight_decay,
                                     initializer=mx.init.Xavier(factor_type="in", magnitude=2.34)) 
    import logging
    head = "%(asctime)-15s %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=head)

    model.fit(X=data_train, eval_data=data_val, 
              eval_metrix=mx.metric.np(Accuracy),
              batch_end_callback=mx.callback.Speedometer(batch_size, 20),
              epoch_end_callback=mx.callback.do_checkpoint(prefix_out))
    data_train.join()
    data_val.join()
