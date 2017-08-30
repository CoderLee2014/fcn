import mxnet as mx
import numpy as np
import cv2
from tools.image_processing import resize, transform
from tools.rand_sampler import RandSampler


class DetRecordIter(mx.io.DataIter):
    """
    The new detection iterator wrapper for mx.io.ImageDetRecordIter which is
    written in C++, it takes record file as input and runs faster.
    Supports various augment operations for object detection.
    Parameters:
    -----------
    path_imgrec : str
        path to the record file
    path_imglist : str
        path to the list file to replace the labels in record
    batch_size : int
        batch size
    data_shape : tuple
        (3, height, width)
    label_width : int
        specify the label width, use -1 for variable length
    label_pad_width : int
        labels must have same shape in batches, use -1 for automatic estimation
        in each record, otherwise force padding to width in case you want t
        rain/validation to match the same width
    label_pad_value : float
        label padding value
    resize_mode : str
        force - resize to data_shape regardless of aspect ratio
        fit - try fit to data_shape preserving aspect ratio
        shrink - shrink to data_shape only, preserving aspect ratio
    mean_pixels : list or tuple
        mean values for red/green/blue
    kwargs : dict
        see mx.io.ImageDetRecordIter
    Returns:
    ----------
    """
    def __init__(self, path_imgrec, batch_size, data_shape, path_imglist="",
                 mean_pixels=[128, 128, 128],
                 **kwargs):
        super(DetRecordIter, self).__init__()
        if isinstance(data_shape, int):
            data_shape = (3,data_shape, data_shape)
        print(path_imgrec)
        label_width = 80*80*9
        self.rec = mx.io.ImageRecordIter(
            path_imgrec     = path_imgrec,
            path_imglist    = path_imglist,
            label_width     = label_width,
            batch_size      = batch_size,
            data_shape      = data_shape,
            mean_r          = mean_pixels[0],
            mean_g          = mean_pixels[1],
            mean_b          = mean_pixels[2],
            shuffle=True,
            **kwargs)

        self.provide_label = None
        self._get_batch()
        if not self.provide_label:
            raise RuntimeError("Invalid ImageDetRecordIter: " + path_imgrec)
        self.reset()

    @property
    def provide_data(self):
        return self.rec.provide_data

    def reset(self):
        self.rec.reset()

    def iter_next(self):
        return self._get_batch()

    def next(self):
        if self.iter_next():
            return self._batch
        else:
            raise StopIteration

    def _get_batch(self):
        self._batch = self.rec.next()
        if not self._batch:
            return False

        if self.provide_label is None:
            # estimate the label shape for the first batch, always reshape to n*5
            first_label = self._batch.label[0][0].asnumpy()
            self.batch_size = self._batch.label[0].shape[0]
            self.label_shape = (self.batch_size, 9, 80, 80)
            self.provide_label = [('label', self.label_shape)]

        # modify label
        label = self._batch.label[0].asnumpy()
        label = label.reshape(self.batch_size, 80, 80, 9).transpose(0, 3, 1, 2)
        self._batch.label = [mx.nd.array(label)]
        return True


class DetIter(mx.io.DataIter):
    """
    Detection Iterator, which will feed data and label to network
    Optional data augmentation is performed when providing batch

    Parameters:
    ----------
    imdb : Imdb
        image database
    batch_size : int
        batch size
    data_shape : int or (int, int)
        image shape to be resized
    shuffle: bool
        whether to shuffle initial image list, default False
    mean_pixels: float or float list
        [R, G, B], mean pixel values
    is_train : bool
        whether in training phase, default True, if False, labels might
        be ignored
    """
    def __init__(self, imdb, batch_size, data_shape, shuffle=False, \
                 mean_pixels=[128, 128, 128], \
                 is_train=True):
        super(DetIter, self).__init__()

        self._imdb = imdb
        self.batch_size = batch_size
        if isinstance(data_shape, int):
            data_shape = (data_shape, data_shape)
        self._data_shape = data_shape
        self._mean_pixels = mean_pixels
        self.is_train = is_train

        self._current = 0
        self._size = imdb.num_images
        self._index = np.arange(self._size)

        self._data = None
        self._label = None
        self._shuffle = shuffle
    
        self._get_batch()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in self._data.items()]

    @property
    def provide_label(self):
        if self.is_train:
            return [(k, v.shape) for k, v in self._label.items()]
        else:
            return []

    def reset(self):
        self._current = 0
        if self._shuffle:
            np.random.shuffle(self._index)

    def iter_next(self):
        return self._current < self._size

    def next(self):
        if self.iter_next():
            self._get_batch()
            data_batch = mx.io.DataBatch(data=self._data.values(),
                                   label=self._label.values(),
                                   pad=self.getpad(), index=self.getindex())
            self._current += self.batch_size
            return data_batch
        else:
            raise StopIteration

    def getindex(self):
        return self._current // self.batch_size

    def getpad(self):
        pad = self._current + self.batch_size - self._size
        return 0 if pad < 0 else pad

    def _get_batch(self):
        """
        Load data/label from dataset
        """
        batch_data = []
        batch_label = []
        batch_bbox = []
        for i in range(self.batch_size):
            if (self._current + i) >= self._size:
                if not self.is_train:
                    continue
                # use padding from middle in each epoch
                idx = (self._current + i + self._size // 2) % self._size
                index = self._index[idx]
            else:
                index = self._index[self._current + i]
            # index = self.debug_index
            im_path = self._imdb.image_path_from_index(index)
            img = cv2.imread(im_path)
            #transform into mxnet tensor
            img = img.copy()
            img = cv2.resize(img, (320, 320))
            img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
            img = img.astype(float)
            img -= self._mean_pixels
            channel_swap = (2, 0, 1)
            data = img.transpose(channel_swap)
            label, loc_target = self._imdb.label_from_index(index) if self.is_train else (None, None) #(80,80) (80,80,9)
            #data, label = self._data_augmentation(img, gt)
            batch_data.append(data)
            if self.is_train:
                batch_label.append(label)
                batch_bbox.append(loc_target)
        self._data = {'data': mx.nd.array(np.array(batch_data))}
        if self.is_train:
            self._label = {'label': mx.nd.array(np.array(batch_label).transpose(0, 3, 1, 2)), 'loc_target': mx.nd.array(np.array(batch_bbox).transpose(0, 3, 1, 2))}
        else:
            self._label = {'label': None, 'loc_target': None}

