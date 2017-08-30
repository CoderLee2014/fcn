import mxnet as mx
import numpy as np
import cv2
from tools.image_processing import resize, transform
from tools.rand_sampler import RandSampler

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

