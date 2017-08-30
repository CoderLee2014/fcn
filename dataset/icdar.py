from __future__ import print_function
import os
import numpy as np
from imdb import Imdb
import cv2
import re


class ICDAR(Imdb):
    """
    Implementation of Imdb for ICDAR dataset

    Parameters:
    ----------
    image_set : str
        set to be used, can be train, val, trainval, test
    year : str
        year of dataset, can be 2007, 2010, 2012...
    devkit_path : str
        devkit path of VOC dataset
    shuffle : boolean
        whether to initial shuffle the image list
    is_train : boolean
        if true, will load annotations
    """
    def __init__(self, image_set, devkit_path, shuffle=False, is_train=False):
        super(ICDAR, self).__init__('ICDAR_' + image_set)
        self.image_set = image_set
        self.devkit_path = devkit_path
        self.data_path = os.path.join(devkit_path, 'ICDAR')
        self.extension = '.jpg'
        self.is_train = is_train

        self.classes = ['text', ]

        self.config = {#'use_difficult': True,
                       'comp_id': 'comp4',
                       'padding': 157}
        self.num_classes = len(self.classes)
        self.image_set_index = self._load_image_set_index(shuffle)
        self.num_images = len(self.image_set_index)
        if is_train:
            self.labels, self.loc_targets = self._load_image_labels()


    def _load_image_set_index(self, shuffle):
        """
        find out which indexes correspond to given image set (train or val)

        Parameters:
        ----------
        shuffle : boolean
            whether to shuffle the image list
        Returns:
        ----------
        entire list of images specified in the setting
        """
        image_set_index_file = os.path.join(self.data_path, self.image_set + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file) as f:
            image_set_index = [x.strip() for x in f.readlines()]
        if shuffle:
            np.random.shuffle(image_set_index)
        return image_set_index
    
    def image_path_from_index(self, index):
        """
        given image index, find out full path

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        full path of this image
        """
        assert self.image_set_index is not None, "Dataset not initialized"
        name = self.image_set_index[index]
        image_file = os.path.join(self.data_path, self.image_set, name + self.extension)
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file
    

    def _image_path_from_index(self, index):
        """
        given image index, find out annotation path

        Parameters:
        ----------
        index: int
            index of a specific image

        Returns:
        ----------
        full path of annotation file
        """
        image_file = os.path.join(self.data_path, '_', self.image_set, index + self.extension)
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def label_from_index(self, index):
        """
        given image index, return preprocessed ground-truth

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        ground-truths of this image
        """
        assert self.labels is not None, "Labels not processed"
        return self.labels[index], self.loc_targets[index] 
    def _label_path_from_index(self, index):
        """
        given image index, find out annotation path

        Parameters:
        ----------
        index: int
            index of a specific image

        Returns:
        ----------
        full path of annotation file
        """
        label_file = os.path.join(self.data_path, self.image_set + '_GT', 'gt_' + index + '.txt')
        assert os.path.exists(label_file), 'Path does not exist: {}'.format(label_file)
        return label_file

    def _load_image_labels(self):
        """
        preprocess all ground-truths

        Returns:
        ----------
        labels packed in [num_images x max_num_objects x 9] tensor
        """
        labels = []
        loc_targets = []
        # load ground-truth from xml annotations
        for idx in self.image_set_index:
            #image_file = self._image_path_from_index(idx)
            #height, width, channels = cv2.imread(image_file).shape
            label_file = self._label_path_from_index(idx)
            gt_boxes = []
            for line in open(label_file):
                line = line.decode('utf-8-sig').encode('utf-8').strip().split(',')
                cls_id = self.classes.index('text')
                box = []
                for i in range(8):
                    if i % 2 == 0:
                        box.append(float(line[i])/4.0)
                    else: 
                        box.append(float(line[i])/(720.0/320))
                #box = [int(elem) for elem in line[:8]]
                gt_boxes.append(np.array(box)/4.0)
            label, loc_target = self.labeling(80, 80, gt_boxes, idx)
            labels.append(label)
            loc_targets.append(loc_target)
        
        return np.array(labels), np.array(loc_targets)

    def labeling(self, width, height, gt_boxes, index):
        label = np.zeros((height, width, 1)) 
        #img = np.ones((height, width, 3))*255
        loc_target = np.zeros((height, width, 8))
        for y in range(height):
            for x in range(width):
                for box in gt_boxes:
                    if self.isInsideGT((x,y), box):
                        label[y, x, :] = 1
                        #img[y,x,:] = 0
                        loc_target[y, x, :] = [box[0] - x, box[1] - y,
                                              box[2] - x, box[3] - y,
                                              box[4] - x, box[5] - y,
                                              box[6] - x, box[7] - y]
        #cv2.imwrite(index+'.jpg',img)
        return label, loc_target

                      
    def isInsideGT(self, (x,y), box):
        res = False
        i = -1
        l = 4 # 4points
        j = l - 1
        minx = box[::2].min()
        maxx = box[::2].max()
        miny = box[1::2].min()
        maxy = box[1::2].max()
        if x < minx or x >maxx or y < miny or y > maxy:
            return False
        while i < l - 1:
            i += 1
            if ((box[2*i]<=x and x<box[2*j]) or (box[2*j]<=x and x<box[2*i])):
                if (y<(box[2*j+1]-box[2*i+1])*(x-box[2*i])/(box[2*j]-box[2*i])+box[2*i+1]):
                    res = not res
            j = i
        return res

        
        
