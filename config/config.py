import os
from easydict import EasyDict as edict
from tools.rand_sampler import RandCropper, RandPadder
import numpy as np

cfg = edict()
cfg.ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')


cfg.BATCH_SIZE = 128

cfg.CLS_OHEM = True
cfg.CLS_OHEM_RATIO = 0.7
cfg.BBOX_OHEM = False
cfg.BBOX_OHEM_RATIO = 0.7
cfg.CLS_THRESH = 0.7
cfg.OVERLAP_RATIO = 0.5

cfg.EPS = np.finfo(np.float32).eps
cfg.LR_EPOCH = [8, 14]
# training
cfg.TRAIN = edict()
cfg.TRAIN.RAND_SAMPLERS = [RandCropper(min_scale=1., max_trials=1, max_sample=1),
    RandCropper(min_scale=.3, min_aspect_ratio=.5, max_aspect_ratio=2., min_overlap=.1),
    RandCropper(min_scale=.3, min_aspect_ratio=.5, max_aspect_ratio=2., min_overlap=.3),
    RandCropper(min_scale=.3, min_aspect_ratio=.5, max_aspect_ratio=2., min_overlap=.5),
    RandCropper(min_scale=.3, min_aspect_ratio=.5, max_aspect_ratio=2., min_overlap=.7),
    RandPadder(max_scale=2., min_aspect_ratio=.5, max_aspect_ratio=2., min_gt_scale=.05),
    RandPadder(max_scale=3., min_aspect_ratio=.5, max_aspect_ratio=2., min_gt_scale=.05),
    RandPadder(max_scale=4., min_aspect_ratio=.5, max_aspect_ratio=2., min_gt_scale=.05),]
# cfg.TRAIN.RAND_SAMPLERS = []
cfg.TRAIN.RAND_MIRROR = True
cfg.TRAIN.INIT_SHUFFLE = True
cfg.TRAIN.EPOCH_SHUFFLE = True # shuffle training list after each epoch
cfg.TRAIN.RAND_SEED = None
cfg.TRAIN.RESIZE_EPOCH = 1 # save model every N epoch


# validation
cfg.VALID = edict()
cfg.VALID.RAND_SAMPLERS = []
cfg.VALID.RAND_MIRROR = True
cfg.VALID.INIT_SHUFFLE = True
cfg.VALID.EPOCH_SHUFFLE = True
cfg.VALID.RAND_SEED = None
