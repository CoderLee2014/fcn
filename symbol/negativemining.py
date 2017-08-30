import mxnet as mx
import numpy as np
from config.config import cfg
import random

class NegativeMiningOperator(mx.operator.CustomOp):
    def __init__(self, ratio=0.2):
        super(NegativeMiningOperator, self).__init__()
        self.ratio = ratio

    def forward(self, is_train, req, in_data, out_data, aux):
        cls_prob = in_data[0].asnumpy() # batchsize x 1 x 80 x 80
        label = in_data[1].asnumpy()[:, [0], :, :] # batchsize x 1 x 80 x 80
        loc_target = in_data[1].asnumpy()[:, 1:, :, :]

        cls_shape = cls_prob.shape
        self.assign(out_data[3], req[3], mx.nd.array(label))

        # cls task
        cls_prob = cls_prob.reshape(-1, 1)
        label = label.reshape(-1, 1)
        cls_keep = np.ones(cls_prob.shape[0])
        valid_inds = np.where(label > 0)[0] #using all positive samples
        cls_keep[valid_inds] = 1

        # Hard negative mining
        valid_neg_inds = np.where(label < 1)[0]
        cls_neg = cls_prob[valid_neg_inds, :]
        cls_neg[cls_neg < 0] = 0
        cls_neg = (cls_neg) + cfg.EPS
        np.nan_to_num(cls_neg)
        cls_neg = np.log(cls_neg)
        hard_num = int(len(valid_inds) * 3 * self.ratio)
        keep = np.argsort(cls_neg)[::-1][:hard_num]
        keep2 = np.argsort(cls_neg)[::-1][hard_num:]
        keep2 = random.sample(keep2, int((1 - self.ratio)*len(valid_inds)*3))
        cls_keep[valid_neg_inds[keep]] = 1
        cls_keep[valid_neg_inds[keep2]] = 1
        self.assign(out_data[0], req[0], mx.nd.array(cls_keep.reshape(cls_shape)))


        # bbox
        bbox_keep = np.zeros((cls_prob.shape[0], 8))
        bbox_keep[valid_inds] = 1
        bbox_keep = bbox_keep.reshape((cls_shape[0], cls_shape[2], cls_shape[3], 8)).transpose(0, 3, 1, 2)
        self.assign(out_data[1], req[1], mx.nd.array(bbox_keep))

        self.assign(out_data[2], req[2], mx.nd.array(loc_target))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        grad = mx.nd.zeros(in_data[0].shape)
        self.assign(in_grad[0], req[0], grad)

@mx.operator.register("negativemining")
class NegativeMiningProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(NegativeMiningProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['cls_prob', 'label']

    def list_outputs(self):
        return ['cls_keep', 'bbox_keep', 'loc_target', 'label']

    def infer_shape(self, in_shape):
        print in_shape
        bbox_shape = (in_shape[0][0], 8, in_shape[0][2], in_shape[0][3])
        return in_shape, [in_shape[0], bbox_shape, bbox_shape, in_shape[0]]

    def create_operator(self, ctx, shapes, dtypes):
        return NegativeMiningOperator()
     
