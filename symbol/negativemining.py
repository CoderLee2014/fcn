import mxnet as mx
import numpy as np
from config.config import cfg

class NegativeMiningOperator(mx.operator.CustomOp):
    def __init__(self, cls_ohem=cfg.CLS_OHEM, cls_ohem_ratio=cfg.CLS_OHEM_RATIO,
            bbox_ohem=cfg.BBOX_OHEM, bbox_ohem_ratio=cfg.BBOX_OHEM_RATIO, 
            cls_thresh=cfg.CLS_THRESH, overlap_ratio=cfg.OVERLAP_RATIO):
        super(NegativeMiningOperator, self).__init__()
        self.cls_ohem = cls_ohem
        self.cls_ohem_ratio = cls_ohem_ratio
        self.bbox_ohem = bbox_ohem
        self.bbox_ohem_ratio = bbox_ohem_ratio
        self.cls_thresh = cls_thresh
        self.overlap_ratio = overlap_ratio

    def forward(self, is_train, req, in_data, out_data, aux):
        cls_prob = in_data[0].asnumpy() # batchsize x 1 x 80 x 80
        bbox_pred = in_data[1].asnumpy() # batchsize x 8 x 80 x 80
        label = in_data[2].asnumpy() # batchsize x 1 x 80 x 80
        bbox_target = in_data[3].asnumpy()

        #print "cls_prob:", len(cls_prob[cls_prob>0.7])
        self.assign(out_data[0], req[0], in_data[0])
        self.assign(out_data[1], req[1], in_data[1])

        # cls task
        cls_prob = cls_prob.reshape(-1, 1)
        label = label.reshape(-1, 1)
        cls_keep = np.zeros(cls_prob.shape[0])
        valid_inds = np.where(label > 0)[0] #using all positive samples
        cls_keep[valid_inds] = 1

        # Hard negative mining
        valid_neg_inds = np.where(label < 1)[0]
        cls_neg = cls_prob[valid_neg_inds, :]
        cls_neg = (cls_neg) + cfg.EPS
        log_loss = np.log(cls_neg)
        keep = np.argsort(cls_neg)[::-1][:len(valid_inds)*2]
        cls_keep[valid_neg_inds[keep]] = 1
        self.assign(out_data[2], req[2], mx.nd.array(cls_keep))


        # bbox
        bbox_keep = np.zeros((cls_prob.shape[0], 8))
        bbox_keep[valid_inds] = 1
        self.assign(out_data[3], req[3], mx.nd.array(bbox_keep.reshape(bbox_pred.shape)))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        cls_pred = out_data[0].asnumpy().reshape(-1, 1)
        bbox_pred = out_data[1].asnumpy().reshape(-1, 8)
        cls_keep = out_data[2].asnumpy()
        bbox_keep = out_data[3].asnumpy().reshape(-1, 8)
        label = in_data[2].asnumpy().reshape(-1,1)

        keep = np.where(cls_keep == 0)[0]
        cls_pred[keep, :] = 0
        cls_grad = 2 * np.sign(0.5 - label)*(cls_pred - label)
        cls_grad[cls_grad<0] = 0
        cls_grad[np.where(label == 1)[0], :] *= -1
        cls_grad = cls_grad / 320.0 / 320

        bbox_grad = bbox_keep /(len(np.where(bbox_keep == 1)[0]))
        self.assign(in_grad[0], req[0], mx.nd.array(cls_grad.reshape(in_data[0].shape)))
        #self.assign(in_grad[1], req[1], mx.nd.array(bbox_grad.reshape(in_data[1].shape)))

@mx.operator.register("negativemining")
class NegativeMiningProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(NegativeMiningProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['cls_prob', 'bbox_pred', 'label', 'bbox_target']

    def list_outputs(self):
        return ['cls_out', 'bbox_out', 'cls_keep', 'bbox_keep']

    def infer_shape(self, in_shape):
        print in_shape
        keep_shape = (in_shape[0][0]*in_shape[0][2]*in_shape[0][3], )
        return in_shape, [in_shape[0], in_shape[1], keep_shape, in_shape[1]]

    def create_operator(self, ctx, shapes, dtypes):
        return NegativeMiningOperator()
