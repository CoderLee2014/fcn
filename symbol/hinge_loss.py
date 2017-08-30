import mxnet as mx
import numpy as np
from config.config import cfg

class HingeLossOperator(mx.operator.CustomOp):
    def __init__(self, grad_scale=1.0):
        super(HingeLossOperator, self).__init__()
        self.grad_scale = grad_scale

    def forward(self, is_train, req, in_data, out_data, aux):
        cls_pred = in_data[0].asnumpy().reshape(-1, 1) # batchsize x 1 x 80 x 80 => (-1, 1)
        cls_keep = in_data[1].asnumpy().reshape(-1, 1) # batchsize x 1 x 80 x 80
        label = in_data[2].asnumpy().reshape(-1, 1) # batchsize x 1 x 80 x 80

        # compute hinge loss
        keep = np.where(cls_keep == 0)[0]
        cls_pred[keep, :] = 0
        cls_loss = 2 * np.sign(0.5 - label)*(cls_pred - label)
        cls_loss[cls_loss < 0] = 0
        cls_loss[np.where(label == 1)[0], :] *= -1
        cls_loss = cls_loss / 320.0/ 320.0 
        self.assign(out_data[0], req[0], mx.nd.array(cls_loss.reshape(in_data[0].shape)))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], out_data[0] * self.grad_scale)

@mx.operator.register("hingeloss")
class HingeLossProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(HingeLossProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['cls_pred', 'mask', 'label']

    def list_outputs(self):
        return ['cls_loss']

    def infer_shape(self, in_shape):
        print in_shape
        out_shape = (1, )
        return in_shape, [in_shape[0]] 

    def create_operator(self, ctx, shapes, dtypes):
        return HingeLossOperator()
