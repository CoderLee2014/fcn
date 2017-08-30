import mxnet as mx
import numpy as np
from config import config


class Metrics(mx.metric.EvalMetric):
    def __init__(self):
        super(Metrics, self).__init__(['Accuracy', 'loc_loss', 'cls_loss'], 3)

    def update(self, labels, preds):
        """
         preds: 
                cls_prob, batch_size x 1 x 80 x 80 
                cls_keep, batch_size x 1 x 80 x 80
                loc_loss, float
        # label: batch_size x 9 x 80 x 80
        """
        cls_pred = preds[0].asnumpy().reshape(-1, 1)
        cls_keep = preds[1].asnumpy().reshape(-1, 1)
        loc_loss = preds[2].asnumpy()
        cls_loss = preds[3].asnumpy()

        #mask = np.where(cls_mask[:,0, :, :] > 0) 
        label = labels[0].asnumpy()[:, [0], :, :].reshape(-1, 1)

        cls_pred[cls_pred < 0.7] = 0
        cls_pred[cls_pred >= 0.7] = 1
        keep = np.where(cls_keep >0)[0]
        cls_pred = cls_pred[keep]
        label = label[keep]

        self.sum_metric[0] += (cls_pred.flat == label.flat).sum()
        #self.num_inst[0] += len(cls_pred.flat)
        self.num_inst[0] += len(cls_pred.flat)

        self.sum_metric[1] += np.sum(loc_loss)
        self.num_inst[1] += (cls_pred.shape[0])


        self.sum_metric[2] += np.sum(cls_loss)
        self.num_inst[2] += (cls_pred.shape[0])

    def get(self):
        """
        Get the current evaluation result.
        Override the default behavior
        Returns
        -------
        name : str
        Name of the metric.
        value : float
        Value of the evaluation.
        """
        if self.num is None:
            if self.num_inst == 0:
                return (self.name, float('nan'))
            else:
                return (self.name, self.sum_metric / self.num_inst)
        else:
            names = ['%s'%(self.name[i]) for i in range(self.num)]
            values = [x / y if y != 0 else float('nan') \
                for x, y in zip(self.sum_metric, self.num_inst)]
        return (names, values) 
