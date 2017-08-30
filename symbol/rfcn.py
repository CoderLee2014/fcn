import mxnet as mx
import negativemining
import hinge_loss
from config import config
from nms import nms

eps = 1e-10 + 1e-5
bn_mom = 0.9
fix_gamma = False


def filter_map(kernel=1, stride=1, pad=0):
    return (stride, (kernel-stride)/2-pad)

def compose_fp(fp_first, fp_second):
    return (fp_first[0]*fp_second[0], fp_first[0]*fp_second[1]+fp_first[1])

def compose_fp_list(fp_list):
    fp_out = (1.0, 0.0)
    for fp in fp_list:
        fp_out = compose_fp(fp_out, fp)
    return fp_out

def inv_fp(fp_in):
    return (1.0/fp_in[0], -1.0*fp_in[1]/fp_in[0])

def offset():
    pass    
    
def ConvUnit(data, kernel, num_filter,name, pad=(1, 1)):
    conv = mx.symbol.Convolution(
        data=data, kernel=kernel, num_filter=num_filter, name=name, pad=pad)
    bn = mx.symbol.BatchNorm(data=conv, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom)
    relu = mx.symbol.Activation(data=bn, act_type="relu")
    
    return relu
 

def rfcn(mode="train"):
    data = mx.sym.Variable('data')
    label = mx.symbol.Variable(name="label")
    
    conv1 = ConvUnit(data, kernel=(5, 5), num_filter=32, name='conv1')
    pool1 = mx.symbol.Pooling(
        data=conv1, pool_type="max", kernel=(2, 2), stride=(2, 2))
    
    conv2 = ConvUnit(data=pool1, kernel=(3, 3), num_filter=64, name='conv2')
    conv3 = ConvUnit(data=conv2, kernel=(3, 3), num_filter=64, name='conv3')
    pool2 = mx.symbol.Pooling(
        data=conv3, pool_type="max", kernel=(2, 2), stride=(2, 2))
    

    conv4 = ConvUnit(data=pool2, kernel=(3, 3), num_filter=128, name='conv4')
    conv5 = ConvUnit(data=conv4, kernel=(3, 3), num_filter=128, name='conv5')
    pool3 = mx.symbol.Pooling(
        data=conv5, pool_type="max", kernel=(2, 2), stride=(2, 2))
    
    conv6 = ConvUnit(data=pool3, kernel=(3, 3), num_filter=256, name='conv6')
    conv7 = ConvUnit(data=conv6, kernel=(3, 3), num_filter=256, name='conv7')
    pool4 = mx.symbol.Pooling(
        data=conv7, pool_type="max", kernel=(2, 2), stride=(2, 2))
    

    conv8 = ConvUnit(data=pool4, kernel=(3, 3), num_filter=512, name='conv8')
    conv9 = ConvUnit(data=conv8, kernel=(3, 3), num_filter=512, name='conv9')
    pool5 = mx.symbol.Pooling(
        data=conv9, pool_type="max", kernel=(2, 2), stride=(2, 2))


    conv10 = ConvUnit(data=pool5, kernel=(3, 3), num_filter=512, name='conv10')
    conv11 = ConvUnit(data=conv10, kernel=(3, 3), num_filter=512, name='conv11')
    pool6 = mx.symbol.Pooling(
        data=conv11, pool_type="max", kernel=(2, 2), stride=(2, 2))

    conv12 = ConvUnit(data=pool6, kernel=(1, 1), num_filter=128, name='conv12')
    upsample1 = mx.symbol.Deconvolution(data=conv12, kernel=(2, 2), stride=(2, 2), num_filter=128, target_shape=(10, 10), name='upsample1')
    
    #upsample1 + pool5
    branch3 = ConvUnit(data=pool5, kernel=(1, 1), num_filter=128, name='branch3')
    pool5c = mx.symbol.Crop(*[branch3, upsample1], name='pool5c')
    fused1 = upsample1 + pool5c
    upsample2 = mx.symbol.Deconvolution(data=fused1, kernel=(2, 2), stride=(2, 2), num_filter=128, target_shape=(20, 20))

    #upsample2 + pool4
    branch2 = ConvUnit(data=pool4, kernel=(1, 1), num_filter=128, name='branch2')
    pool4c = mx.symbol.Crop(*[branch2, upsample2],  name='pool4c')
    fused2 = upsample2 + pool4c
    upsample3 = mx.symbol.Deconvolution(data=fused2, kernel=(2, 2), stride=(2, 2), num_filter=128, target_shape=(40, 40))

    #upsample3 + pool3
    branch1 = ConvUnit(data=pool3, kernel=(1, 1), num_filter=128, name='branch1')
    pool3c = mx.symbol.Crop(*[branch1, upsample3], name='pool3c')
    fused3 = upsample3 + pool3c
    upsample4 = mx.symbol.Deconvolution(data=fused3, kernel=(2, 2), stride=(2, 2), num_filter=128, target_shape=(80, 80))

    
    #classification task:
    cls_conv = mx.symbol.Convolution(data=upsample4, kernel=(1, 1), num_filter=1, name="cls_prob")
    #cls_pred_reshape = mx.sym.reshape(data=cls_conv, shape=(-3, -2, 1))
    #cls_prob = mx.symbol.SVMOutput(data=cls_pred_reshape, label=mx.sym.reshape(data=label, shape=(-3,-2)), margin=0.5, name="cls_prob")

    #regression task:
    conv_reg = ConvUnit(data=upsample4, kernel=(1, 1), num_filter=128, name='reg_conv1', pad=(0, 0))
    conv_reg = mx.symbol.Convolution(data=conv_reg, kernel=(1, 1), num_filter=8, name='reg_conv2')


    bbox_pred = mx.symbol.sigmoid(data=conv_reg, name='bbox_pred')
    bbox_pred = 80 * bbox_pred - 40

    # negative mining 
    dets = mx.symbol.Custom(cls_prob=cls_conv, label=label,
                        op_type='negativemining', name='negativemining')
    cls_keep =  mx.sym.stop_gradient(dets[0])
    bbox_keep = dets[1]
    loc_target = dets[2]
    cls_target = dets[3]

    if mode == "test":
        group = mx.symbol.Group([cls_conv, conv_reg])
    else:
        cls_loss = mx.symbol.Custom(cls_pred=cls_conv, mask=cls_keep, label=cls_target, op_type="hingeloss", name="hingeloss")

        loc_loss_ = mx.symbol.smooth_l1(data=(bbox_pred * bbox_keep - loc_target), scalar=1., name="loc_loss_")
        loc_loss = mx.symbol.MakeLoss(data=loc_loss_, name="loc_loss", grad_scale=0.01, normalization="valid")

        # monitoring training status
        cls_prob = mx.sym.MakeLoss(data=cls_conv, grad_scale=0, name="cls_preds")

        group = mx.symbol.Group([cls_prob, cls_keep, loc_loss, cls_loss])
    
    return group

def get_symbol(nms_thresh=0.5, force_suppress=True):
    """
    This is the detection network
    Parameters:
    ----------
    num_classes: int
        number of object classes not including background
    nms_thresh : float
        threshold of overlap for non-maximum suppression
    Returns:
    ----------
    mx.Symbol
    """
    net = rfcn()
    # print net.get_internals().list_outputs()
    cls_preds = net.get_internals()["cls_prob_output"]
    bbox_pred = net.get_internals()['bbox_pred_output']

    return mx.symbol.Group([cls_preds, bbox_pred])


