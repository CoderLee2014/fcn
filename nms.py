import numpy as np


def nms(dets, thresh=0.5, mode="Union"):
    """
    greedily select boxes with high confidence
    keep boxes overlap <= thresh
    rule out overlap > thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap <= thresh
    :return: indexes to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "Union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
   
    #step 2: filter the word space 
    inds = range(len(x1))
    keep_ori = keep
    for k in keep_ori:
        inds_exp = list(set(inds) - set([k]))
        xx1 = np.maximum(x1[k], x1[inds_exp])
        yy1 = np.maximum(y1[k], y1[inds_exp])
        xx2 = np.minimum(x2[k], x2[inds_exp])
        yy2 = np.minimum(y2[k], y2[inds_exp])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[k] + areas[inds_exp] - inter)
        ind_max = np.argmax(ovr)
        if ovr[ind_max] > thresh:
            keep.append(inds_exp[ind_max])

    #step 3: merge 
    retain = []
    for i in range(len(keep) - 1):
        xx1 = np.maximum(x1[keep[i]], x1[keep[i+1:]])
        yy1 = np.maximum(y1[keep[i]], y1[keep[i+1:]])
        xx2 = np.maximum(x2[keep[i]], x2[keep[i+1:]])
        yy2 = np.maximum(y2[keep[i]], y2[keep[i+1:]])

         
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[keep[i]] + areas[keep[i+1:]] - inter)
        inds = np.where(ovr<0.2)[0]
        for j in inds:
            retain.append(keep[i+1+j])
    return dets[retain]
