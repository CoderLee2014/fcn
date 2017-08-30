import cv2
import numpy as np

def isInsideGT((x,y), box):
        res = False
        i = -1
        l = 4 # 4points
        j = l - 1
        minx = box[::2].min()
        maxx = box[::2].max()
        miny = box[1::2].min()
        maxy = box[1::2].max()
        if (x < minx or x > maxx or y < miny or y > maxy):
            return False
        while i < l - 1:
            i += 1
            if ((box[2*i]<=x and x<box[2*j]) or (box[2*j]<=x and x<box[2*i])):
                if (y<(box[2*j+1]-box[2*i+1])*(x-box[2*i])*1.0/(box[2*j]-box[2*i])+box[2*i+1]):
                    res = not res
            j = i
        return res

def labeling(width, height, gt_boxes):
        label = np.zeros((height, width, 1))
        img = np.ones((height, width, 3))*255
        loc_target = np.zeros((height, width, 8))
        for y in range(height):
            for x in range(width):
                for box in gt_boxes:
                    if isInsideGT((x,y), box):
                        label[y, x, :] = 1
                        img[y,x,:] = 0
                        loc_target[y, x, :] = [box[0] - x, box[1] - y,
                                              box[2] - x, box[3] - y,
                                              box[4] - x, box[5] - y,
                                              box[6] - x, box[7] - y]
        cv2.imwrite('labeled.jpg',img)

image_path = './icdar2015_train/img_43.jpg'
label_file = './icdar2015_train_GT/gt_img_43.txt'

img = cv2.imread(image_path).copy()
if_resize = True

gt_boxes = []
print "Start to load bboxes..."
for line in open(label_file):
    line = line.decode('utf-8-sig').encode('utf-8').strip().split(',')
    box = [int(elem) for elem in line[:8]]
    box = np.array(box)
    if if_resize:
        box[::2] = box[::2]/ 4.0
        box[1::2] = box[1::2]/ (720.0/320)
        box = box / 4.0
    gt_boxes.append(box)

print "start to label each pixel..."
if if_resize:
    labeling(80, 80, gt_boxes)
else:
    labeling(img.shape[1], img.shape[0], gt_boxes)

