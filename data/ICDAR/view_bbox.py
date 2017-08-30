import cv2
import numpy as np

image_path = './icdar2015_train/img_43.jpg'
label_file = './icdar2015_train_GT/gt_img_43.txt'

resize = True

img = cv2.imread(image_path).copy()
#img = img[:320, :320]
if resize:
    img = cv2.resize(img, (80, 80))
for line in open(label_file):
    line = line.decode('utf-8-sig').encode('utf-8').strip().split(',')
    pts = np.array([[int(line[0]), int(line[1])],
                    [int(line[2]), int(line[3])],
                    [int(line[4]), int(line[5])],
                    [int(line[6]), int(line[7])]])
    if resize:
        pts[:, 0] = pts[:, 0]/4.0
        pts[:, 1] = pts[:, 1] /(720.0/320)
        pts[:, :] = pts[:, :] / 4.0
    cv2.polylines(img, [pts], True, (0, 255, 255))
cv2.imwrite('bbox-img.jpg', img)

