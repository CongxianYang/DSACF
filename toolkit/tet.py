import cv2
import numpy as np

image = cv2.imread('/home/xiancong/Data_set/RGBT234/jump/visible/01464v.jpg')
bbox1 = [482,191,528,322]#425.3216042643343, y1=184.6931439053289, x2=462.6783957356657, y2=277.3068560946711)
    #label1 = 'anno'
    # picture_path为图片路径;(cv读取的文件为BGR形式)
cv2.rectangle(image, (bbox1[0], bbox1[1]), (bbox1[2], bbox1[3]), (0, 0, 255), 3)
cv2.imwrite('/home/xiancong/桌面/cv_drwn.jpg',image)


