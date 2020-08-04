# -*- coding: utf-8 -*-
# @Time : 2020/7/22 17:36
# @Author : cos0sin0
# @Email : cos0sin0@qq.com
import cv2
import numpy as np

def show_boxes(image,boxes,scores=None):
    for pts in boxes:
        pts = np.array(pts,dtype=np.int)
        cv2.polylines(image, [pts], 1, (255, 0, 255))
    cv2.imwrite('log/vis.jpg',image)


if __name__ == '__main__':
    img = np.zeros((800, 1000, 3), np.uint8)
    points = np.array([[910, 650], [206, 650], [458, 500], [696, 500]],dtype=np.int)
    cv2.polylines(img, [points], 1, (255, 0, 255))

    cv2.imshow('test',img)
    cv2.waitKey(0)