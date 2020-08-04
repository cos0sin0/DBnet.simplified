# -*- coding: utf-8 -*-
# @Time : 2020/7/22 14:10
# @Author : cos0sin0
# @Email : cos0sin0@qq.com

import cv2
import numpy as np


def resize_image(image,polys,new_width,new_height):
    height, width = image.shape[:2]
    ori_scale = height/width
    new_scale = new_height/new_width
    canvas = np.zeros((new_height,new_width, 3), np.uint8)
    if ori_scale > new_scale:
        zoom = new_height/height
        re_width = int(zoom*width)
        image = cv2.resize(image, (re_width, new_height))
        canvas[:, :re_width, :] = image
    else:
        zoom = new_width/width
        re_height = int(zoom*height)
        image = cv2.resize(image, (new_width, re_height))
        canvas[:re_height, :, :] = image
    new_polys = polys.copy()
    new_polys[:, :, 0] = new_polys[:, :, 0] * zoom
    new_polys[:, :, 1] = new_polys[:, :, 1] * zoom
    # cv2.imshow('resize',canvas)
    # cv2.waitKey(0)
    return canvas,new_polys

def adjust_size(image,polys,zoom = 32):
    height, width = image.shape[:2]
    hzoom = round(height/zoom)
    new_height = hzoom*zoom
    wzoom = round(width/zoom)
    new_width = wzoom*zoom
    image = cv2.resize(image, (new_width, new_height))
    new_polys = polys.copy()
    new_polys[:, :, 0] = new_polys[:, :, 0] * new_width/width
    new_polys[:, :, 1] = new_polys[:, :, 1] * new_height/height
    return image,new_polys



if __name__ == '__main__':
    a = np.array([[[1,2],[3,4]]])
    print(a[:,:,0])