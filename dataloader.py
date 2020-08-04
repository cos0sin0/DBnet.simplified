# -*- coding: utf-8 -*-
# @Time : 2020/7/22 13:39
# @Author : cos0sin0
# @Email : cos0sin0@qq.com
import numpy as np
import re
import os
import json
from torch.utils import data
import cv2
from process import make_polyons,make_seg_border,make_seg_map,normalize_image
from utils import resize_image,adjust_size

class SroieOcrDataset(data.Dataset):
    r'''Dataset reading from images.
       Args:
           Processes: A series of Callable object, which accept as parameter and return the data dict,
               typically inherrited the `DataProcess`(data/processes/data_process.py) class.
       '''

    def __init__(self, data_dir='data/sroie/ocr/', is_resize=True, width=768, height=1536):
        self.data_dir = data_dir
        self.images = []
        self.gts = []
        self.parser_pattern = re.compile('(\d{1,4},\d{1,4},\d{1,4},\d{1,4},\d{1,4},\d{1,4},\d{1,4},\d{1,4}),(.*)')
        self.init_samples()
        self.num_samples = len(self.images)
        self.is_resize = is_resize
        self.width = width
        self.height = height

    def init_samples(self):

        def get_names():
            fnames = os.listdir(self.data_dir)
            names = []
            # rate = 0

            for name in fnames:
                if name.endswith('.jpg'):
                    image_path = os.path.join(self.data_dir, name)
                    # img = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    # h, w = img.shape[0:2]
                    # rate +=h/w
                    # print("h/w",h/w)
                    names.append(image_path[:-4])
            # print('average:',rate/len(names))
            return names

        def process_txt(tpath):
            with open(tpath, 'r', encoding='utf-8') as fd:
                msg = fd.read()
                lines = msg.strip().split('\n')
                polys = []
                for line in lines:
                    line = line.strip()
                    matched = self.parser_pattern.match(line)
                    if matched:
                        pointstr = matched.group(1)
                        text = matched.group(2)
                        points = pointstr.split(',')
                        item = {}
                        poly = np.array(points,dtype=np.int).reshape((-1, 2)).tolist()
                        item['text'] = text
                        item['points'] = poly
                        polys.append(item)
                    else:
                        print("missing match",line)
                return polys

        data_list = get_names()
        for prefix_name in data_list:
            image_path =  prefix_name + '.jpg'
            json_path =  prefix_name + '.txt'
            polys = process_txt(json_path)
            self.gts.append(polys)
            self.images.append(image_path)

    def __getitem__(self, index, retry=0):
        if index >= self.num_samples:
            index = index % self.num_samples
        data = {}
        image_path = self.images[index]
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        polys = self.gts[index]

        data['filename'] = image_path
        data['data_id'] = image_path
        data['image'] = img
        data['polys'] = polys
        data['ignore_tags'] = [False] * len(polys)

        data = make_polyons(data)  # add polyon into data

        if self.is_resize:
            img, polygons = resize_image(img, data['polygons'], self.width, self.height)
        else:
            img, polygons = adjust_size(img, data['polygons'])
        data['image'] = img
        data['polygons'] = polygons
        # timg = img
        # cv2.polylines(timg, polygons.astype(np.int), 1, (255, 0, 255))
        # cv2.imshow('visual', timg)
        # cv2.waitKey(0)

        data = make_seg_map(data)  # add mask and gt into data
        data = make_seg_border(data)  # add thresh_map and thresh_mask into data
        data = normalize_image(data)  # normalize image
        superfluous = ['polygons', 'filename', 'ignore_tags']
        for key in superfluous:
            del data[key]
        # tt =  data['image']
        # # tt = tt.astype(np.uint8)
        # tt = np.transpose(tt,(1,2,0))
        # print(tt.shape)
        # cv2.imshow('final',tt)
        # cv2.waitKey(0)
        # cv2.imshow('thresh_map',data['thresh_map'])
        # cv2.waitKey(0)
        # if image_path == 'data/4.tif':
        #     print('binggo')
        #     image = data['image']
        #     image = np.transpose(image, (1, 2, 0))
        #     cv2.imshow('gt',image)
        #     cv2.waitKey(0)

        return data['image'], data['gt'], data['mask'], data['thresh_map'], data['thresh_mask']

    def __len__(self):
        return len(self.images)

class ImageDataset(data.Dataset):
    r'''Dataset reading from images.
    Args:
        Processes: A series of Callable object, which accept as parameter and return the data dict,
            typically inherrited the `DataProcess`(data/processes/data_process.py) class.
    '''

    def __init__(self, data_dir='data/',is_resize=False,width=640,height=960):
        self.data_dir = data_dir
        self.images = []
        self.gts = []
        self.init_samples()
        self.num_samples = len(self.images)
        self.is_resize = is_resize
        self.width = width
        self.height= height


    def init_samples(self):

        def get_names():
            fnames = os.listdir(self.data_dir)
            names = []
            for name in fnames:
                if name.endswith('.json'):
                    names.append(name[:-5])
            return names

        def process_json(jpath):
            with open(jpath,'r',encoding='utf-8') as fd:
                json_obj = json.load(fd)
                objs = json_obj['shapes']
                polys = []
                for obj in objs:
                    label = obj['label']
                    points = obj['points']
                    shape_type = obj['shape_type']
                    item = {}
                    if shape_type == 'rectangle':
                        poly = [points[0][0], points[0][1], points[1][0], points[0][1], points[1][0], points[1][1],
                                points[0][0], points[1][1]]
                        poly = np.array(poly).reshape((-1, 2)).tolist()
                    else:
                        poly = points
                    item['label'] = label
                    item['points'] = poly
                    polys.append(item)
                return polys

        data_list = get_names()
        for prefix_name in data_list:
            image_path=os.path.join(self.data_dir,prefix_name + '.tif')
            json_path = os.path.join(self.data_dir,prefix_name + '.json')
            polys = process_json(json_path)
            self.gts.append(polys)
            self.images.append(image_path)


    def __getitem__(self, index, retry=0):
        if index >= self.num_samples:
            index = index % self.num_samples
        data = {}
        image_path = self.images[index]
        polys = self.gts[index]
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        data['filename'] = image_path
        data['data_id'] = image_path
        data['image'] = img
        data['polys'] = polys
        data['ignore_tags'] = [False]*len(polys)

        data = make_polyons(data)#add polyon into data

        if self.is_resize:
            img,polygons = resize_image(img,data['polygons'],self.width,self.height)
        else:
            img,polygons = adjust_size(img,data['polygons'])
        data['image'] = img
        data['polygons'] = polygons
        # timg = img
        # cv2.polylines(timg, polygons.astype(np.int), 1, (255, 0, 255))
        # cv2.imshow('visual', timg)
        # cv2.waitKey(0)

        data = make_seg_map(data)#add mask and gt into data
        data = make_seg_border(data)#add thresh_map and thresh_mask into data
        data = normalize_image(data)# normalize image
        superfluous = ['polygons', 'filename', 'ignore_tags']
        for key in superfluous:
            del data[key]
        # tt =  data['image']
        # # tt = tt.astype(np.uint8)
        # tt = np.transpose(tt,(1,2,0))
        # print(tt.shape)
        # cv2.imshow('final',tt)
        # cv2.waitKey(0)
        # cv2.imshow('thresh_map',data['thresh_map'])
        # cv2.waitKey(0)
        # if image_path == 'data/4.tif':
        #     print('binggo')
        #     image = data['image']
        #     image = np.transpose(image, (1, 2, 0))
        #     cv2.imshow('gt',image)
        #     cv2.waitKey(0)


        return data['image'],data['gt'],data['mask'],data['thresh_map'],data['thresh_mask']

    def __len__(self):
        return len(self.images)

if __name__ == '__main__':
    id = SroieOcrDataset()
    for data in id:
        print(data)