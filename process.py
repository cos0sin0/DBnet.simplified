# -*- coding: utf-8 -*-
# @Time : 2020/7/22 14:16
# @Author : cos0sin0
# @Email : cos0sin0@qq.com
from collections import OrderedDict

import numpy as np
import cv2
import torch
from shapely.geometry import Polygon
import pyclipper


def _draw_polygons(image, polygons):
    for i in range(len(polygons)):
        polygon = polygons[i].reshape(-1, 2).astype(np.int32)
        color = (0, 0, 255)  # depict polygons in red
        cv2.polylines(image, [polygon], True, color, 1)
    cv2.imshow('polygon',image)
    cv2.waitKey(0)

def make_polyons(data,debug=False):
    polygons = []
    annotations = data['polys']
    ignore_tags = data['ignore_tags']
    for annotation in annotations:
        polygons.append(np.array(annotation['points']))
    polygons = np.array(polygons)
    ignore_tags = np.array(ignore_tags)
    filename = data.get('filename', data['data_id'])
    if debug:
        _draw_polygons(data['image'], polygons)
    return OrderedDict(image=data['image'],
                       polygons=polygons,
                       ignore_tags = ignore_tags,
                       filename=filename)

def make_seg_map(data, min_text_size=8,shrink_ratio=0.4):
    '''
    data: a dict typically returned from `MakeICDARData`,
        where the following keys are contrains:
            image*, polygons*, ignore_tags*, shape, filename
            * means required.
    '''
    def validate_polygons(polygons,ignore_tags, h, w):
        '''
        polygons (numpy.array, required): of shape (num_instances, num_points, 2)
        '''
        if len(polygons) == 0:
            return polygons

        polygons[:, :, 0] = np.clip(polygons[:, :, 0], 0, w - 1)
        polygons[:, :, 1] = np.clip(polygons[:, :, 1], 0, h - 1)

        for i in range(polygons.shape[0]):
            area = polygon_area(polygons[i])
            if abs(area) < 1:
                ignore_tags[i] = True
            if area > 0:
                polygons[i] = polygons[i][(0, 3, 2, 1), :]
        return polygons

    def polygon_area( polygon):
        edge = [
            (polygon[1][0] - polygon[0][0]) * (polygon[1][1] + polygon[0][1]),
            (polygon[2][0] - polygon[1][0]) * (polygon[2][1] + polygon[1][1]),
            (polygon[3][0] - polygon[2][0]) * (polygon[3][1] + polygon[2][1]),
            (polygon[0][0] - polygon[3][0]) * (polygon[0][1] + polygon[3][1])
        ]
        return np.sum(edge) / 2.

    polygons = data['polygons']
    image = data['image']
    filename = data['filename']
    ignore_tags = data['ignore_tags']
    h, w = image.shape[:2]
    polygons = validate_polygons(polygons,ignore_tags, h, w)
    gt = np.zeros((1, h, w), dtype=np.float32)
    mask = np.ones((h, w), dtype=np.float32)
    for i in range(polygons.shape[0]):
        polygon = polygons[i]
        height = min(np.linalg.norm(polygon[0] - polygon[3]),
                     np.linalg.norm(polygon[1] - polygon[2]))
        width = min(np.linalg.norm(polygon[0] - polygon[1]),
                    np.linalg.norm(polygon[2] - polygon[3]))
        if  min(height, width) < min_text_size:
            cv2.fillPoly(mask, polygon.astype(
                np.int32)[np.newaxis, :, :], 0)
            ignore_tags[i] = True
        else:
            polygon_shape = Polygon(polygon)
            distance = polygon_shape.area * \
                       (1 - np.power(shrink_ratio, 2)) / polygon_shape.length
            subject = [tuple(l) for l in polygons[i]]
            padding = pyclipper.PyclipperOffset()
            padding.AddPath(subject, pyclipper.JT_ROUND,
                            pyclipper.ET_CLOSEDPOLYGON)
            shrinked = padding.Execute(-distance)
            if shrinked == []:
                cv2.fillPoly(mask, polygon.astype(
                    np.int32)[np.newaxis, :, :], 0)
                ignore_tags[i] = True
                continue
            shrinked = np.array(shrinked[0]).reshape(-1, 2)
            cv2.fillPoly(gt[0], [shrinked.astype(np.int32)], 1)

    if filename is None:
        filename = ''
    data.update(image=image,
                polygons=polygons,
                gt=gt, mask=mask,
                filename=filename)
    return data

def make_seg_border(data, shrink_ratio=0.4, thresh_max=0.7,thresh_min=0.3):
    '''
        data: a dict typically returned from `MakeICDARData`,
            where the following keys are contrains:
                image*, polygons*, ignore_tags*, shape, filename
                * means required.
    '''
    def draw_border_map(polygon, canvas, mask):
        polygon = np.array(polygon)
        assert polygon.ndim == 2
        assert polygon.shape[1] == 2

        polygon_shape = Polygon(polygon)
        distance = polygon_shape.area * \
            (1 - np.power(shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(l) for l in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND,
                        pyclipper.ET_CLOSEDPOLYGON)
        padded_polygon = np.array(padding.Execute(distance)[0])
        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1

        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin

        xs = np.broadcast_to(
            np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
        ys = np.broadcast_to(
            np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

        distance_map = np.zeros(
            (polygon.shape[0], height, width), dtype=np.float32)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            absolute_distance = point_line_distance(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = distance_map.min(axis=0)

        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
            1 - distance_map[
                ymin_valid-ymin:ymax_valid-ymax+height,
                xmin_valid-xmin:xmax_valid-xmax+width],
            canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])

    def point_line_distance(xs, ys, point_1, point_2):
        '''
        compute the distance from point to a line
        ys: coordinates in the first axis
        xs: coordinates in the second axis
        point_1, point_2: (x, y), the end of the line
        '''
        height, width = xs.shape[:2]
        square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
        square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
        square_distance = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

        cosin = (square_distance - square_distance_1 - square_distance_2) / (2 * np.sqrt(square_distance_1 * square_distance_2))
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_distance_1 * square_distance_2 *square_sin / square_distance)
        # print('distance',square_distance_1,square_distance_2,square_sin,square_distance)

        result[cosin < 0] = np.sqrt(np.fmin(square_distance_1, square_distance_2))[cosin < 0]
        # self.extend_line(point_1, point_2, result)
        return result

    image = data['image']
    polygons = data['polygons']
    ignore_tags = data['ignore_tags']

    canvas = np.zeros(image.shape[:2], dtype=np.float32)
    mask = np.zeros(image.shape[:2], dtype=np.float32)

    for i in range(len(polygons)):
        if ignore_tags[i]:
            continue
        draw_border_map(polygons[i], canvas, mask=mask)
    canvas = canvas * (thresh_max - thresh_min) + thresh_min
    data['thresh_map'] = canvas
    data['thresh_mask'] = mask
    return data

_RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])

def normalize_image( data):
    def add_pos_channel(image):
        image_shape = image.shape[-2:]
        height, width = image_shape
        w = torch.arange(1, width + 1, dtype=torch.float32)
        h = torch.arange(1, height + 1, dtype=torch.float32)
        y, x = torch.meshgrid(h, w)
        x = x / width
        y = y / height
        z = torch.cat((x.unsqueeze(0), y.unsqueeze(0)), 0)

        z = z.to(torch.device('cpu'))
        image = torch.cat((image, z), 0)
        return image
    assert 'image' in data, '`image` in data is required by this process'
    image = data['image']
    image = image.astype("float32")
    image -= _RGB_MEAN
    image /= 255.
    # image = torch.from_numpy(image).permute(2, 0, 1).float()
    image = np.transpose(image,(2,0,1))
    # image = add_pos_channel(image)
    data['image'] = image
    return data