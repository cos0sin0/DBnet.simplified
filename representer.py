# -*- coding: utf-8 -*-
# @Time : 2020/7/23 15:13
# @Author : cos0sin0
# @Email : cos0sin0@qq.com
import cv2
import numpy as np
from shapely.geometry import Polygon
import pyclipper

_max_candidates=2000
_box_thresh=0.5
_min_size=3
_scale_ratio=0.4

def unclip(box, unclip_ratio=1.5):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded


def get_mini_boxes(contour):
    if len(contour) == 0:
        return [],0
    try:
        bounding_box = cv2.minAreaRect(contour)
    except Exception:
        print('[error] get_mini_boxes')
        print('contour',contour)
        print('error: (-215:Assertion failed) total >= 0 && (depth == CV_32F || depth == CV_32S) in function "cv::convexHull"')
        return [],0,0
    angle = bounding_box[2]
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2],
           points[index_3], points[index_4]]
    return box, min(bounding_box[1]),min(abs(angle),90-abs(angle))


def box_score_fast(heatmap, _box):
    h, w = heatmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(heatmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def polygons_from_bitmap(heatmap,bitmap,dest_width,dest_height):
    if type(heatmap) == list or type(heatmap) == np.ndarray:
        bitmap = bitmap[0]
        heatmap = heatmap[0]
    else:
        assert bitmap.size(0) == 1
        bitmap = bitmap.cpu().numpy()[0][0]  # The first channel
        heatmap = heatmap.cpu().detach().numpy()[0][0]
    height, width = bitmap.shape
    boxes = []
    scores = []

    contours, _ = cv2.findContours(
        (bitmap * 255).astype(np.uint8),
        cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape((-1, 2))
        if points.shape[0] < 4:
            continue
        # _, sside = self.get_mini_boxes(contour)
        # if sside < self.min_size:
        #     continue
        score = box_score_fast(heatmap, points.reshape(-1, 2))
        if _box_thresh > score:
            continue

        if points.shape[0] > 2:
            box = unclip(points, unclip_ratio=2.0)
            if len(box) > 1:
                continue
        else:
            continue
        box = box.reshape(-1, 2)
        _, sside,angle = get_mini_boxes(box.reshape((-1, 1, 2)))
        if sside < _min_size + 2:
            continue
        if angle > 10:
            continue
        if not isinstance(dest_width, int):
            dest_width = dest_width.item()
            dest_height = dest_height.item()

        box[:, 0] = np.clip(
            np.round(box[:, 0] / width * dest_width), 0, dest_width)
        box[:, 1] = np.clip(
            np.round(box[:, 1] / height * dest_height), 0, dest_height)
        boxes.append(box.tolist())
        scores.append(score)
    return boxes, scores


def boxes_from_bitmap(heatmap,bitmap,dest_width,dest_height):
    '''
    _bitmap: single map with shape (1, H, W),
        whose values are binarized as {0, 1}
    '''

    assert bitmap.size(0) == 1
    bitmap = bitmap.cpu().numpy()[0][0]  # The first channel
    heatmap = heatmap.cpu().detach().numpy()[0][0]
    height, width = bitmap.shape
    contours, _ = cv2.findContours(
        (bitmap * 255).astype(np.uint8),
        cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)
    boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
    scores = np.zeros((num_contours,), dtype=np.float32)

    for index in range(num_contours):
        contour = contours[index]
        points, sside,angle = get_mini_boxes(contour)
        if sside < _min_size:
            continue
        if angle > 10:
            continue
        points = np.array(points)
        score = box_score_fast(heatmap, points.reshape(-1, 2))
        if _box_thresh > score:
            continue
        box = unclip(points).reshape(-1, 1, 2)
        box, sside,angle = get_mini_boxes(box)
        if sside < _min_size + 2:
            continue
        if angle > 10:
            continue
        box = np.array(box)
        if not isinstance(dest_width, int):
            dest_width = dest_width.item()
            dest_height = dest_height.item()

        box[:, 0] = np.clip(
            np.round(box[:, 0] / width * dest_width), 0, dest_width)
        box[:, 1] = np.clip(
            np.round(box[:, 1] / height * dest_height), 0, dest_height)
        boxes[index, :, :] = box.astype(np.int16)
        scores[index] = score
    return boxes, scores


def map2box(pred,dest_width,dest_height,thresh=0.3,is_output_polygon=False):
    def binarize(pred,):
        return pred > thresh
    if isinstance(pred, dict):
        pred = pred['binary']
    segmentation = binarize(pred)
    if is_output_polygon:
        boxes, scores = polygons_from_bitmap(pred,segmentation, dest_width, dest_height)
    else:
        boxes, scores = boxes_from_bitmap(pred,segmentation, dest_width, dest_height)
    return boxes, scores