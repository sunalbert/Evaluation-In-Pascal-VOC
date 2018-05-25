# -*- coding: utf-8 -*-
# @Time    : 5/24/18 10:54 AM
# @Author  : yunfan
# @File    : voc_eval.py

import numpy as np
import json
from DetmAPinVOC import DetmAPinVOC
from gluoncv.data.pascal_voc.detection import VOCDetection

DEBUG = True

VOC_2007_JSON_PATH = './VOC2007-SSD-512.json'


def res_to_allbbox(voc_classes, path=VOC_2007_JSON_PATH):
    with open(path, 'r') as f:
        res = json.load(f)
    detections = {}
    for res_id in res.keys():
        val = res[res_id]  # [{"loc":[xmin, ymin, xmax, ymax], "soc":0.8, "clsna":"car", "clsid":6},{}...]
        im_ind = res_id[-10:-4]  # 006907
        if im_ind == '000128':
            print("000128")
        for bbox in val:
            soc = bbox['soc']
            clsna = bbox['clsna']
            clsid = bbox['clsid']
            loc = bbox['loc']
            loc.append(float(soc))
            cls_ind = voc_classes.index(clsna)
            if cls_ind not in detections.keys():
                detections[cls_ind] = {}
            if im_ind not in detections[cls_ind].keys():
                detections[cls_ind][im_ind] = []
            detections[cls_ind][im_ind].append(loc)

    for cls_ind in detections.keys():
        rr = detections[cls_ind]
        for im_ind in rr:
            detections[cls_ind][im_ind] = np.array(detections[cls_ind][im_ind])
    return detections


def from_VOC_label():
    voc_2007_test_set = VOCDetection(
        root='/Users/yunfanlu/WorkPlace/MyData/VOCDevkit',
        splits=((2007, 'test'),)
    )
    voc_2007_det_label = {}
    for ind in range(len(voc_2007_test_set)):
        image_ind = voc_2007_test_set._items[ind][1]
        bbox_list = voc_2007_test_set[ind][1]
        for bbox in bbox_list:
            bbox_coord = bbox[:4].tolist()
            class_id = int(bbox[4])
            class_name = voc_2007_test_set.CLASSES[class_id]
            class_new_id = pascalVOC.classes.index(class_name)
            if class_new_id not in voc_2007_det_label.keys():
                voc_2007_det_label[class_new_id] = {}
            if image_ind not in voc_2007_det_label[class_new_id].keys():
                voc_2007_det_label[class_new_id][image_ind] = []
            bbox_coord.append(1.0)
            voc_2007_det_label[class_new_id][image_ind].append(bbox_coord)

    for i in voc_2007_det_label.keys():
        for j in voc_2007_det_label[i].keys():
            voc_2007_det_label[i][j] = np.array(voc_2007_det_label[i][j])
    return voc_2007_det_label

if __name__ == '__main__':
    pascalVOC = DetmAPinVOC(
        image_set='2007_test',
        devkit_path='/Users/yunfanlu/WorkPlace/MyData/VOCDevkit')

    if DEBUG:
        detections = from_VOC_label()
    else:
        detections = res_to_allbbox(voc_classes=pascalVOC.classes)

    pascalVOC.evaluate_detections(detections=detections)
