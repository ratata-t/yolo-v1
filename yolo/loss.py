import torch
import torch.nn as nn

import numpy as np

import config
from utils.data import Bbox


OBJ_NAMES = [
     'aeroplane',
     'bicycle',
     'bird',
     'boat',
     'bottle',
     'bus',
     'car',
     'cat',
     'chair',
     'cow',
     'diningtable',
     'dog',
     'horse',
     'motorbike',
     'person',
     'pottedplant',
     'sheep',
     'sofa',
     'train',
     'tvmonitor'
]


def IoU(bbox1: Bbox, bbox2: Bbox):
    xmin = max(bbox1.xmin, bbox2.xmin)
    ymin = max(bbox1.ymin, bbox2.ymin)
    xmax = min(bbox1.xmax, bbox2.xmax)
    ymax = min(bbox1.ymax, bbox2.ymax)

    x = max(0, xmax - xmin)
    y = max(0, ymax - ymin)
    intersection = x * y

    union = bbox1.area + bbox2.area - intersection
    return intersection / union


def coord_loss(x, y, bbox: Bbox):
    # Make ground truth [-0.5, 0.5] (relative to cell center)
    return (x - bbox.x_gt) ** 2 + (y - bbox.y_gt) ** 2


def perimeter_loss(w, h, bbox:Bbox):
    return (w ** 0.5 - bbox.w_gt ** 0.5) ** 2 + (h ** 0.5 - bbox.h_gt ** 0.5) ** 2


def confidence_loss(c, ious, responsible_bbox):
    return (c - ious[responsible_bbox]) ** 2


def classification_loss(c, pred_vector, name):
    class_to_index = {el:i for i, el in enumerate(OBJ_NAMES)}
    pred_distribution = pred_vector[-config.C:]

    distribution_gt = torch.zeros(config.C)
    distribution_gt[class_to_index[name[0]]] = 1
    return nn.MSELoss()(distribution_gt, pred_distribution)


def noobj_loss(ious, responsible_bbox, pred_vector):
    noobj_loss_val = 0

    for b, iou in enumerate(ious):
        if b != responsible_bbox:
            c = pred_vector[b * 5 + 4]
            noobj_loss_val += (c - iou) ** 2
    return noobj_loss_val


def yolo_loss(pred, names, bboxes):
    coord_loss_val = perimeter_loss_val = 0
    confidence_loss_val = classification_loss_val = 0
    noobj_loss_val = 0
    lmb_ccord = 5
    lmb_noobj = 0.5

    for name, bbox in zip(names, bboxes):
        bbox_gt = Bbox(*bbox)

        # Cut out vector corresponding to the current cell
        pred_vector = pred[bbox_gt.section_x, bbox_gt.section_y]

        ious = []
        for b in range(config.B):
            pred_bbox = pred_vector[b * 5 : (b + 1) * 5]
            ious.append(IoU(Bbox(*pred_bbox[:4]), bbox_gt))

        responsible_bbox = np.argmax(ious)
        x, y, w, h, c = pred_vector[responsible_bbox * 5 : (responsible_bbox + 1) * 5]
        w = torch.nn.functional.relu(w)
        h = torch.nn.functional.relu(h)

        # Calculate loss for the responsible bbox
        coord_loss_val += coord_loss(x, y, bbox_gt)
        perimeter_loss_val += perimeter_loss(w, h, bbox_gt)
        confidence_loss_val += confidence_loss(c, ious, responsible_bbox)
        classification_loss_val += classification_loss(c, pred_vector, name)
        noobj_loss_val += noobj_loss(ious, responsible_bbox, pred_vector)

    #  for empty boxes:
    loss = (
        lmb_ccord * (coord_loss_val + perimeter_loss_val)
        + confidence_loss_val
        + classification_loss_val
        + lmb_noobj * noobj_loss_val
    )
    return loss
