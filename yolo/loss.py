import torch
import torch.nn as nn

import numpy as np

import config
from utils.data import get_obj_names


OBJ_NAMS = [
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


def IoU(bbox1, bbox2):
    x_min1, x_max1, y_min1, y_max1 = bbox1
    x_min2, x_max2, y_min2, y_max2 = bbox2
    x_min = max(x_min1, x_min2)
    y_min = max(y_min1, y_min2)
    x_max = min(x_max1, x_max2)
    y_max = min(y_max1, y_max2)

    x = max(0, x_max - x_min)
    y = max(0, y_max - y_min)
    intersection = x * y

    union = (
        (x_max1 - x_min1) * (y_max1 - y_min1)
        + (x_max2 - x_min2) * (y_max2 - y_min2)
        - intersection
    )
    return intersection / union


class Bbox:
    s_width = config.WIDTH / config.S
    s_height = config.HEIGHT / config.S

    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax        

    @property
    def x_c(self):
        return (self.xmin + self.xmax) / 2.0
    
    @property
    def y_c(self):
        return (self.ymin + self.ymax) / 2.0

    @property
    def section_x(self):
        # move in init?
        return int(self.x_c / self.s_width)

    @property
    def section_y(self):
        # move in init?
        return int(self.y_c / self.s_height)

    @property
    def x_gt(self):
        return (self.x_c - self.s_width * (self.section_x + 0.5)) / self.s_width

    @property
    def y_gt(self):
        return (self.y_c - self.s_height * (self.section_y + 0.5)) / self.s_height

    @property
    def w_gt(self):
        return (self.xmax - self.xmin)/self.s_width # gt - groung truth

    @property
    def h_gt(self):
        return (self.ymax - self.ymin)/self.s_height
    
    


def coord_loss(x, y, bbox: Bbox):
    # Make ground truth [-0.5, 0.5] (relative to cell center)
    return (x - bbox.x_gt) ** 2 + (y - bbox.y_gt) ** 2


def perimeter_loss(w, h, bbox:Bbox):
    return (w ** 0.5 - bbox.w_gt ** 0.5) ** 2 + (h ** 0.5 - bbox.h_gt ** 0.5) ** 2


def confidence_loss(c, ious, responsible_bbox):
    return (c - ious[responsible_bbox]) ** 2


def classification_loss(c, pred_vector):
    class_to_index = {el:i for i, el in enumerate(OBJ_NAMS)}
    pred_distribution = pred_vector[-config.C:]

    distribution_gt = torch.zeros(config.C)
    distribution_gt[class_to_index[OBJ_NAMS[0]]] = 1
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
            ious.append(IoU(pred_bbox[:4], bbox))

        responsible_bbox = np.argmax(ious)
        x, y, w, h, c = pred_vector[responsible_bbox * 5 : (responsible_bbox + 1) * 5]
        w = torch.nn.functional.relu(w)
        h = torch.nn.functional.relu(h)

        # Calculate loss for the responsible bbox
        coord_loss_val += coord_loss(x, y, bbox_gt)
        perimeter_loss_val += perimeter_loss(w, h, bbox_gt)
        confidence_loss_val += confidence_loss(c, ious, responsible_bbox)
        classification_loss_val += classification_loss(c, pred_vector)
        noobj_loss_val += noobj_loss(ious, responsible_bbox, pred_vector)

    #  for empty boxes:
    loss = (
        lmb_ccord * (coord_loss_val + perimeter_loss_val)
        + confidence_loss_val
        + classification_loss_val
        + lmb_noobj * noobj_loss_val
    )
    return loss
