import torchvision.transforms as transforms
from PIL import Image
import skimage.draw
import numpy as np
import torch 

import config


def voc_to_yolo(image: Image.Image, annotation):
    x_scale = config.WIDTH / image.width
    y_scale = config.HEIGHT / image.height
    image = image.resize((config.WIDTH, config.HEIGHT))
    bboxes = []
    names = []
    for obj in annotation["annotation"]["object"]:
        xmin = int(int(obj["bndbox"]["xmin"]) * x_scale)
        xmax = int(int(obj["bndbox"]["xmax"]) * x_scale)
        ymin = int(int(obj["bndbox"]["ymin"]) * y_scale)
        ymax = int(int(obj["bndbox"]["ymax"]) * y_scale)
        names.append(obj["name"])
        bboxes.append((xmin, xmax, ymin, ymax))
    return image, names, bboxes


def draw_image_with_bboxes(image, names, bboxes):
    for bbox in bboxes:
        xmin, xmax, ymin, ymax = bbox
        rr, cc = skimage.draw.rectangle_perimeter((ymin, xmin), (ymax, xmax))
        image = np.array(image)
        image[rr, cc, :] = (0, 255, 0)
        image = Image.fromarray(image)
    return image



def prepare_data(image, annotation):
    image, names, bboxes = voc_to_yolo(image, annotation)
    return torch.as_tensor(np.array(image)).permute(2, 0, 1)/255., (names, bboxes)