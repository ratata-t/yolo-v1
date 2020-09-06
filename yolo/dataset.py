import torchvision.transforms as transforms
import torchvision

from utils.preprocessing import voc_to_yolo


def load_dataset():
    voc_train = torchvision.datasets.VOCDetection(
        "voc2012", image_set="train", download=False, transforms=prepare_data
    )
    return voc_train


def prepare_data(image, annotation):
    image, names, bboxes = voc_to_yolo(image, annotations)
    return transforms.PILToTensor()(image) / 255.0, (names, bboxes)
