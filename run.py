import torchvision

from yolo.model import Yolo
from utils.preprocessing import prepare_data


if __name__ == "__main__":
    model = Yolo()
    voc_train = torchvision.datasets.VOCDetection(
        "data/voc2012", image_set="train", download=False, transforms=prepare_data
    )
    image, (name, bbox) = voc_train[0]
    image = image / 255.0
    pred = model(image.unsqueeze(0))
    print(pred.shape)
