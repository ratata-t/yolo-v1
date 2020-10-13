import torchvision
import torch

from yolo.model import Yolo
from yolo.loss import yolo_loss
from utils.preprocessing import prepare_data


if __name__ == "__main__":
    model = Yolo()
    voc_train_tensor = torchvision.datasets.VOCDetection(
        "data/voc2012", image_set="train", download=False, transforms=prepare_data
    )
    dataloader = torch.utils.data.DataLoader(voc_train_tensor)
    elem = next(iter(dataloader))
    image, (name, bbox) = elem
    pred = model(image)
    loss = yolo_loss(pred, name, bbox)
    print(loss)

    # image, (name, bbox) = voc_train[0]
    # image = image / 255.0
    # pred = model(image.unsqueeze(0))
    # print(pred.shape)
    
