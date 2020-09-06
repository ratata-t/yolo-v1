import torchvision.models as models
import torch.nn as nn
import config


class Yolo(nn.Module):
    def __init__(self, backbone=None):
        super().__init__()
        if backbone is None:
            self.backbone = models.resnet18(pretrained=True)
            self.backbone.fc = nn.Sequential()
        else:
            self.backbone = backbone
        self.fc1 = nn.Linear(512, 4096)
        self.fc2 = nn.Linear(4096, config.S * config.S * (config.B * 5 + config.C))

    def forward(self, inputs):
        backbone = self.backbone(inputs)
        fc1 = self.fc1(backbone)
        fc2 = self.fc2(fc1).view((config.S, config.S, config.B * 5 + config.C))
        return fc2
