from tqdm import tqdm
import torch
from torch.utils.data import SubsetRandomSampler
import torchvision
import matplotlib.pyplot as plt

from yolo import loss, model
from utils.preprocessing import prepare_data


def train_model(model, train_loader, loss, optimizer, num_epochs):
    loss_history = []
    train_history = []
    
    for epoch in tqdm(range(num_epochs)):
        loss_accum = 0
        correct_samples = 0
        total_samples = 0
        
        for i, (image, (name, bbox)) in enumerate(train_loader):
            model.train()
            image_gpu = image.to(device)

            prediction = model(image_gpu)
            loss_value = loss(prediction, name, bbox)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            loss_accum += loss_value.item()
            # print(loss_value)
        
        ave_loss = loss_accum / (i+1)
        loss_history.append(float(ave_loss))


if __name__ == "__main__":
    model = model.Yolo()
    voc_train_tensor = torchvision.datasets.VOCDetection(
        "data/voc2012", image_set="train", download=False, transforms=prepare_data)
    train_sampler = SubsetRandomSampler([0])
    dataloader = torch.utils.data.DataLoader(voc_train_tensor, sampler=train_sampler)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_model(model, dataloader, loss.yolo_loss, optimizer, 50)
    