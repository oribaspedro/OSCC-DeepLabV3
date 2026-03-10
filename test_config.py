import torch
import itertools
import csv
import os
import numpy as np

from torch.utils.data import DataLoader, Subset
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models import ResNet50_Weights
from dataset import train_dataset, test_dataset
from utils import set_seed
from evaluate import evaluate

set_seed()
epochs = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

train_loader = DataLoader(
    train_dataset,
    batch_size = 4,
    shuffle = True
)

test_loader = DataLoader(
    test_dataset,
    batch_size = 4,
    shuffle = True
)

model = deeplabv3_resnet50(
    weights = None,
    weights_backbone = ResNet50_Weights.IMAGENET1K_V1,
    num_classes = 1,
    aux_loss = True
).to(device)

criterion = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 3e-4,
    weight_decay = 0.0
)

for epoch in range(epochs):
    model.train()

    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)

        outputs = model(imgs)

        loss = criterion(outputs['out'], masks)

        loss += 0.4 * criterion(outputs['aux'], masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "deeplabv3_oscc.pth")

results = evaluate(model, test_loader, device)
print(results)