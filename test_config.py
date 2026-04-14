import torch
import csv
import os
import numpy as np

from torch.utils.data import DataLoader, Subset
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models import ResNet50_Weights
from sklearn.model_selection import KFold

from dataset import train_dataset, test_dataset
from utils import set_seed
from evaluate import evaluate
from train import train

set_seed()
epochs = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

fold_dice = []
fold_iou = []

os.makedirs("results", exist_ok=True)

for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):

    print(f"Fold {fold+1}/{5}")

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)
    
    train_loader = DataLoader(
        train_subset,
        batch_size = 4,
        shuffle = True
    )

    val_loader = DataLoader(
        val_subset,
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

    train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        epochs=epochs,
        device=device,
        aux_loss=True
    )

    metrics = evaluate(model, val_loader, device)

    fold_dice.append(metrics["dice"])
    fold_iou.append(metrics["iou"])

    print("Fold Dice:", metrics["dice"])
    print("Fold IoU:", metrics["iou"])

torch.save(model.state_dict(), "deeplabv3_oscc.pth")