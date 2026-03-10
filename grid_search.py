import torch
import itertools
import csv
import os
import numpy as np

from torch.utils.data import DataLoader, Subset
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models import ResNet50_Weights
from sklearn.model_selection import KFold

from train import train
from evaluate import evaluate


def run_grid_search(dataset, device):

    param_grid = {
        "weights_backbone": [None, "imagenet"],
        "aux_loss": [False, True],
        "lr": [1e-4, 3e-4, 1e-5],
        "batch_size": [2, 4],
        "weight_decay": [0, 1e-5, 1e-4]
    }

    keys = param_grid.keys()
    values = param_grid.values()

    grid = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results = []

    os.makedirs("results", exist_ok=True)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    for config in grid:

        print("\n====================================")
        print("Running config:", config)
        print("====================================")

        fold_dice = []
        fold_iou = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):

            print(f"\n--- Fold {fold+1}/5 ---")

            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            train_loader = DataLoader(
                train_subset,
                batch_size=config["batch_size"],
                shuffle=True
            )

            val_loader = DataLoader(
                val_subset,
                batch_size=config["batch_size"],
                shuffle=False
            )

            # Backbone weights
            if config["weights_backbone"] == "imagenet":
                backbone_weights = ResNet50_Weights.IMAGENET1K_V1
            else:
                backbone_weights = None

            # Novo modelo para cada fold
            model = deeplabv3_resnet50(
                weights=None,
                weights_backbone=backbone_weights,
                num_classes=1,
                aux_loss=config["aux_loss"]
            ).to(device)

            criterion = torch.nn.BCEWithLogitsLoss()

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config["lr"],
                weight_decay=config["weight_decay"]
            )

            # Treino
            train(
                model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                epochs=30,
                device=device,
                aux_loss=config["aux_loss"]
            )

            # Avaliação
            metrics = evaluate(model, val_loader, device)

            fold_dice.append(metrics["dice"])
            fold_iou.append(metrics["iou"])

            print("Fold Dice:", metrics["dice"])
            print("Fold IoU:", metrics["iou"])

        mean_dice = np.mean(fold_dice)
        mean_iou = np.mean(fold_iou)

        result_row = {
            **config,
            "mean_dice": mean_dice,
            "mean_iou": mean_iou
        }

        results.append(result_row)

        print("\nMean Dice:", mean_dice)
        print("Mean IoU:", mean_iou)

    csv_path = "results/grid_results.csv"

    with open(csv_path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nGrid search finalizado!")
    print("Resultados salvos em:", csv_path)

    return results