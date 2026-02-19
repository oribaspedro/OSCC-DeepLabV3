import torch
import itertools
import csv
import os

from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models import ResNet50_Weights

from train import train
from evaluate import evaluate


def run_grid_search(train_dataset, val_dataset, device):

    param_grid = {
        "weights_backbone": [None, "imagenet"],
        "aux_loss": [False, True],
        "lr": [3e-4, 1e-4, 1e-5],
        "batch_size": [2, 4, 8],
        "weight_decay": [0, 1e-5]
    }

    keys = param_grid.keys()
    values = param_grid.values()

    grid = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results = []

    os.makedirs("results", exist_ok=True)

    for config in grid:

        print("\n====================================")
        print("Running config:", config)
        print("====================================")
        
        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False
        )

        # Backbone weights
        if config["weights_backbone"] == "imagenet":
            backbone_weights = ResNet50_Weights.IMAGENET1K_V1
        else:
            backbone_weights = None

        # Model
        model = deeplabv3_resnet50(
            weights=None,
            weights_backbone=backbone_weights,
            num_classes=1,
            aux_loss=config["aux_loss"]
        ).to(device)

        # Loss e Optimizer
        criterion = torch.nn.BCEWithLogitsLoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"]
        )

        # Training
        train(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            epochs=5,   # use pequeno no grid
            device=device,
            aux_loss=config["aux_loss"]
        )

        # Evaluation
        metrics = evaluate(model, val_loader, device)

        result_row = {
            **config,
            "mean_dice": metrics["dice"],
            "mean_iou": metrics["iou"]
        }

        results.append(result_row)

        print("Dice:", metrics["dice"])
        print("IoU:", metrics["iou"])

    # Save CSV
    csv_path = "results/grid_results.csv"

    with open(csv_path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nGrid search finalizado!")
    print("Resultados salvos em:", csv_path)

    return results