import os
import torch
import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

from grid_search import run_grid_search
from utils import set_seed

BASE_DIR = "data"

TRAIN_IMG_DIR = os.path.join(BASE_DIR, "training/tumor/tma")
TRAIN_MASK_DIR = os.path.join(BASE_DIR, "training/lesion_annotations")

VAL_IMG_DIR = os.path.join(BASE_DIR, "testing/tumor/tma")
VAL_MASK_DIR = os.path.join(BASE_DIR, "testing/lesion_annotations")


class OSCCDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.images = sorted(os.listdir(images_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(self.images_dir, img_name)

        name, ext = os.path.splitext(img_name)
        mask_name = name + "_mask" + ext
        mask_path = os.path.join(self.masks_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = transforms.ToTensor()(np.array(image))
        mask = (np.array(mask) > 0).astype(np.float32)
        mask = torch.tensor(mask).unsqueeze(0)

        return image, mask


def main():

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_dataset = OSCCDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
    val_dataset = OSCCDataset(VAL_IMG_DIR, VAL_MASK_DIR)

    results = run_grid_search(train_dataset, val_dataset, device)

    save_results_to_csv(results)

    print("\nGrid search finalizado.")
    print("Resultados salvos em results/grid_results.csv")


def save_results_to_csv(results):

    import os
    import pandas as pd

    os.makedirs("results", exist_ok=True)

    df = pd.DataFrame(results)

    # ordenar pelo melhor Dice
    df = df.sort_values(by="mean_dice", ascending=False)

    df.to_csv("results/grid_results.csv", index=False)

    print("\nMelhor configuração:")
    print(df.iloc[0])


if __name__ == "__main__":
    main()