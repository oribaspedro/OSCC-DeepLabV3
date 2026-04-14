import os
import torch
import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

from grid_search import run_grid_search
from utils import set_seed
from dataset import train_dataset, test_dataset
from grid_search_losses import run_loss_experiment
'''
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
'''

def main():

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_dataset = OSCCDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
    test_dataset = OSCCDataset(TEST_IMG_DIR, TEST_MASK_DIR)

    results = run_grid_search(train_dataset, device)
    print(run_loss_experiment(train_dataset, device))

    save_results_to_csv(results)


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