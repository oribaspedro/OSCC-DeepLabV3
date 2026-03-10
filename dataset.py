import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

BASE_DIR = "data"

TRAIN_IMG_DIR = os.path.join(BASE_DIR, "training/tumor/tma")
TRAIN_MASK_DIR = os.path.join(BASE_DIR, "training/lesion_annotations")

TEST_IMG_DIR = os.path.join(BASE_DIR, "testing/tumor/tma")
TEST_MASK_DIR = os.path.join(BASE_DIR, "testing/lesion_annotations")

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

train_dataset = OSCCDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
test_dataset = OSCCDataset(TEST_IMG_DIR, TEST_MASK_DIR)