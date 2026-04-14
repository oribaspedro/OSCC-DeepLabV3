import torch
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models import ResNet50_Weights

from dataset import test_dataset
from evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_loader = DataLoader(
    test_dataset,
    batch_size = 4,
    shuffle = False
)

model = deeplabv3_resnet50(
    weights=None,
    weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
    num_classes=1,
    aux_loss=True
)

model.load_state_dict(torch.load("deeplabv3_oscc.pth", map_location=device))

model = model.to(device)

model.eval()


results = evaluate(model, test_loader, device)
print(results)