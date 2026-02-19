import torch
from metrics import compute_confusion, compute_metrics

def evaluate(model, loader, device):

    model.eval()
    TP = FP = TN = FN = 0

    with torch.no_grad():
        for imgs, masks in loader:

            imgs = imgs.to(device)
            masks = masks.to(device)

            outputs = model(imgs)['out']
            preds = torch.sigmoid(outputs) > 0.5

            tp, fp, tn, fn = compute_confusion(
                preds.cpu().numpy(),
                masks.cpu().numpy()
            )

            TP += tp
            FP += fp
            TN += tn
            FN += fn

    return compute_metrics(TP, FP, TN, FN)