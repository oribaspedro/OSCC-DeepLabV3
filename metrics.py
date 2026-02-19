def compute_confusion(pred, gt):
    TP = ((pred == 1) & (gt == 1)).sum()
    FP = ((pred == 1) & (gt == 0)).sum()
    TN = ((pred == 0) & (gt == 0)).sum()
    FN = ((pred == 0) & (gt == 1)).sum()
    return TP, FP, TN, FN


def compute_metrics(TP, FP, TN, FN):
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    iou = TP / (TP + FP + FN + 1e-8)
    dice = 2 * TP / (2 * TP + FP + FN + 1e-8)
    sensitivity = TP / (TP + FN + 1e-8)
    specificity = TN / (TN + FP + 1e-8)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "iou": iou,
        "dice": dice,
        "sensitivity": sensitivity,
        "specificity": specificity
    }