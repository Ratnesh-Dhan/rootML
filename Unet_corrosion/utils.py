import torch


def compute_iou(pred, mask, num_classes):

    pred = torch.argmax(pred, dim=1)

    ious = []

    for cls in range(num_classes):

        pred_inds = pred == cls
        target_inds = mask == cls

        intersection = (
            pred_inds & target_inds
        ).float().sum()

        union = (
            pred_inds | target_inds
        ).float().sum()

        if union == 0:
            continue

        ious.append(
            (intersection / union).item()
        )

    if len(ious) == 0:
        return 0

    return sum(ious) / len(ious)

def compute_class_iou(preds, targets, num_classes):

    preds = torch.argmax(preds, dim=1)

    class_ious = []

    for cls in range(num_classes):

        pred_cls = preds == cls
        target_cls = targets == cls

        intersection = (
            pred_cls & target_cls
        ).sum().item()

        union = (
            pred_cls | target_cls
        ).sum().item()

        if union == 0:
            iou = float("nan")
        else:
            iou = intersection / union

        class_ious.append(iou)

    return class_ious