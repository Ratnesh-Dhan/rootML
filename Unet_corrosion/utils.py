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