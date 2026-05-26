import torch


@torch.no_grad()
def multiclass_iou(logits, targets, num_classes, ignore_index=None, eps=1e-7):
    """
    logits: [B,C,H,W]
    targets: [B,H,W]
    """
    preds = torch.argmax(logits, dim=1)

    ious = []

    for cls in range(num_classes):
        if cls == ignore_index:
            continue

        pred_cls = preds == cls
        target_cls = targets == cls

        intersection = torch.logical_and(pred_cls, target_cls).sum().float()
        union = torch.logical_or(pred_cls, target_cls).sum().float()

        if union == 0:
            continue

        iou = (intersection + eps) / (union + eps)
        ious.append(iou)

    if len(ious) == 0:
        return torch.tensor(0.0, device=logits.device)

    return torch.stack(ious).mean()


@torch.no_grad()
def per_class_iou(logits, targets, num_classes, eps=1e-7):
    preds = torch.argmax(logits, dim=1)

    results = {}

    for cls in range(num_classes):
        pred_cls = preds == cls
        target_cls = targets == cls

        intersection = torch.logical_and(pred_cls, target_cls).sum().float()
        union = torch.logical_or(pred_cls, target_cls).sum().float()

        if union == 0:
            results[cls] = None
        else:
            results[cls] = ((intersection + eps) / (union + eps)).item()

    return results