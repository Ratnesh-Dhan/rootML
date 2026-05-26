import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class DiceCELoss(nn.Module):
    def __init__(self, dice_weight=1.0, ce_weight=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

        self.dice = smp.losses.DiceLoss(
            mode="multiclass",
            from_logits=True,
        )

        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        # logits: [B,4,H,W]
        # targets: [B,H,W]
        dice_loss = self.dice(logits, targets)
        ce_loss = self.ce(logits, targets)

        return self.dice_weight * dice_loss + self.ce_weight * ce_loss