import torch
from torchvision.models.detection import (maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_Weights)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import DataLoader
from dataset import (VOCDataset, collate_fn)


print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# Dataset loader
train_dataset = VOCDataset(
    "/mnt/z/DATASETS/pascal-voc-2012-DatasetNinja/train"
)
val_dataset = VOCDataset(
    "/mnt/z/DATASETS/pascal-voc-2012-DatasetNinja/val"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=8,
    collate_fn=collate_fn
)
val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=8,
    collate_fn=collate_fn
)

# Model initialization
model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_Weights)
num_classes = 21

in_features = (model.roi_heads.box_predictor.cls_score.in_features)
model.roi_heads.box_predictor = (FastRCNNPredictor(in_features, num_classes))

in_features_mask = (model.roi_heads.mask_predictor.conv5_mask.in_channels)
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

device = torch.device("cuda")
model.to(device)

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    running_loss = 0

    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device)
                    for k, v in t.items()
                    }
                    for t in targets
                ]
        
        loss_dict = model(images, targets)

        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}: "f"{running_loss/len(train_loader):.4f}")