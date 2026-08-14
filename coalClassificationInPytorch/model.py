import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.features = nn.Sequential(
            # First block
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, eps=1e-3, momentum=0.01),
            nn.ReLU(),

            # Second block
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=3),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        # Equivalent to GlobalAveragePooling2d
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)

        return x