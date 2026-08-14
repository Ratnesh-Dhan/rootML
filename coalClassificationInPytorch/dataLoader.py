import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


TRAIN_DIR = r"/mnt/d/NML ML Works/newCoalByDeepBhaiya/16/TRAINING 16"
VALIDATION_DIR = r"/mnt/d/NML ML Works/newCoalByDeepBhaiya/16/VALIDATION"


def load_dataset(batch_size):

    # Equivalent to:
    # rescale=1./255
    # target_size=(16, 16)
    train_transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
    ])

    validation_transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
    ])

    # Equivalent to Keras flow_from_directory()
    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=train_transform
    )

    validation_dataset = datasets.ImageFolder(
        root=VALIDATION_DIR,
        transform=validation_transform
    )

    # Equivalent to shuffle=True for training
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # Equivalent to shuffle=False for validation
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, validation_loader