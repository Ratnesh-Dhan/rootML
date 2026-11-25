
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# Update these paths to match your actual folders.
DEVELOPMENT_DIR = Path(
    "/mnt/z/DATASETS/Coal_Patch/dataset_split/development"
)

TEST_DIR = Path(
    "/mnt/z/DATASETS/Coal_Patch/dataset_split/test"
)


def get_datasets():

    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
    ])

    development_dataset = datasets.ImageFolder(
        root=DEVELOPMENT_DIR,
        transform=transform
    )

    test_dataset = datasets.ImageFolder(
        root=TEST_DIR,
        transform=transform
    )

    # Ensure class indices are identical.
    if development_dataset.class_to_idx != test_dataset.class_to_idx:
        raise ValueError(
            "Class mappings differ between development and test datasets."
        )

    return development_dataset, test_dataset


def make_loader(
    dataset,
    indices=None,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    seed=42,
):

    if indices is not None:
        dataset = Subset(dataset, indices)

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        generator=generator,
    )

    return loader
