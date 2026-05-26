from utils.augment import train_transform, val_transform
from utils.datasetloader import CorrosionDataset
import os

if __name__=="__main__":
    
    # Base Path
    base_path = "/mnt/d/DATASETS/CORROSION/Corrosion_Condition_State_Classification/Corrosion_Condition_State_Classification/512x512"

    train_images = os.path.join(base_path, "Train", "images_512")
    train_masks = os.path.join(base_path, "Train", "mask_512")
    val_images = os.path.join(base_path, "Test", "images_512")
    val_masks = os.path.join(base_path, "Test", "mask_512")

    train_dataset = CorrosionDataset(
        train_images,
        train_masks,
        transforms=train_transform
    )

    val_dataset = CorrosionDataset(
        val_images,
        val_masks,
        transforms=val_transform
    )
    