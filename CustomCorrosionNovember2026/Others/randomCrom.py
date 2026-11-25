import cv2
import albumentations as A
from pathlib import Path
from tqdm import tqdm

# img_dir = Path("/mnt/d/DATASETS/CORROSION/Corrosion_Condition_State_Classification/Corrosion_Condition_State_Classification/512x512/Train/images_512")
# mask_dir = Path("/mnt/d/DATASETS/CORROSION/Corrosion_Condition_State_Classification/Corrosion_Condition_State_Classification/512x512/Train/mask_512")
# out_img_dir = Path("/mnt/d/DATASETS/CORROSION/Corrosion_Condition_State_Classification/Corrosion_Condition_State_Classification/512x512/CroppedTrain/images")
# out_mask_dir = Path("/mnt/d/DATASETS/CORROSION/Corrosion_Condition_State_Classification/Corrosion_Condition_State_Classification/512x512/CroppendTrain/masks")
img_dir = Path(r"D:\DATASETS\CORROSION\Corrosion_Condition_State_Classification\Corrosion_Condition_State_Classification\512x512\Train\images_512")
mask_dir = Path(r"D:\DATASETS\CORROSION\Corrosion_Condition_State_Classification\Corrosion_Condition_State_Classification\512x512\Train\mask_512")
out_img_dir = Path(r"D:\DATASETS\CORROSION\Corrosion_Condition_State_Classification\Corrosion_Condition_State_Classification\512x512\CroppedTrain\images")
out_mask_dir = Path(r"D:\DATASETS\CORROSION\Corrosion_Condition_State_Classification\Corrosion_Condition_State_Classification\512x512\CroppedTrain\masks")

out_img_dir.mkdir(parents=True, exist_ok=True)
out_mask_dir.mkdir(parents=True, exist_ok=True)

CROPS_PER_IMAGE = 5 
AUGS_PER_IMAGE = 10 # 1 original + 10 augmented = 11x total
resize_only = A.Resize(height=512, width=512)

# RandomResizedCrop handles variable input sizes and resizes to 512x512 in one step
transform = A.Compose([
    A.RandomResizedCrop(size=(512, 512), scale=(0.2, 1.0), ratio=(0.8, 1.2), p=1.0),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=30, val_shift_limit=0, p=0.5),
])

saved_count = 0

for img_path in tqdm(sorted(img_dir.glob("*.*"))):
    mask_path = mask_dir / f"{img_path.stem}.png"  # Adjust extension if needed
    
    image = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)  # Preserves exact pixel class values

    if image is None or mask is None:
        continue

    # Saving original image & mask
    orig = resize_only(image=image, mask=mask)
    cv2.imwrite(str(out_img_dir/ f"{img_path.stem}_orig.png"), orig["image"])
    cv2.imwrite(str(out_mask_dir/f"{img_path.stem}_orig.png"), orig["mask"])
    saved_count += 1


    # save augmented
    for i in range(AUGS_PER_IMAGE): # CROPS_PER_IMAGE
        augmented = transform(image=image, mask=mask)
        
        cv2.imwrite(str(out_img_dir / f"{img_path.stem}_aug{i}.png"), augmented["image"])
        cv2.imwrite(str(out_mask_dir / f"{img_path.stem}_aug{i}.png"), augmented["mask"])
        saved_count += 1

print(f"Done! Total training image-mask pairs saved: {saved_count}")