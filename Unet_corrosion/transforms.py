import albumentations as A


def get_train_transform():

    return A.Compose([
        A.Resize(512, 512),

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        A.RandomRotate90(p=0.5),

        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=20,
            p=0.5,
        ),

        A.RandomBrightnessContrast(p=0.5),

        A.GaussNoise(p=0.3),
    ])


def get_val_transform():

    return A.Compose([
        A.Resize(512, 512)
    ])