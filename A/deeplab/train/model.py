import segmentation_models_pytorch as smp

model = smp.DeepLabV3Plus(
    encoder_name="efficientnet-b0",
    encoder_weights="imagenet",
    in_channels=3,
    classes=4
)