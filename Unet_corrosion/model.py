import segmentation_models_pytorch as smp


def build_model(num_classes):

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    )

    return model