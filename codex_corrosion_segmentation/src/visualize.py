import cv2
import numpy as np
import matplotlib.pyplot as plt


def class_mask_to_bgr(mask, class_to_color):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, bgr_color in class_to_color.items():
        color_mask[mask == class_id] = bgr_color

    return color_mask


def save_sanity_check(image_tensor, mask_tensor, save_path, class_to_color):
    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    image = np.clip(image, 0, 1)

    mask = mask_tensor.cpu().numpy().astype(np.uint8)
    mask_bgr = class_mask_to_bgr(mask, class_to_color)
    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)

    overlay = (0.65 * image * 255 + 0.35 * mask_rgb).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(mask_rgb)
    axes[1].set_title("Mask")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_prediction_visualization(image_tensor, target_mask, pred_mask, save_path, class_to_color):
    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    image = np.clip(image, 0, 1)

    target_bgr = class_mask_to_bgr(target_mask.cpu().numpy().astype(np.uint8), class_to_color)
    pred_bgr = class_mask_to_bgr(pred_mask.cpu().numpy().astype(np.uint8), class_to_color)

    target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
    pred_rgb = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB)

    overlay = (0.65 * image * 255 + 0.35 * pred_rgb).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(target_rgb)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred_rgb)
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    axes[3].imshow(overlay)
    axes[3].set_title("Prediction Overlay")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()