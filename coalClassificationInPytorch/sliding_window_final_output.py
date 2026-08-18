import os
import gc
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model import CNN


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

WINDOW_SIZE = 16
STRIDE = 16
BATCH_SIZE = 1024
NUM_CLASSES = 5

# Full image directory
IMAGE_ROOT = (
    "/mnt/z/DATASETS/CoalFullImagesC&DBM/final32New"
)

RESULT_ROOT = "./results"


# ============================================================
# Model checkpoints
# ============================================================

MODEL_PATHS = {
    "Adam": (
        "./models_aug17_2026_100_epochs/Adam/"
        "checkpoint_best_weights.pth"
    ),

    "Adadelta": (
        "./models_aug17_2026_100_epochs/Adadelta/"
        "checkpoint_best_weights.pth"
    ),

    "Adagrad": (
        "./models_aug17_2026_100_epochs/Adagrad/"
        "checkpoint_best_weights.pth"
    ),

    "AdamW": (
        "./models_aug17_2026_100_epochs/AdamW/"
        "checkpoint_best_weights.pth"
    ),

    "Nadam": (
        "./models_aug17_2026_100_epochs/Nadam/"
        "checkpoint_best_weights.pth"
    ),

    "RMSprop": (
        "./models_aug17_2026_100_epochs/RMSprop/"
        "checkpoint_best_weights.pth"
    ),
}


# ============================================================
# Load model
# ============================================================

def load_model(model_path):

    model = CNN(
        num_classes=NUM_CLASSES
    )

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE
    )

    # Our training script saved this format
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# Sliding Window Inference
# ============================================================

def sliding_window_inference(
    model,
    image,
    window_size=16,
    stride=16,
    class_num=5
):

    h, w, _ = image.shape

    # --------------------------------------------------------
    # Convert image to tensor
    #
    # OpenCV image is HWC uint8 [0, 255]
    # PyTorch model expects CHW float [0, 1]
    # --------------------------------------------------------

    image_tensor = torch.from_numpy(
        image
    ).float()

    image_tensor = image_tensor / 255.0

    # HWC -> CHW
    image_tensor = image_tensor.permute(
        2, 0, 1
    )

    # Add batch dimension
    # [C, H, W] -> [1, C, H, W]
    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(
        DEVICE,
        non_blocking=True
    )

    # --------------------------------------------------------
    # Extract patches
    #
    # TensorFlow:
    # tf.image.extract_patches(...)
    #
    # PyTorch:
    # F.unfold(...)
    # --------------------------------------------------------

    patches = F.unfold(
        image_tensor,
        kernel_size=window_size,
        stride=stride
    )

    # Shape:
    #
    # [1, C * window_size * window_size, N]
    #
    # Convert to:
    #
    # [N, C, window_size, window_size]

    patches = patches.squeeze(0)

    patches = patches.transpose(
        0, 1
    )

    patches = patches.reshape(
        -1,
        3,
        window_size,
        window_size
    )

    total_patches = patches.shape[0]

    # --------------------------------------------------------
    # Batched inference
    # --------------------------------------------------------

    predicted_classes = []

    with torch.inference_mode():

        for i in range(
            0,
            total_patches,
            BATCH_SIZE
        ):

            batch = patches[
                i:i + BATCH_SIZE
            ]

            # Mixed precision
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=DEVICE.type == "cuda"
            ):

                outputs = model(batch)

            # Raw logits → predicted class
            predictions = torch.argmax(
                outputs,
                dim=1
            )

            predicted_classes.append(
                predictions.cpu()
            )

    predicted_classes = torch.cat(
        predicted_classes
    ).numpy()

    # --------------------------------------------------------
    # Count classes
    # --------------------------------------------------------

    counts = np.bincount(
        predicted_classes,
        minlength=class_num
    )

    cavity = (
        counts[0]
        if class_num > 0
        else 0
    )

    cavity_filled = (
        counts[1]
        if class_num > 1
        else 0
    )

    inertinite = (
        counts[2]
        if class_num > 2
        else 0
    )

    minerals = (
        counts[3]
        if class_num > 3
        else 0
    )

    vitrinite = (
        counts[4]
        if class_num > 4
        else 0
    )

    return (
        cavity,
        cavity_filled,
        inertinite,
        minerals,
        vitrinite
    )


# ============================================================
# Main
# ============================================================

print("=" * 70)
print("PYTORCH SLIDING WINDOW INFERENCE")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print()


# ============================================================
# Process every optimizer
# ============================================================

for optimizer_name, model_path in MODEL_PATHS.items():

    print("\n")
    print("=" * 70)
    print(
        f"Currently running: {optimizer_name}"
    )
    print("=" * 70)

    if not os.path.exists(model_path):

        print(
            f"WARNING: Model not found:\n"
            f"{model_path}"
        )

        continue

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model(
        model_path
    )

    # --------------------------------------------------------
    # Result directory
    # --------------------------------------------------------

    result_folder = os.path.join(
        RESULT_ROOT,
        optimizer_name
    )

    os.makedirs(
        result_folder,
        exist_ok=True
    )

    image_folder = os.path.join(
        result_folder,
        "images"
    )

    os.makedirs(
        image_folder,
        exist_ok=True
    )

    output_file = os.path.join(
        result_folder,
        "final_output_new_full.txt"
    )

    # --------------------------------------------------------
    # Output file
    # --------------------------------------------------------

    with open(
        output_file,
        "w"
    ) as f:

        # ----------------------------------------------------
        # Outer folders
        # ----------------------------------------------------

        outer_folders = sorted(
            os.listdir(IMAGE_ROOT)
        )

        for outer_folder in outer_folders:

            folder_path = os.path.join(
                IMAGE_ROOT,
                outer_folder
            )

            # Skip files
            if not os.path.isdir(
                folder_path
            ):
                continue

            # ------------------------------------------------
            # Only JPG files
            # ------------------------------------------------

            actual_files = [
                file_name
                for file_name in os.listdir(
                    folder_path
                )
                if file_name.lower().endswith(
                    ".jpg"
                )
            ]

            if len(actual_files) == 0:
                continue

            print(
                f"\nFolder: {outer_folder}"
            )

            print(
                f"Total images: "
                f"{len(actual_files)}"
            )

            total_mineral_percentage = 0.0

            # ------------------------------------------------
            # Process images
            # ------------------------------------------------

            for file_name in tqdm(
                actual_files,
                desc=outer_folder
            ):

                image_path = os.path.join(
                    folder_path,
                    file_name
                )

                # ------------------------------------------------
                # Read image
                #
                # cv2.imread gives uint8 [0,255],
                # which matches the original TF pipeline
                # before /255 normalization.
                # ------------------------------------------------

                img = cv2.imread(
                    image_path
                )

                if img is None:

                    print(
                        f"WARNING: Could not read "
                        f"{image_path}"
                    )

                    continue

                # OpenCV BGR -> RGB
                img = cv2.cvtColor(
                    img,
                    cv2.COLOR_BGR2RGB
                )

                # ------------------------------------------------
                # Black rectangle
                # ------------------------------------------------

                img = cv2.rectangle(
                    img,
                    (2146, 30),
                    (2572, 162),
                    (0, 0, 0),
                    -1
                )

                # ------------------------------------------------
                # Sliding window inference
                # ------------------------------------------------

                (
                    cavity,
                    cavity_filled,
                    inertinite,
                    minerals,
                    vitrinite
                ) = sliding_window_inference(
                    model,
                    img,
                    window_size=WINDOW_SIZE,
                    stride=STRIDE,
                    class_num=NUM_CLASSES
                )

                # ------------------------------------------------
                # Percentages
                # ------------------------------------------------

                total_number = (
                    cavity
                    + cavity_filled
                    + inertinite
                    + minerals
                    + vitrinite
                )

                if total_number == 0:
                    continue

                cavity_percentage = round(
                    cavity / total_number * 100,
                    2
                )

                cavity_filled_percentage = round(
                    cavity_filled / total_number * 100,
                    2
                )

                inertinite_percentage = round(
                    inertinite / total_number * 100,
                    2
                )

                minerals_percentage = round(
                    minerals / total_number * 100,
                    2
                )

                vitrinite_percentage = round(
                    vitrinite / total_number * 100,
                    2
                )

                # ------------------------------------------------
                # Mineral %
                # ------------------------------------------------

                total_mineral_percentage += (
                    minerals_percentage
                    + cavity_filled_percentage
                )

            # ----------------------------------------------------
            # Folder statistics
            # ----------------------------------------------------

            total_images = len(
                actual_files
            )

            average = (
                total_mineral_percentage
                / total_images
            )

            average_ash = (
                average / 1.1
            )

            print(
                f"Optimizer used = "
                f"{optimizer_name}"
            )

            print(
                f"Total mineral % = "
                f"{total_mineral_percentage}"
            )

            print(
                f"Average mineral % = "
                f"{average}"
            )

            print(
                f"Average ash % = "
                f"{average_ash}"
            )

            # ----------------------------------------------------
            # Save results
            # ----------------------------------------------------

            f.write(
                f"{outer_folder} "
                f"Total mineral %: "
                f"{total_mineral_percentage}\n"
            )

            f.write(
                f"{outer_folder} "
                f"Average mineral %: "
                f"{average}\n"
            )

            f.write(
                f"{outer_folder} "
                f"Average ash %: "
                f"{average_ash}\n"
            )

            f.write(
                "-" * 40
                + "\n"
            )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()

print("\nInference complete.")