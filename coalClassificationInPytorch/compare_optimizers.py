import os
import pandas as pd
import matplotlib.pyplot as plt


BASE_RESULT_DIR = "./results_feb23_2026"

OPTIMIZER_NAMES = [
    "Adam",
    "Adagrad",
    "RMSprop",
    "Adadelta",
    "Nadam",
    "AdamW",
]


def load_histories():

    histories = {}

    for optimizer in OPTIMIZER_NAMES:

        csv_path = os.path.join(
            BASE_RESULT_DIR,
            optimizer,
            "training_history.csv"
        )

        if not os.path.exists(csv_path):
            print(
                f"WARNING: Missing {csv_path}"
            )
            continue

        histories[optimizer] = pd.read_csv(
            csv_path
        )

    return histories


def plot_accuracy(histories):

    plt.figure(figsize=(12, 7))

    for optimizer, history in histories.items():

        plt.plot(
            history["epoch"],
            history["val_accuracy"],
            label=optimizer
        )

    plt.title(
        "Validation Accuracy - Optimizer Comparison"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")

    plt.legend()
    plt.grid()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            BASE_RESULT_DIR,
            "all_optimizers_validation_accuracy.png"
        ),
        dpi=300
    )

    plt.close()


def plot_loss(histories):

    plt.figure(figsize=(12, 7))

    for optimizer, history in histories.items():

        plt.plot(
            history["epoch"],
            history["val_loss"],
            label=optimizer
        )

    plt.title(
        "Validation Loss - Optimizer Comparison"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")

    plt.legend()
    plt.grid()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            BASE_RESULT_DIR,
            "all_optimizers_validation_loss.png"
        ),
        dpi=300
    )

    plt.close()


def plot_train_accuracy(histories):

    plt.figure(figsize=(12, 7))

    for optimizer, history in histories.items():

        plt.plot(
            history["epoch"],
            history["accuracy"],
            label=optimizer
        )

    plt.title(
        "Training Accuracy - Optimizer Comparison"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy")

    plt.legend()
    plt.grid()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            BASE_RESULT_DIR,
            "all_optimizers_training_accuracy.png"
        ),
        dpi=300
    )

    plt.close()


def plot_train_loss(histories):

    plt.figure(figsize=(12, 7))

    for optimizer, history in histories.items():

        plt.plot(
            history["epoch"],
            history["loss"],
            label=optimizer
        )

    plt.title(
        "Training Loss - Optimizer Comparison"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")

    plt.legend()
    plt.grid()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            BASE_RESULT_DIR,
            "all_optimizers_training_loss.png"
        ),
        dpi=300
    )

    plt.close()


if __name__ == "__main__":

    histories = load_histories()

    if not histories:
        raise RuntimeError(
            "No training history CSV files found."
        )

    plot_accuracy(histories)
    plot_loss(histories)
    plot_train_accuracy(histories)
    plot_train_loss(histories)

    print(
        "\n✓ Optimizer comparison graphs generated."
    )