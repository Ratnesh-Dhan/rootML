from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))


# ============================================================
# CONFIG
# ============================================================
from configs.config import CHECKPOINT_DIR
PLOT_DIR = CHECKPOINT_DIR / "plots"

OPTIMIZERS = [
    "adamw",
    "adam",
    "sgd",
    "rmsprop",
]


# ============================================================
# LOAD RESULTS
# ============================================================

def load_results():
    results = {}

    for optimizer in OPTIMIZERS:
        csv_path = CHECKPOINT_DIR / optimizer / "metrics.csv"

        if not csv_path.exists():
            print(f"[WARNING] Missing: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        # Normalize optimizer name for display
        display_name = optimizer.upper()

        results[display_name] = df

        print(
            f"[OK] {display_name:<8} "
            f"{len(df):>3} epochs | "
            f"best mIoU = {df['val_miou'].max():.4f}"
        )

    if not results:
        raise RuntimeError("No metrics.csv files found.")

    return results


# ============================================================
# GENERIC PLOT
# ============================================================

def plot_metric(
    results,
    column,
    ylabel,
    title,
    filename,
):
    plt.figure(figsize=(12, 7))

    for optimizer, df in results.items():
        plt.plot(
            df["epoch"],
            df[column],
            linewidth=2,
            label=optimizer,
        )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    output_path = PLOT_DIR / filename
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"[SAVED] {output_path}")


# ============================================================
# BEST mIOU BAR CHART
# ============================================================

def plot_best_miou(results):
    optimizers = []
    best_scores = []

    for optimizer, df in results.items():
        optimizers.append(optimizer)
        best_scores.append(df["val_miou"].max())

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        optimizers,
        best_scores,
    )

    plt.xlabel("Optimizer")
    plt.ylabel("Best Validation mIoU")
    plt.title("Best Validation mIoU by Optimizer")

    plt.ylim(0, 1)

    # Display values above bars
    for bar, score in zip(bars, best_scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{score:.4f}",
            ha="center",
            fontsize=11,
        )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    output_path = PLOT_DIR / "best_miou_comparison.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"[SAVED] {output_path}")


# ============================================================
# SUMMARY CSV
# ============================================================

def create_summary(results):
    rows = []

    for optimizer, df in results.items():

        best_idx = df["val_miou"].idxmax()
        best_row = df.loc[best_idx]

        rows.append(
            {
                "optimizer": optimizer,
                "best_epoch": int(best_row["epoch"]),
                "best_val_miou": best_row["val_miou"],
                "train_miou_at_best": best_row["train_miou"],
                "val_loss_at_best": best_row["val_loss"],
                "train_loss_at_best": best_row["train_loss"],
            }
        )

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        "best_val_miou",
        ascending=False,
    )

    output_path = PLOT_DIR / "optimizer_summary.csv"

    summary.to_csv(
        output_path,
        index=False,
    )

    print(f"[SAVED] {output_path}")

    print("\n" + "=" * 70)
    print("OPTIMIZER SUMMARY")
    print("=" * 70)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("LOADING OPTIMIZER RESULTS")
    print("=" * 70)

    results = load_results()

    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    # 1. Validation mIoU
    plot_metric(
        results,
        column="val_miou",
        ylabel="Validation mIoU",
        title="Validation mIoU vs Epoch",
        filename="val_miou_vs_epoch.png",
    )

    # 2. Training mIoU
    plot_metric(
        results,
        column="train_miou",
        ylabel="Training mIoU",
        title="Training mIoU vs Epoch",
        filename="train_miou_vs_epoch.png",
    )

    # 3. Validation loss
    plot_metric(
        results,
        column="val_loss",
        ylabel="Validation Loss",
        title="Validation Loss vs Epoch",
        filename="val_loss_vs_epoch.png",
    )

    # 4. Training loss
    plot_metric(
        results,
        column="train_loss",
        ylabel="Training Loss",
        title="Training Loss vs Epoch",
        filename="train_loss_vs_epoch.png",
    )

    # 5. Best mIoU comparison
    plot_best_miou(results)

    # 6. Summary CSV
    create_summary(results)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()