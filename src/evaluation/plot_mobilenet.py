
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT = Path("reports/figures")
OUT.mkdir(parents=True, exist_ok=True)

stages = {
    "Frozen Backbone": "reports/mobilenetv2_head_history.csv",
    "Fine-Tuning": "reports/mobilenetv2_finetune_history.csv",
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for stage, path in stages.items():
    df = pd.read_csv(path)
    epochs = range(1, len(df) + 1)

    best_epoch = df["val_loss"].idxmin() + 1

    axes[0].plot(
        epochs, df["accuracy"],
        marker="o", label=f"{stage} Train"
    )
    axes[0].plot(
        epochs, df["val_accuracy"],
        marker="o", linestyle="--",
        label=f"{stage} Validation"
    )

    axes[1].plot(
        epochs, df["loss"],
        marker="o", label=f"{stage} Train"
    )
    axes[1].plot(
        epochs, df["val_loss"],
        marker="o", linestyle="--",
        label=f"{stage} Validation"
    )

    print(
        f"{stage}: best epoch={best_epoch}, "
        f"val_loss={df['val_loss'].min():.4f}, "
        f"val_accuracy={df.loc[best_epoch - 1, 'val_accuracy']:.4f}"
    )

axes[0].set_title("MobileNetV2 Accuracy")
axes[0].set_xlabel("Epoch within stage")
axes[0].set_ylabel("Accuracy")

axes[1].set_title("MobileNetV2 Loss")
axes[1].set_xlabel("Epoch within stage")
axes[1].set_ylabel("Loss")

for ax in axes:
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    OUT / "mobilenetv2_training_curves.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()

print("Saved: reports/figures/mobilenetv2_training_curves.png")
