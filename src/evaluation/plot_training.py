
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

history = pd.read_csv("reports/cnn_history.csv")

output_dir = Path("reports/figures")
output_dir.mkdir(parents=True, exist_ok=True)

epochs = range(1, len(history) + 1)
best_epoch = history["val_loss"].idxmin() + 1

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy
axes[0].plot(epochs, history["accuracy"],
             marker="o", label="Training")
axes[0].plot(epochs, history["val_accuracy"],
             marker="o", label="Validation")
axes[0].axvline(best_epoch, linestyle="--",
                color="red", label=f"Best epoch: {best_epoch}")
axes[0].set_title("CNN Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Loss
axes[1].plot(epochs, history["loss"],
             marker="o", label="Training")
axes[1].plot(epochs, history["val_loss"],
             marker="o", label="Validation")
axes[1].axvline(best_epoch, linestyle="--",
                color="red", label=f"Best epoch: {best_epoch}")
axes[1].set_title("CNN Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    output_dir / "cnn_training_curves.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()

print("Training curves saved successfully.")
print(f"Best epoch: {best_epoch}")
