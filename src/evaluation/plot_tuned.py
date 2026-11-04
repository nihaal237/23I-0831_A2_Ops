
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("reports/mobilenetv2_tuned_history.csv")
epochs = range(1, len(df) + 1)
best_epoch = df["val_loss"].idxmin() + 1

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, metric, ylabel in [
    (axes[0], "accuracy", "Accuracy"),
    (axes[1], "loss", "Loss")
]:
    ax.plot(epochs, df[metric], marker="o", label="Training")
    ax.plot(
        epochs, df[f"val_{metric}"],
        marker="o", label="Validation"
    )
    ax.axvline(
        best_epoch, color="red", linestyle="--",
        label=f"Best epoch: {best_epoch}"
    )
    ax.set_title(f"Tuned MobileNetV2 {ylabel}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()

output = Path("reports/figures/tuned_mobilenet_curves.png")
output.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(output, dpi=200)
plt.close()

print("Saved:", output)
print("Best epoch:", best_epoch)
