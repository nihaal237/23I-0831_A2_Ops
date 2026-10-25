
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

df = pd.read_csv("reports/dataset_audit.csv")
df = df[df["corrupt"] == False].copy()

OUT = Path("reports/figures")
OUT.mkdir(parents=True, exist_ok=True)

classes = ["AI-generated", "Real"]

# Figure 1: Sample images from each class
fig, axes = plt.subplots(2, 5, figsize=(15, 7))

for row, label in enumerate(classes):
    subset = df[df["class"] == label]
    samples = subset.sample(n=5, random_state=42)

    for col, (_, record) in enumerate(samples.iterrows()):
        with Image.open(record["path"]) as img:
            image = img.convert("RGB").copy()

        axes[row, col].imshow(image)
        axes[row, col].axis("off")

        if col == 0:
            axes[row, col].set_title(label, fontsize=12)

plt.suptitle("Sample Images: AI-Generated vs Real", fontsize=16)
plt.tight_layout()
plt.savefig(OUT / "sample_images.png", dpi=200)
plt.close()

# Figures 2–4: Histograms
features = {
    "width": ("Image Width", "Pixels"),
    "height": ("Image Height", "Pixels"),
    "file_size_bytes": ("File Size", "Bytes"),
}

for feature, (title, xlabel) in features.items():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, label in zip(axes, classes):
        values = df.loc[df["class"] == label, feature].dropna()

        ax.hist(values, bins=30, edgecolor="black", alpha=0.75)
        ax.set_title(label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of Images")
        ax.axvline(
            values.median(),
            color="red",
            linestyle="--",
            label=f"Median: {values.median():.0f}",
        )
        ax.legend()

    fig.suptitle(f"{title} Distribution by Class")
    plt.tight_layout()
    plt.savefig(OUT / f"{feature}_histogram.png", dpi=200)
    plt.close()

print("EDA completed successfully!")
print("Generated figures:")
for file in sorted(OUT.glob("*.png")):
    print(" -", file)
