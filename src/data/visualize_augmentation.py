
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pipeline import create_dataset

train_ds = create_dataset("train", training=True)

images, labels = next(iter(train_ds))

fig, axes = plt.subplots(2, 4, figsize=(12, 7))

for i, ax in enumerate(axes.flat):
    image = np.clip(images[i].numpy(), 0, 1)
    ax.imshow(image)

    label = "AI-generated" if labels[i].numpy() == 1 else "Real"
    ax.set_title(label)
    ax.axis("off")

fig.suptitle(
    "VisionGuard: Augmented Training Images",
    fontsize=16
)

plt.tight_layout()

output = Path("reports/figures")
output.mkdir(parents=True, exist_ok=True)

plt.savefig(
    output / "augmented_training_batch.png",
    dpi=200
)

plt.close()

print("Saved augmented training batch successfully.")
