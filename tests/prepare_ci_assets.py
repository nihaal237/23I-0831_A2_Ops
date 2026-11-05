
from pathlib import Path
import csv

import numpy as np
import tensorflow as tf
from PIL import Image

tf.keras.utils.set_random_seed(42)

# Temporary lightweight model for CI tests only.
model = tf.keras.Sequential([
    tf.keras.Input(shape=(224, 224, 3)),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True)

model.save(model_dir / "mobilenetv2_tuned.keras")

# Create a valid sample image and split file.
data_dir = Path("data/ci")
split_dir = Path("data/splits")

data_dir.mkdir(parents=True, exist_ok=True)
split_dir.mkdir(parents=True, exist_ok=True)

image_path = data_dir / "sample.jpg"

pixels = np.full(
    (64, 64, 3), 128, dtype=np.uint8
)
Image.fromarray(pixels).save(image_path)

with (split_dir / "val.csv").open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.DictWriter(
        file, fieldnames=["path", "label"]
    )
    writer.writeheader()
    writer.writerow({
        "path": str(image_path),
        "label": 0
    })

print("CI test assets created successfully.")
