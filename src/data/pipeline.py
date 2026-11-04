
from pathlib import Path

import pandas as pd
import tensorflow as tf

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

AUTOTUNE = tf.data.AUTOTUNE

# Image augmentation is applied only to training samples.
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
], name="training_augmentation")


def load_image(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )
    image.set_shape([None, None, 3])

    # Resize with padding to preserve aspect ratio.
    image = tf.image.resize_with_pad(
        image, IMAGE_SIZE[0], IMAGE_SIZE[1]
    )

    # Pixel range [0, 1] for the custom CNN.
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


def create_dataset(split, training=False):
    csv_path = Path("data/splits") / f"{split}.csv"
    df = pd.read_csv(csv_path)

    paths = df["path"].astype(str).tolist()
    labels = df["label"].astype("float32").tolist()

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(df),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=AUTOTUNE
    )

    if training:
        dataset = dataset.map(
            lambda image, label: (
                augmentation(image, training=True),
                label
            ),
            num_parallel_calls=AUTOTUNE
        )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


if __name__ == "__main__":
    train_ds = create_dataset("train", training=True)
    val_ds = create_dataset("val")
    test_ds = create_dataset("test")

    for images, labels in train_ds.take(1):
        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)
        print("Pixel range:",
              float(tf.reduce_min(images).numpy()),
              float(tf.reduce_max(images).numpy()))

    print("Training, validation, and test datasets ready.")
