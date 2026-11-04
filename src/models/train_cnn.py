
from pathlib import Path
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.pipeline import create_dataset

SEED = 42
EPOCHS = 20
LEARNING_RATE = 0.001

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


def build_cnn():
    inputs = tf.keras.Input(shape=(224, 224, 3))

    x = tf.keras.layers.Conv2D(
        16, 3, padding="same", activation="relu"
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(
        32, 3, padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(
        64, 3, padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    outputs = tf.keras.layers.Dense(
        1, activation="sigmoid"
    )(x)

    model = tf.keras.Model(
        inputs, outputs, name="VisionGuard_CNN"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    return model


def main():
    train_ds = create_dataset("train", training=True)
    val_ds = create_dataset("val", training=False)

    model = build_cnn()

    print("\n========== MODEL SUMMARY ==========")
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / "cnn_best.keras"),
            monitor="val_loss",
            save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6
        ),
        tf.keras.callbacks.CSVLogger(
            str(REPORT_DIR / "cnn_training_log.csv")
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    pd.DataFrame(history.history).to_csv(
        REPORT_DIR / "cnn_history.csv",
        index=False
    )

    best_epoch = int(
        np.argmin(history.history["val_loss"]) + 1
    )

    print("\n========== TRAINING COMPLETE ==========")
    print("Best epoch:", best_epoch)
    print(
        "Best validation loss:",
        min(history.history["val_loss"])
    )
    print(
        "Validation accuracy at best epoch:",
        history.history["val_accuracy"][best_epoch - 1]
    )
    print("Saved model: models/cnn_best.keras")


if __name__ == "__main__":
    main()
