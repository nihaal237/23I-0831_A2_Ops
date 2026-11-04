
import os

# Configure CPU threads before importing TensorFlow.
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

from pathlib import Path
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.pipeline import create_dataset

SEED = 42
EPOCHS = 12
LEARNING_RATE = 5e-5
DROPOUT = 0.2
UNFROZEN_LAYERS = 20

random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# Load the frozen-backbone checkpoint.
model = tf.keras.models.load_model(
    MODEL_DIR / "mobilenetv2_head.keras",
    compile=False
)

# Locate MobileNetV2 backbone.
base_model = next(
    layer for layer in model.layers
    if isinstance(layer, tf.keras.Model)
    and "mobilenetv2" in layer.name.lower()
)

# Unfreeze only the final backbone block.
base_model.trainable = True

for layer in base_model.layers:
    layer.trainable = False

for layer in base_model.layers[-UNFROZEN_LAYERS:]:
    if not isinstance(
        layer, tf.keras.layers.BatchNormalization
    ):
        layer.trainable = True

# Apply tuned dropout.
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Dropout):
        layer.rate = DROPOUT

trainable = [
    layer.name
    for layer in base_model.layers
    if layer.trainable
]

print("\n========== TUNED CONFIGURATION ==========")
print("Learning rate:", LEARNING_RATE)
print("Dropout:", DROPOUT)
print("Unfrozen backbone layers:", len(trainable))
print("Epochs:", EPOCHS)

# Compile after changing trainable layers.
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

train_ds = create_dataset("train", training=True)
val_ds = create_dataset("val", training=False)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_DIR / "mobilenetv2_tuned.keras"),
        monitor="val_loss",
        save_best_only=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7
    )
]

print("\n========== TUNED MODEL TRAINING ==========")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=2
)

history_df = pd.DataFrame(history.history)
history_df.to_csv(
    REPORT_DIR / "mobilenetv2_tuned_history.csv",
    index=False
)

best_idx = history_df["val_loss"].idxmin()
best_row = history_df.iloc[best_idx]

print("\n========== TRAINING COMPLETE ==========")
print("Best epoch:", best_idx + 1)
print("Best validation accuracy:",
      round(float(best_row["val_accuracy"]), 4))
print("Best validation AUC:",
      round(float(best_row["val_auc"]), 4))
print("Best validation loss:",
      round(float(best_row["val_loss"]), 4))

print("Saved: models/mobilenetv2_tuned.keras")
