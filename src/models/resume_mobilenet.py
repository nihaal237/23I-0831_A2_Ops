
from pathlib import Path
import random
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.pipeline import create_dataset

SEED = 42
EPOCHS = 6

random.seed(SEED)
np.random.seed(SEED)

tf.random.set_seed(SEED)

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

# Load the completed Stage 1 model.
model = tf.keras.models.load_model(
    MODEL_DIR / "mobilenetv2_head.keras",
    compile=False
)

# Locate the nested MobileNetV2 backbone.
base_model = next(
    layer for layer in model.layers
    if isinstance(layer, tf.keras.Model)
    and "mobilenetv2" in layer.name.lower()
)

print("Loaded pretrained checkpoint successfully.")

# Stage 2: Unfreeze final backbone layers.
base_model.trainable = True

for layer in base_model.layers[:-20]:
    layer.trainable = False

# Keep BatchNormalization frozen.
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

trainable_layers = [
    layer.name
    for layer in base_model.layers
    if layer.trainable
]

print("\nFine-tuning layers:")
for name in trainable_layers:
    print(" -", name)

print("Trainable backbone layers:", len(trainable_layers))

REPORT_DIR.mkdir(exist_ok=True)
(REPORT_DIR / "mobilenetv2_trainable_layers.txt").write_text(
    "\n".join(trainable_layers),
    encoding="utf-8"
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
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
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_DIR / "mobilenetv2_finetuned.keras"),
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

print("\n========== RESUMED FINE-TUNING ==========")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=2
)

pd.DataFrame(history.history).to_csv(
    REPORT_DIR / "mobilenetv2_finetune_history.csv",
    index=False
)

head_history = pd.read_csv(
    REPORT_DIR / "mobilenetv2_head_history.csv"
)

head_loss = head_history["val_loss"].min()
finetune_loss = min(history.history["val_loss"])

if finetune_loss < head_loss:
    selected = MODEL_DIR / "mobilenetv2_finetuned.keras"
    selected_stage = "Fine-tuned"
else:
    selected = MODEL_DIR / "mobilenetv2_head.keras"
    selected_stage = "Frozen backbone"

shutil.copy2(
    selected,
    MODEL_DIR / "mobilenetv2_best.keras"
)

print("\n========== TRAINING COMPLETE ==========")
print("Stage 1 best validation loss:", head_loss)
print("Stage 2 best validation loss:", finetune_loss)
print("Selected model:", selected_stage)
print("Saved: models/mobilenetv2_best.keras")
