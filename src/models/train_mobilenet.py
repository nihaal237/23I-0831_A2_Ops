
from pathlib import Path
import random
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.pipeline import create_dataset

SEED = 42
HEAD_EPOCHS = 8
FINETUNE_EPOCHS = 6

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

train_ds = create_dataset("train", training=True)
val_ds = create_dataset("val", training=False)

# 1. Load ImageNet-pretrained MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
    alpha=0.35
)

base_model.trainable = False

# 2. Build classification model
inputs = tf.keras.Input(shape=(224, 224, 3))

# Our pipeline produces [0, 1].
# MobileNetV2 expects [-1, 1].
x = tf.keras.layers.Rescaling(
    scale=2.0, offset=-1.0
)(inputs)

x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(
    1, activation="sigmoid"
)(x)

model = tf.keras.Model(
    inputs, outputs, name="VisionGuard_MobileNetV2"
)


def compile_model(learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc")
        ]
    )


def callbacks(filepath):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(filepath),
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


# ==========================================
# STAGE 1 — FEATURE EXTRACTION
# ==========================================

print("\n========== STAGE 1: FROZEN BACKBONE ==========")

compile_model(learning_rate=0.001)
model.summary()

print("Trainable backbone layers: 0")

history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=HEAD_EPOCHS,
    callbacks=callbacks(
        MODEL_DIR / "mobilenetv2_head.keras"
    )
)

pd.DataFrame(history_head.history).to_csv(
    REPORT_DIR / "mobilenetv2_head_history.csv",
    index=False
)

head_best_loss = min(
    history_head.history["val_loss"]
)

print("Stage 1 best validation loss:", head_best_loss)


# ==========================================
# STAGE 2 — FINE-TUNING
# ==========================================

print("\n========== STAGE 2: FINE-TUNING ==========")

base_model.trainable = True

# Freeze earlier layers.
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Keep BatchNorm frozen for stable fine-tuning.
for layer in base_model.layers:
    if isinstance(
        layer, tf.keras.layers.BatchNormalization
    ):
        layer.trainable = False

trainable_layers = [
    layer.name
    for layer in base_model.layers
    if layer.trainable
]

print("\nTrainable backbone layers:")
for name in trainable_layers:
    print(" -", name)

print("Total trainable backbone layers:",
      len(trainable_layers))

(REPORT_DIR / "mobilenetv2_trainable_layers.txt").write_text(
    "\n".join(trainable_layers),
    encoding="utf-8"
)

# Recompile after changing trainable layers.
compile_model(learning_rate=1e-5)

history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINETUNE_EPOCHS,
    callbacks=callbacks(
        MODEL_DIR / "mobilenetv2_finetuned.keras"
    )
)

pd.DataFrame(history_finetune.history).to_csv(
    REPORT_DIR / "mobilenetv2_finetune_history.csv",
    index=False
)

finetune_best_loss = min(
    history_finetune.history["val_loss"]
)

# Choose the checkpoint with lowest validation loss.
if finetune_best_loss < head_best_loss:
    source = MODEL_DIR / "mobilenetv2_finetuned.keras"
    selected_stage = "Fine-tuned"
else:
    source = MODEL_DIR / "mobilenetv2_head.keras"
    selected_stage = "Frozen backbone"

shutil.copy2(
    source,
    MODEL_DIR / "mobilenetv2_best.keras"
)

print("\n========== TRAINING COMPLETE ==========")
print("Stage 1 best validation loss:", head_best_loss)
print("Stage 2 best validation loss:", finetune_best_loss)
print("Selected model:", selected_stage)
print("Saved: models/mobilenetv2_best.keras")
