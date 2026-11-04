
import os

# Set CPU limits before importing TensorFlow.
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

from pathlib import Path
import random

import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.pipeline import create_dataset

SEED = 42
TRIALS = 8
EPOCHS = 3

random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

CHECKPOINT = Path("models/mobilenetv2_head.keras")
OUTPUT = Path("reports")
OUTPUT.mkdir(exist_ok=True)

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Missing Stage 1 checkpoint: {CHECKPOINT}"
    )

train_ds = create_dataset("train", training=True)
val_ds = create_dataset("val", training=False)


def build_model(hp):
    # Each trial starts from the same saved weights.
    model = tf.keras.models.load_model(
        CHECKPOINT,
        compile=False
    )

    base_model = next(
        layer for layer in model.layers
        if isinstance(layer, tf.keras.Model)
        and "mobilenetv2" in layer.name.lower()
    )

    learning_rate = hp.Choice(
        "learning_rate",
        [1e-5, 5e-5]
    )

    dropout = hp.Choice(
        "dropout",
        [0.2, 0.4]
    )

    unfrozen = hp.Choice(
        "unfrozen_layers",
        [10, 20]
    )

    base_model.trainable = True

    for layer in base_model.layers:
        layer.trainable = False

    for layer in base_model.layers[-unfrozen:]:
        if not isinstance(
            layer, tf.keras.layers.BatchNormalization
        ):
            layer.trainable = True

    # Update the existing dropout layer.
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dropout):
            layer.rate = dropout

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

    return model


tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective=kt.Objective(
        "val_loss", direction="min"
    ),
    max_trials=TRIALS,
    directory="tuner_results",
    project_name="visionguard_mobilenet",
    overwrite=True,
    seed=SEED,
    max_consecutive_failed_trials=3
)

print("\n========== VISIONGUARD TUNING ==========")

tuner.search(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=2,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2
        )
    ]
)

rows = []

for trial in tuner.oracle.trials.values():
    rows.append({
        "trial": trial.trial_id,
        "status": trial.status,
        "learning_rate": trial.hyperparameters.get(
            "learning_rate"
        ),
        "dropout": trial.hyperparameters.get("dropout"),
        "unfrozen_layers": trial.hyperparameters.get(
            "unfrozen_layers"
        ),
        "best_val_loss": trial.score
    })

results = pd.DataFrame(rows)

results = results.sort_values(
    "best_val_loss",
    ascending=True,
    na_position="last"
)

results.to_csv(
    OUTPUT / "visionguard_tuning_results.csv",
    index=False
)

print("\n========== FINAL TUNING RESULTS ==========")
print(results.to_string(index=False))

completed = results[results["status"] == "COMPLETED"]

if len(completed) < TRIALS:
    raise RuntimeError(
        f"Only {len(completed)} of {TRIALS} trials completed."
    )

best_hp = tuner.get_best_hyperparameters(1)[0]

print("\nBEST CONFIGURATION")
print("Learning rate:", best_hp.get("learning_rate"))
print("Dropout:", best_hp.get("dropout"))
print(
    "Unfrozen layers:",
    best_hp.get("unfrozen_layers")
)
