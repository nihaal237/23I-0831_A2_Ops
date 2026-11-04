
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras_tuner as kt
import pandas as pd

SEED = 42
tf.keras.utils.set_random_seed(SEED)

# Load Fashion-MNIST
(x_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()

# Use a small, reproducible subset for the tool demonstration
rng = np.random.default_rng(SEED)
indices = rng.permutation(len(x_train))[:12000]

x = x_train[indices].astype("float32") / 255.0
y = y_train[indices]

# Separate training and validation sets
x_train, x_val = x[:10000], x[10000:]
y_train, y_val = y[:10000], y[10000:]


def build_model(hp):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            units=hp.Choice("dense_units", [64, 128, 256]),
            activation="relu"
        ),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    learning_rate = hp.Choice(
        "learning_rate",
        [1e-2, 1e-3, 1e-4]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=4,
    executions_per_trial=1,
    directory="tuner_results",
    project_name="fashion_mnist_toy",
    overwrite=True,
    seed=SEED
)

print("\n========== STARTING TUNING ==========")

tuner.search(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=3,
    batch_size=64,
    verbose=1
)

print("\n========== TUNING RESULTS ==========")

tuner.results_summary()

rows = []

for trial in tuner.oracle.trials.values():
    rows.append({
        "trial": trial.trial_id,
        "dense_units": trial.hyperparameters.get("dense_units"),
        "learning_rate": trial.hyperparameters.get("learning_rate"),
        "best_val_accuracy": trial.score
    })

results = pd.DataFrame(rows)
results = results.sort_values(
    "best_val_accuracy", ascending=False
)

output = Path("reports")
output.mkdir(exist_ok=True)

results.to_csv(
    output / "toy_tuner_results.csv",
    index=False
)

print("\n", results.to_string(index=False))

best_hp = tuner.get_best_hyperparameters(1)[0]

print("\nBest units:", best_hp.get("dense_units"))
print("Best learning rate:", best_hp.get("learning_rate"))
