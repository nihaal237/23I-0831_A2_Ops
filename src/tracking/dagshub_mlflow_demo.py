
import os

os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

import mlflow
import tensorflow as tf
import numpy as np
from pathlib import Path

tf.keras.utils.set_random_seed(42)

TRACKING_URI = (
    "https://dagshub.com/"
    "nihaal237/23I-0831_A2_Ops.mlflow"
)

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("VisionGuard_DagsHub_Toy")

# Load a small Fashion-MNIST subset.
(x_train, y_train), _ = (
    tf.keras.datasets.fashion_mnist.load_data()
)

rng = np.random.default_rng(42)
indices = rng.permutation(len(x_train))[:3000]

x = x_train[indices].astype("float32") / 255.0
y = y_train[indices]

model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

with mlflow.start_run(run_name="dagshub_connection_test"):
    mlflow.log_params({
        "learning_rate": 0.001,
        "epochs": 2,
        "batch_size": 64,
        "dataset": "Fashion-MNIST"
    })

    mlflow.set_tag("purpose", "remote_tracking_demo")

    history = model.fit(
        x[:2500], y[:2500],
        validation_data=(x[2500:], y[2500:]),
        epochs=2,
        batch_size=64,
        verbose=1
    )

    for epoch in range(2):
        for metric, values in history.history.items():
            mlflow.log_metric(
                metric, float(values[epoch]),
                step=epoch + 1
            )

    print(
        "\nFinal validation accuracy:",
        history.history["val_accuracy"][-1]
    )

print("\nSuccessfully logged experiment to DagsHub!")
