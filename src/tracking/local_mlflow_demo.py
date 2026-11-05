
import os

os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf

SEED = 42

tf.keras.utils.set_random_seed(SEED)

# Use SQLite instead of the legacy filesystem backend.
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("VisionGuard_MLflow_Toy")

# Load Fashion-MNIST.
(x_train, y_train), _ = (
    tf.keras.datasets.fashion_mnist.load_data()
)

# Small subset for quick experimentation.
rng = np.random.default_rng(SEED)
indices = rng.permutation(len(x_train))[:6000]

x = x_train[indices].astype("float32") / 255.0
y = y_train[indices]

x_train, x_val = x[:5000], x[5000:]
y_train, y_val = y[:5000], y[5000:]


def build_model(learning_rate):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def run_experiment(learning_rate):
    tf.keras.utils.set_random_seed(SEED)

    with mlflow.start_run(
        run_name=f"lr_{learning_rate}"
    ):
        mlflow.log_param(
            "learning_rate", learning_rate
        )
        mlflow.log_param("epochs", 3)
        mlflow.log_param("batch_size", 64)
        mlflow.log_param("dense_units", 64)

        mlflow.set_tag(
            "experiment_type", "toy_demo"
        )
        mlflow.set_tag(
            "dataset", "Fashion-MNIST"
        )

        model = build_model(learning_rate)

        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=3,
            batch_size=64,
            verbose=1
        )

        # Log metrics for every epoch.
        for epoch in range(
            len(history.history["loss"])
        ):
            for metric, values in history.history.items():
                mlflow.log_metric(
                    metric,
                    float(values[epoch]),
                    step=epoch + 1
                )

        # Save accuracy plot.
        fig, ax = plt.subplots(figsize=(7, 4))

        ax.plot(
            history.history["accuracy"],
            label="Training"
        )
        ax.plot(
            history.history["val_accuracy"],
            label="Validation"
        )

        ax.set_title(
            f"Learning Rate: {learning_rate}"
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()

        mlflow.log_figure(
            fig, "accuracy_curve.png"
        )
        plt.close(fig)

        # Save the trained Keras model in MLflow.
        mlflow.tensorflow.log_model(
            model,
            name="toy_model"
        )

        best_accuracy = max(
            history.history["val_accuracy"]
        )

        print(
            f"\nLearning rate: {learning_rate}"
        )
        print(
            f"Best validation accuracy: "
            f"{best_accuracy:.4f}"
        )


# Two distinct experiments.
run_experiment(0.001)
run_experiment(0.0001)

print("\nBoth MLflow runs completed!")
print("Experiment: VisionGuard_MLflow_Toy")
