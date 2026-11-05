
import os
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

import re
import subprocess
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.pipeline import create_dataset

SEED = 42
EPOCHS = 3
LEARNING_RATES = [1e-5, 5e-5]

mlflow.set_tracking_uri(
    "https://dagshub.com/nihaal237/23I-0831_A2_Ops.mlflow"
)
mlflow.set_experiment("VisionGuard_Model_Training")

commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], text=True
).strip()

pointer = Path("data/raw.dvc").read_text(encoding="utf-8")
dvc_hash = re.search(r"md5:\s*(\S+)", pointer).group(1)

results = []

for lr in LEARNING_RATES:
    tf.keras.utils.set_random_seed(SEED)

    model = tf.keras.models.load_model(
        "models/mobilenetv2_head.keras",
        compile=False
    )

    backbone = next(
        layer for layer in model.layers
        if isinstance(layer, tf.keras.Model)
        and "mobilenetv2" in layer.name.lower()
    )

    backbone.trainable = True

    for layer in backbone.layers:
        layer.trainable = False

    for layer in backbone.layers[-20:]:
        if not isinstance(
            layer, tf.keras.layers.BatchNormalization
        ):
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    train_ds = create_dataset("train", training=True)
    val_ds = create_dataset("val", training=False)

    run_name = f"Controlled_MobileNet_LR_{lr}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "architecture": "MobileNetV2",
            "learning_rate": lr,
            "dropout": 0.3,
            "last_layers_considered": 20,
            "epochs": EPOCHS,
            "batch_size": 16,
            "seed": SEED
        })

        mlflow.set_tags({
            "experiment_type": "controlled_lr_comparison",
            "git_commit_at_logging": commit,
            "dvc_data_version": dvc_hash,
            "initial_checkpoint": "mobilenetv2_head.keras"
        })

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            verbose=2
        )

        for epoch, row in pd.DataFrame(
            history.history
        ).iterrows():
            for metric, value in row.items():
                mlflow.log_metric(
                    metric, float(value), step=epoch + 1
                )

        best_idx = int(
            np.argmin(history.history["val_loss"])
        )

        results.append({
            "run": run_name,
            "learning_rate": lr,
            "dropout": 0.3,
            "best_epoch": best_idx + 1,
            "val_accuracy": history.history[
                "val_accuracy"
            ][best_idx],
            "val_auc": history.history[
                "val_auc"
            ][best_idx],
            "val_loss": history.history[
                "val_loss"
            ][best_idx]
        })

    del model
    tf.keras.backend.clear_session()

output = Path("reports/controlled_mlflow_comparison.csv")
df = pd.DataFrame(results)
df.to_csv(output, index=False)

print("\n========== CONTROLLED EXPERIMENTS ==========")
print(df.to_string(index=False))
print("\nSaved:", output)
