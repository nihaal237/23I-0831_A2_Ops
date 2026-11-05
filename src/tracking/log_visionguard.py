
import os
import subprocess
from pathlib import Path

import mlflow
import pandas as pd

TRACKING_URI = (
    "https://dagshub.com/"
    "nihaal237/23I-0831_A2_Ops.mlflow"
)

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("VisionGuard_Model_Training")

def git_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True
    ).strip()

# These are previously completed training runs.
RUNS = [
    {
        "name": "CNN_Baseline",
        "history": "reports/cnn_history.csv",
        "learning_rate": 0.001,
        "architecture": "Custom CNN",
        "dropout": 0.4,
    },
    {
        "name": "MobileNetV2_Frozen",
        "history": "reports/mobilenetv2_head_history.csv",
        "learning_rate": 0.001,
        "architecture": "MobileNetV2",
        "dropout": 0.3,
    },
    {
        "name": "MobileNetV2_FineTuned",
        "history": "reports/mobilenetv2_finetune_history.csv",
        "learning_rate": 1e-5,
        "architecture": "MobileNetV2",
        "dropout": 0.3,
    },
    {
        "name": "MobileNetV2_Tuned",
        "history": "reports/mobilenetv2_tuned_history.csv",
        "learning_rate": 5e-5,
        "architecture": "MobileNetV2",
        "dropout": 0.2,
    },
]

# Read the dataset's DVC content hash.
dvc_pointer = Path("data/raw.dvc").read_text(
    encoding="utf-8"
)

import re
match = re.search(r"md5:\s*(\S+)", dvc_pointer)
dvc_hash = match.group(1) if match else "unknown"

for config in RUNS:
    history = pd.read_csv(config["history"])
    best_idx = history["val_loss"].idxmin()
    best = history.iloc[best_idx]

    with mlflow.start_run(run_name=config["name"]):
        mlflow.log_params({
            "architecture": config["architecture"],
            "learning_rate": config["learning_rate"],
            "dropout": config["dropout"],
            "seed": 42,
            "batch_size": 16,
            "image_size": 224,
        })

        mlflow.set_tags({
            "git_commit": git_commit(),
            "dvc_data_version": dvc_hash,
            "seed": "42",
            "dataset": "VisionGuard",
            "logging_method": "manual",
            "run_origin": "historical_training_import",
        })

        for epoch, row in history.iterrows():
            for metric in [
                "accuracy", "loss", "auc",
                "val_accuracy", "val_loss", "val_auc"
            ]:
                if metric in history.columns:
                    mlflow.log_metric(
                        metric,
                        float(row[metric]),
                        step=int(epoch + 1)
                    )

        mlflow.log_metric(
            "best_val_loss",
            float(best["val_loss"])
        )
        mlflow.log_metric(
            "best_val_accuracy",
            float(best["val_accuracy"])
        )
        mlflow.log_metric(
            "best_epoch",
            int(best_idx + 1)
        )

        mlflow.log_artifact(config["history"])

    print(f"Logged: {config['name']}")

print("\nHistorical VisionGuard runs logged successfully!")
