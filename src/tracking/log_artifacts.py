
import re
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

MODEL_PATH = Path("models/mobilenetv2_tuned.keras")
FIGURES = Path("reports/figures")
SPLITS = Path("data/splits")

files = {
    "model": MODEL_PATH,
    "evaluation": FIGURES / "mobilenetv2_tuned_confusion_matrix.png",
    "roc": FIGURES / "final_test_roc.png",
    "splits_train": SPLITS / "train.csv",
    "splits_val": SPLITS / "val.csv",
    "splits_test": SPLITS / "test.csv",
    "environment": Path("requirements.txt"),
    "metrics": Path("reports/final_test_metrics.csv"),
}

# Validate files before creating the remote run.
for name, path in files.items():
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}")

git_hash = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    text=True
).strip()

pointer_text = Path("data/raw.dvc").read_text(
    encoding="utf-8"
)
match = re.search(r"md5:\s*(\S+)", pointer_text)
dvc_hash = match.group(1) if match else "unknown"

metrics = pd.read_csv(files["metrics"])
tuned = metrics.loc[
    metrics["Model"] == "MobileNetV2 Tuned"
].iloc[0]

with mlflow.start_run(
    run_name="MobileNetV2_Tuned_Artifacts"
):
    mlflow.log_params({
        "architecture": "MobileNetV2",
        "learning_rate": 5e-5,
        "dropout": 0.2,
        "unfrozen_layers": 20,
        "image_size": 224,
        "batch_size": 16,
        "seed": 42,
        "threshold": 0.70,
    })

    mlflow.set_tags({
        "purpose": "model_artifact_packaging",
        "git_commit_at_logging": git_hash,
        "dvc_data_version": dvc_hash,
        "dataset": "VisionGuard",
        "model_selection": "validation_loss",
        "source": "previously_trained_checkpoint",
    })

    mlflow.log_metrics({
        "test_accuracy": float(tuned["Accuracy"]),
        "test_precision": float(tuned["Precision"]),
        "test_recall": float(tuned["Recall"]),
        "test_f1": float(tuned["F1"]),
        "test_roc_auc": float(tuned["ROC-AUC"]),
    })

    print("Uploading trained model...")
    mlflow.log_artifact(
        str(files["model"]),
        artifact_path="model_checkpoint"
    )

    print("Uploading evaluation figures...")
    mlflow.log_artifact(
        str(files["evaluation"]),
        artifact_path="evaluation"
    )
    mlflow.log_artifact(
        str(files["roc"]),
        artifact_path="evaluation"
    )
    mlflow.log_artifact(
        str(files["metrics"]),
        artifact_path="evaluation"
    )

    print("Uploading dataset split lists...")
    for key in ("splits_train", "splits_val", "splits_test"):
        mlflow.log_artifact(
            str(files[key]),
            artifact_path="data_splits"
        )

    print("Uploading environment information...")
    mlflow.log_artifact(
        str(files["environment"]),
        artifact_path="environment"
    )

print("\nAll VisionGuard artifacts uploaded successfully!")
