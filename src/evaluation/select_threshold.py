
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from src.data.pipeline import create_dataset

MODEL_PATH = "models/mobilenetv2_tuned.keras"
OUT = Path("reports")
OUT.mkdir(exist_ok=True)

print("Loading tuned MobileNetV2...")

model = tf.keras.models.load_model(
    MODEL_PATH, compile=False
)

# Validation data only — do not use the test set.
val_ds = create_dataset("val", training=False)

y_true = np.concatenate([
    labels.numpy() for _, labels in val_ds
]).astype(int)

y_prob = model.predict(val_ds, verbose=1).ravel()

results = []

for threshold in np.arange(0.10, 0.901, 0.01):
    threshold = round(float(threshold), 2)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    results.append({
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    })

df = pd.DataFrame(results)

# Select the threshold with highest balanced accuracy.
# For ties, prefer fewer false positives.
best = df.sort_values(
    ["balanced_accuracy", "false_positives"],
    ascending=[False, True]
).iloc[0]

df.to_csv(
    OUT / "threshold_search.csv", index=False
)

summary = pd.DataFrame([best])
summary.to_csv(
    OUT / "selected_threshold.csv", index=False
)

print("\n========== THRESHOLD SELECTION ==========")
print("Model:", MODEL_PATH)
print("Validation samples:", len(y_true))
print("Validation ROC-AUC:",
      round(roc_auc_score(y_true, y_prob), 4))

print("\nDefault threshold (0.50):")
print(
    df[np.isclose(df["threshold"], 0.50)]
    .to_string(index=False)
)

print("\nSelected threshold:")
print(summary.to_string(index=False))

print("\nSaved:")
print("reports/threshold_search.csv")
print("reports/selected_threshold.csv")
