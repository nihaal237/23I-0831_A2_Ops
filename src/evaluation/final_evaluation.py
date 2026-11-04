
import os
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

from src.data.pipeline import create_dataset

tf.keras.utils.set_random_seed(42)

OUT = Path("reports")
FIGURES = OUT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

MODELS = {
    "Custom CNN": "models/cnn_best.keras",
    "MobileNetV2 Frozen": "models/mobilenetv2_head.keras",
    "MobileNetV2 Fine-tuned": "models/mobilenetv2_best.keras",
    "MobileNetV2 Tuned": "models/mobilenetv2_tuned.keras",
}

# Threshold chosen previously using validation data.
TUNED_THRESHOLD = float(
    pd.read_csv(OUT / "selected_threshold.csv")
    .iloc[0]["threshold"]
)

def get_predictions(model, split):
    ds = create_dataset(split, training=False)
    labels = np.concatenate([
        y.numpy() for _, y in ds
    ]).astype(int)
    probabilities = model.predict(ds, verbose=0).ravel()
    return labels, probabilities

def select_validation_threshold(labels, probabilities):
    candidates = []
    for threshold in np.round(np.arange(0.10, 0.901, 0.01), 2):
        pred = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            labels, pred, labels=[0, 1]
        ).ravel()
        candidates.append((
            balanced_accuracy_score(labels, pred),
            -fp,
            float(threshold)
        ))
    return max(candidates)[2]

results = []
roc_data = {}
selected_model_errors = None

for name, path in MODELS.items():
    print(f"\nEvaluating: {name}")

    model = tf.keras.models.load_model(path, compile=False)

    if name == "MobileNetV2 Tuned":
        threshold = TUNED_THRESHOLD
    else:
        y_val, p_val = get_predictions(model, "val")
        threshold = select_validation_threshold(y_val, p_val)

    # Test predictions are made once per model.
    y_test, p_test = get_predictions(model, "test")
    y_pred = (p_test >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    ).ravel()

    results.append({
        "Model": name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(
            y_test, y_pred
        ),
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "F1": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, p_test),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    })

    fpr, tpr, _ = roc_curve(y_test, p_test)
    roc_data[name] = (fpr, tpr)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center")

    ax.set_xticks([0, 1], ["Real", "AI"])
    ax.set_yticks([0, 1], ["Real", "AI"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{name} Confusion Matrix")
    fig.tight_layout()
    fig.savefig(
        FIGURES / (
            name.lower().replace(" ", "_").replace("-", "_")
            + "_confusion_matrix.png"
        ),
        dpi=200
    )
    plt.close(fig)

    if name == "MobileNetV2 Tuned":
        selected_model_errors = (y_test, p_test, y_pred)

    del model
    tf.keras.backend.clear_session()

# Save metrics
results_df = pd.DataFrame(results)
results_df.to_csv(OUT / "final_test_metrics.csv", index=False)

print("\n========== FINAL TEST RESULTS ==========")
print(results_df.round(4).to_string(index=False))

# Combined ROC curve
fig, ax = plt.subplots(figsize=(7, 6))
for name, (fpr, tpr) in roc_data.items():
    auc = results_df.loc[
        results_df["Model"] == name, "ROC-AUC"
    ].iloc[0]
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

ax.plot([0, 1], [0, 1], "k--", label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Test ROC Curves")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES / "final_test_roc.png", dpi=200)
plt.close(fig)

# Analyze misclassified tuned-MobileNetV2 images
test_info = pd.read_csv("data/splits/test.csv")
true, probs, predicted = selected_model_errors

assert len(test_info) == len(true)
assert np.array_equal(test_info["label"].to_numpy(), true)

errors = test_info.copy()
errors["predicted"] = predicted
errors["probability_ai"] = probs
errors = errors[true != predicted].copy()

errors["error_type"] = np.where(
    errors["label"] == 0,
    "False positive",
    "False negative"
)
errors.to_csv(OUT / "test_errors.csv", index=False)

samples = errors.head(10)

fig, axes = plt.subplots(2, 5, figsize=(16, 7))

for ax in axes.flat:
    ax.axis("off")

for ax, (_, record) in zip(axes.flat, samples.iterrows()):
    with Image.open(record["path"]) as image:
        ax.imshow(image.convert("RGB"))
    ax.set_title(
        f'{record["error_type"]}\n'
        f'P(AI)={record["probability_ai"]:.2f}',
        fontsize=10
    )
    ax.axis("off")

fig.suptitle("Tuned MobileNetV2: Misclassified Test Images")
fig.tight_layout()
fig.savefig(
    FIGURES / "test_misclassified_images.png",
    dpi=200
)
plt.close(fig)

print("\nMisclassified tuned-model images:", len(errors))
print("Saved final metrics, ROC, confusion matrices and errors.")
