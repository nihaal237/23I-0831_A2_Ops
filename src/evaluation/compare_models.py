
import pandas as pd
from pathlib import Path

histories = {
    "Custom CNN": "reports/cnn_history.csv",
    "MobileNetV2 Frozen": "reports/mobilenetv2_head_history.csv",
    "MobileNetV2 Fine-tuned": "reports/mobilenetv2_finetune_history.csv",
}

results = []

for model_name, path in histories.items():
    df = pd.read_csv(path)

    best_idx = df["val_loss"].idxmin()
    row = df.iloc[best_idx]

    results.append({
        "Model": model_name,
        "Best Epoch": best_idx + 1,
        "Validation Accuracy": round(float(row["val_accuracy"]), 4),
        "Validation AUC": round(float(row["val_auc"]), 4),
        "Validation Loss": round(float(row["val_loss"]), 4),
    })

comparison = pd.DataFrame(results)
comparison = comparison.sort_values("Validation Loss")

Path("reports").mkdir(exist_ok=True)
comparison.to_csv(
    "reports/model_validation_comparison.csv",
    index=False
)

print("\n========== MODEL COMPARISON ==========")
print(comparison.to_string(index=False))
