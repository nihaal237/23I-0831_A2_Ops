
from pathlib import Path

import pandas as pd

from app.inference import VisionGuardPredictor

predictor = VisionGuardPredictor()

df = pd.read_csv("data/splits/val.csv")

for label, name in [
    (0, "Real"),
    (1, "AI-generated")
]:
    sample = df[df["label"] == label].iloc[0]
    image_path = Path(sample["path"])

    result = predictor.predict(
        image_path.read_bytes()
    )

    print("\n============================")
    print("Actual class:", name)
    print("Image:", image_path.name)
    print("Prediction:", result["prediction"])
    print("AI probability:", result["ai_probability"])
    print("Threshold:", result["threshold"])
    print("============================")

print("\nInference module tested successfully!")
