
import os

os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

import mlflow
import mlflow.tensorflow
import tensorflow as tf
from PIL import Image

MODEL_NAME = "VisionGuard_Image_Classifier"

mlflow.set_tracking_uri(
    "https://dagshub.com/nihaal237/23I-0831_A2_Ops.mlflow"
)

mlflow.set_registry_uri(mlflow.get_tracking_uri())

model_uri = f"models:/{MODEL_NAME}@champion"

print("Loading:", model_uri)

model = mlflow.tensorflow.load_model(model_uri)

# Demonstration using one existing validation image.
# The validation set is not used to update the model.
import pandas as pd

path = pd.read_csv(
    "data/splits/val.csv"
).iloc[0]["path"]

with Image.open(path) as img:
    image = tf.convert_to_tensor(
        __import__("numpy").array(img.convert("RGB"))
    )

image = tf.image.resize_with_pad(
    image, 224, 224
)

image = tf.cast(image, tf.float32) / 255.0
image = tf.expand_dims(image, axis=0)

probability = float(
    model.predict(image, verbose=0)[0][0]
)

threshold = 0.70
prediction = (
    "AI-generated"
    if probability >= threshold
    else "Real"
)

print("\n========== PREDICTION ==========")
print("Image:", path)
print("AI probability:", round(probability, 4))
print("Threshold:", threshold)
print("Prediction:", prediction)
