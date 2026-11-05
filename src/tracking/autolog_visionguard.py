
import os

os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

import subprocess

import mlflow
import mlflow.tensorflow
import tensorflow as tf

from src.data.pipeline import create_dataset
from src.models.train_cnn import build_cnn

SEED = 42
EPOCHS = 2

tf.keras.utils.set_random_seed(SEED)

TRACKING_URI = (
    "https://dagshub.com/"
    "nihaal237/23I-0831_A2_Ops.mlflow"
)

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("VisionGuard_Model_Training")

# Automatically capture training metrics,
# Keras parameters and training metadata.
# Model artifact logging will be handled
# separately in Task 3.4.
mlflow.tensorflow.autolog(log_models=False)

train_ds = create_dataset("train", training=True)
val_ds = create_dataset("val", training=False)

model = build_cnn()

commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    text=True
).strip()

with mlflow.start_run(
    run_name="CNN_Autolog_Demo"
):
    # Additional metadata not automatically captured.
    mlflow.set_tags({
        "logging_method": "tensorflow_autolog",
        "dataset": "VisionGuard",
        "git_commit": commit,
        "seed": str(SEED),
        "purpose": "autologging_demonstration",
    })

    mlflow.log_param("random_seed", SEED)

    print("\n========== AUTOLOG TRAINING ==========")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=2
    )

    print("\nFinal validation accuracy:",
          history.history["val_accuracy"][-1])

print("\nMLflow autologging completed successfully!")
