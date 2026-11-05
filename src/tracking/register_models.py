
import os

os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

import mlflow
import mlflow.tensorflow
import tensorflow as tf
from mlflow.tracking import MlflowClient

TRACKING_URI = (
    "https://dagshub.com/"
    "nihaal237/23I-0831_A2_Ops.mlflow"
)

MODEL_NAME = "VisionGuard_Image_Classifier"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)
mlflow.set_experiment("VisionGuard_Model_Registry")

client = MlflowClient()


def register_checkpoint(path, run_name):
    print(f"\nRegistering {path}...")

    model = tf.keras.models.load_model(
        path, compile=False
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag(
            "source_checkpoint", path
        )

        # Log as a proper MLflow model and register it.
        result = mlflow.tensorflow.log_model(
            model,
            name="model",
            registered_model_name=MODEL_NAME,
            await_registration_for=120
        )

        print("Logged model:", result.model_uri)

    del model
    tf.keras.backend.clear_session()

    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    matching = [
        version for version in versions
        if version.run_id == result.run_id
    ]

    if not matching:
        raise RuntimeError(
            "Model logged, but its registered version "
            "could not be identified."
        )

    return max(matching, key=lambda v: int(v.version))


# Version 1: Frozen MobileNetV2
v1 = register_checkpoint(
    "models/mobilenetv2_head.keras",
    "Register_MobileNetV2_Frozen"
)

print("Registered frozen version:", v1.version)

# Version 2: Tuned MobileNetV2
v2 = register_checkpoint(
    "models/mobilenetv2_tuned.keras",
    "Register_MobileNetV2_Tuned"
)

print("Registered tuned version:", v2.version)

# Assign champion to the tuned model.
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=v2.version
)

print("\n========== REGISTRY COMPLETE ==========")
print("Registered model:", MODEL_NAME)
print("Frozen version:", v1.version)
print("Tuned version:", v2.version)
print("Champion alias:", v2.version)
