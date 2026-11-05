
import os
from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "models/mobilenetv2_tuned.keras"
)

THRESHOLD = float(
    os.getenv("PREDICTION_THRESHOLD", "0.70")
)

IMAGE_SIZE = (224, 224)

# Reject excessively large decoded images.
Image.MAX_IMAGE_PIXELS = 25_000_000


class InvalidImageError(ValueError):
    pass


class VisionGuardPredictor:

    def __init__(self):
        print("Loading VisionGuard model...")

        self.model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        print("Model loaded successfully!")

    def preprocess(self, image_bytes):
        try:
            with Image.open(BytesIO(image_bytes)) as img:
                # Confirm that the file is a supported image.
                if img.format not in {"JPEG", "PNG"}:
                    raise InvalidImageError(
                        "Only JPEG and PNG images are supported."
                    )

                img.load()
                img = img.convert("RGB")

                image = np.asarray(
                    img,
                    dtype=np.float32
                )

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            Image.DecompressionBombError
        ) as exc:
            raise InvalidImageError(
                "The uploaded file is not a valid image."
            ) from exc

        # Same resizing and padding as training.
        image = tf.image.resize_with_pad(
            image,
            IMAGE_SIZE[0],
            IMAGE_SIZE[1]
        )

        # Match the training pipeline: [0, 1].
        image = image / 255.0

        # Add batch dimension.
        image = tf.expand_dims(image, axis=0)

        return image

    def predict(self, image_bytes):
        image = self.preprocess(image_bytes)

        probability = float(
            self.model.predict(
                image,
                verbose=0
            )[0][0]
        )

        predicted_ai = probability >= THRESHOLD

        label = (
            "AI-generated"
            if predicted_ai
            else "Real"
        )

        # Score corresponding to the displayed class.
        class_score = (
            probability
            if predicted_ai
            else 1.0 - probability
        )

        return {
            "prediction": label,
            "ai_probability": round(probability, 4),
            "class_score": round(class_score, 4),
            "threshold": THRESHOLD,
            "model": "MobileNetV2 Tuned",
        }
