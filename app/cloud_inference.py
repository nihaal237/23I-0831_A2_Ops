
import os

import requests


class CloudInferenceError(Exception):
    """Raised when the remote inference service fails."""
    pass


class CloudVisionGuardPredictor:
    def __init__(self):
        self.api_url = os.getenv(
            "CLOUD_API_URL",
            "https://23-i-0831-a2-ops.vercel.app/predict"
        )

    def predict(self, image_bytes):
        try:
            response = requests.post(
                self.api_url,
                files={
                    "image": (
                        "uploaded_image.jpg",
                        image_bytes,
                        "application/octet-stream"
                    )
                },
                timeout=(10, 90),
            )

            response.raise_for_status()
            result = response.json()

            if not result.get("success"):
                raise CloudInferenceError(
                    result.get("error", "Cloud prediction failed.")
                )

            return {
                "prediction": result["prediction"],
                "ai_probability": float(
                    result["ai_probability"]
                ),
                "threshold": float(result["threshold"]),
                "model": result["model"],
            }

        except requests.exceptions.Timeout as error:
            raise CloudInferenceError(
                "Cloud API request timed out."
            ) from error

        except requests.exceptions.RequestException as error:
            raise CloudInferenceError(
                "Could not connect to the cloud API."
            ) from error

        except (ValueError, KeyError, TypeError) as error:
            raise CloudInferenceError(
                "Invalid response received from the cloud API."
            ) from error
