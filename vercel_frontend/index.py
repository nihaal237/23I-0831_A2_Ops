
import os
import tempfile
from pathlib import Path

from database import save_prediction, get_predictions
from werkzeug.utils import secure_filename
from flask import Flask, jsonify, render_template, request
from gradio_client import Client, handle_file
from werkzeug.exceptions import RequestEntityTooLarge

app = Flask(__name__)

SPACE_ID = os.getenv(
    "HF_SPACE_ID",
    "nihaal237/23I-0831-A2-Ops",
)

# Keep below Vercel's function request payload limit.
MAX_UPLOAD_BYTES = 4 * 1024 * 1024
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({
        "status": "healthy",
        "service": "VisionGuard Vercel Frontend",
        "inference_provider": "Hugging Face",
    })


@app.post("/predict")
def predict():
    uploaded = request.files.get("image")

    if uploaded is None or not uploaded.filename:
        return jsonify({
            "success": False,
            "error": "Please select an image.",
        }), 400

    extension = Path(uploaded.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({
            "success": False,
            "error": "Only JPG, JPEG and PNG are supported.",
        }), 400

    image_bytes = uploaded.read()

    if not image_bytes:
        return jsonify({
            "success": False,
            "error": "The uploaded image is empty.",
        }), 400

    if len(image_bytes) > MAX_UPLOAD_BYTES:
        return jsonify({
            "success": False,
            "error": "Image exceeds the upload limit.",
        }), 413

    image_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            dir="/tmp" if os.path.isdir("/tmp") else None,
            delete=False,
        ) as temp:
            temp.write(image_bytes)
            image_path = temp.name

        client = Client(SPACE_ID)

        label, probability, metadata = client.predict(
            handle_file(image_path),
            api_name="/predict",
        )

        
        result = {
            "prediction": label,
            "ai_probability": float(probability),
            "threshold": float(metadata.get("threshold", 0.70)),
            "model": metadata.get("model", "MobileNetV2 Tuned"),
        }

        # Save the cloud prediction in Neon PostgreSQL.
        prediction_id = save_prediction(
            secure_filename(uploaded.filename),
            result,
        )

        return jsonify({
            "success": True,
            "prediction_id": prediction_id,
            **result,
        })


    except Exception:
        app.logger.exception("Cloud inference failed")
        return jsonify({
            "success": False,
            "error": "Cloud inference is temporarily unavailable.",
        }), 502

    finally:
        if image_path and os.path.exists(image_path):
            os.unlink(image_path)


@app.get("/history")
def history():
    predictions, total = get_predictions()

    return render_template(
        "history.html",
        predictions=predictions,
        total_predictions=total,
    )



@app.errorhandler(RequestEntityTooLarge)
def oversized(error):
    return jsonify({
        "success": False,
        "error": "Image exceeds the upload limit.",
    }), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8001, debug=False)
