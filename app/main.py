
import os
from io import BytesIO

from flask import (
    Flask,
    request,
    jsonify,
    render_template,
)
from werkzeug.exceptions import RequestEntityTooLarge
from PIL import Image
from dotenv import load_dotenv

from app.inference import (
    VisionGuardPredictor,
    InvalidImageError,
)

load_dotenv()

app = Flask(__name__)

MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))

app.config["MAX_CONTENT_LENGTH"] = (
    MAX_MB * 1024 * 1024
)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# Load model once at startup.
predictor = VisionGuardPredictor()


def validate_upload(file):
    if not file or not file.filename:
        raise ValueError("Please select an image.")

    extension = file.filename.rsplit(
        ".", 1
    )[-1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            "Only JPG, JPEG and PNG files are allowed."
        )

    image_bytes = file.read()

    if not image_bytes:
        raise ValueError("The uploaded file is empty.")

    if len(image_bytes) > MAX_MB * 1024 * 1024:
        raise ValueError("Image exceeds the upload limit.")

    # Decode and validate using our inference module.
    predictor.preprocess(image_bytes)

    return image_bytes


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        uploaded_file = request.files.get("image")

        image_bytes = validate_upload(uploaded_file)

        result = predictor.predict(image_bytes)

        return jsonify({
            "success": True,
            **result
        })

    except (ValueError, InvalidImageError) as error:
        return jsonify({
            "success": False,
            "error": str(error)
        }), 400


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": predictor.model is not None,
        "service": "VisionGuard"
    })


@app.errorhandler(RequestEntityTooLarge)
def file_too_large(error):
    return jsonify({
        "success": False,
        "error": f"File too large. Maximum size is {MAX_MB} MB."
    }), 413


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "An internal server error occurred."
    }), 500

@app.route("/history", methods=["GET"])
def history():
    return render_template("history.html")

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
