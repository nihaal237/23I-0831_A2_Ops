
import os
from io import BytesIO

from app.database import db, Prediction
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.utils import secure_filename

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

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INSTANCE_DIR = PROJECT_ROOT / "instance"
INSTANCE_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(
    __name__,
    instance_path=str(INSTANCE_DIR)
)

app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
    "DATABASE_URL",
    "sqlite:///visionguard.db"
)

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

with app.app_context():
    db.create_all()

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

        # Run inference.
        result = predictor.predict(image_bytes)

        # Store prediction in the database.
        record = Prediction(
            filename=secure_filename(uploaded_file.filename),
            prediction=result["prediction"],
            ai_probability=result["ai_probability"],
            threshold=result["threshold"],
            model_name=result["model"]
        )

        try:

            db.session.add(record)
            db.session.commit()
        
            from sqlalchemy import text

            print(
    "SERVER DATABASE FILE:",
    db.session.execute(text("PRAGMA database_list")).fetchall(),
    flush=True
)
            print(
    "SERVER COUNT:",
    db.session.query(Prediction).count(),
    flush=True
)
           

        except SQLAlchemyError:
            db.session.rollback()
            app.logger.exception("Failed to save prediction")
            return jsonify({
                "success": False,
                "error": "Could not save prediction."
            }), 500

        return jsonify({
            "success": True,
            "prediction_id": record.id,
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
    records = (
        Prediction.query
        .order_by(Prediction.created_at.desc(), Prediction.id.desc())
        .limit(100)
        .all()
    )

    return render_template(
        "history.html",
        predictions=records,
        total_predictions=Prediction.query.count()
    )

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
