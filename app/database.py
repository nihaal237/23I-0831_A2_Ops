
from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    filename = db.Column(
        db.String(255),
        nullable=False
    )

    prediction = db.Column(
        db.String(30),
        nullable=False
    )

    ai_probability = db.Column(
        db.Float,
        nullable=False
    )

    threshold = db.Column(
        db.Float,
        nullable=False
    )

    model_name = db.Column(
        db.String(100),
        nullable=False
    )

    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "prediction": self.prediction,
            "ai_probability": self.ai_probability,
            "threshold": self.threshold,
            "model": self.model_name,
            "created_at": self.created_at.isoformat()
        }
