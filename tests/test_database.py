
import unittest
from pathlib import Path

import csv

from app.main import app
from app.database import db, Prediction


class TestPredictionDatabase(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def get_count(self):
        with app.app_context():
            return db.session.query(Prediction).count()

    def test_database_table_exists(self):
        from sqlalchemy import inspect

        with app.app_context():
            tables = inspect(db.engine).get_table_names()
            self.assertIn("predictions", tables)

    def test_valid_prediction_saved(self):
        before = self.get_count()

        with open("data/splits/val.csv", newline="", encoding="utf-8") as file:
             sample = next(csv.DictReader(file))

        image_path = Path(sample["path"])

        with image_path.open("rb") as image:
            response = self.client.post(
                "/predict",
                data={"image": (image, image_path.name)},
                content_type="multipart/form-data"
            )

        self.assertEqual(response.status_code, 200)

        result = response.get_json()
        self.assertTrue(result["success"])
        self.assertIn("prediction_id", result)

        self.assertEqual(self.get_count(), before + 1)

        with app.app_context():
            record = db.session.get(
                Prediction, result["prediction_id"]
            )

            self.assertIsNotNone(record)
            self.assertEqual(
                record.prediction, result["prediction"]
            )
            self.assertEqual(
                record.ai_probability,
                result["ai_probability"]
            )

    def test_invalid_upload_not_saved(self):
        before = self.get_count()

        response = self.client.post("/predict")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.get_count(), before)

    def test_history_displays_records(self):
        response = self.client.get("/history")

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            b"Prediction History",
            response.data
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
