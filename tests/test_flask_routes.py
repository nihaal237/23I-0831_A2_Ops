
import unittest
from io import BytesIO

from app.main import app


class TestVisionGuardFlask(unittest.TestCase):

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_homepage(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"VisionGuard", response.data)

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])

    def test_history_page(self):
        response = self.client.get("/history")
        self.assertEqual(response.status_code, 200)

    def test_missing_image(self):
        response = self.client.post("/predict")

        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()["success"])

    def test_invalid_extension(self):
        response = self.client.post(
            "/predict",
            data={
                "image": (
                    BytesIO(b"not an image"),
                    "document.txt"
                )
            },
            content_type="multipart/form-data"
        )

        self.assertEqual(response.status_code, 400)

    def test_fake_jpg(self):
        response = self.client.post(
            "/predict",
            data={
                "image": (
                    BytesIO(b"fake image content"),
                    "fake.jpg"
                )
            },
            content_type="multipart/form-data"
        )

        self.assertEqual(response.status_code, 400)

    def test_empty_image(self):
        response = self.client.post(
            "/predict",
            data={
                "image": (
                    BytesIO(b""),
                    "empty.png"
                )
            },
            content_type="multipart/form-data"
        )

        self.assertEqual(response.status_code, 400)

    def test_oversized_upload(self):
        response = self.client.post(
            "/predict",
            data={
                "image": (
                    BytesIO(b"x" * (10 * 1024 * 1024 + 1)),
                    "large.jpg"
                )
            },
            content_type="multipart/form-data"
        )

        self.assertEqual(response.status_code, 413)


if __name__ == "__main__":
    unittest.main(verbosity=2)
