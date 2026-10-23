import unittest

from fastapi.testclient import TestClient
from app import app


class TestPredictionApplication(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")

    def test_high_performance_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "attendance": 90,
                "internal_marks": 90,
                "assignment_marks": 90,
                "previous_score": 90
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction", response.json())
        self.assertIn(
            response.json()["prediction"],
            ["PASS", "FAIL"]
        )

    def test_low_performance_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "attendance": 20,
                "internal_marks": 20,
                "assignment_marks": 20,
                "previous_score": 20
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction", response.json())
        self.assertIn(
            response.json()["prediction"],
            ["PASS", "FAIL"]
        )

    def test_missing_field_validation(self):
        response = self.client.post(
            "/predict",
            json={
                "attendance": 85,
                "internal_marks": 80
            }
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
