from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Student Result Prediction API",
    description="Predict student results using academic performance data.",
    version="1.0.0"
)

MODEL_PATH = Path("student_result_model.pkl")

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]


class StudentData(BaseModel):
    attendance: float
    internal_marks: float
    assignment_marks: float
    previous_score: float


def load_model():
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="Model file not found."
        )

    return joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return {
        "status": "ok",
        "service": "student-result-prediction"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: StudentData):
    sample = pd.DataFrame([{
        feature: getattr(data, feature)
        for feature in FEATURES
    }])

    model = load_model()
    prediction_code = int(model.predict(sample)[0])
    prediction = "PASS" if prediction_code == 1 else "FAIL"

    return {
        "prediction": prediction,
        "prediction_code": prediction_code
    }
