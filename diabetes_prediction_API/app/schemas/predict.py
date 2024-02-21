from typing import Any, List, Optional

from pydantic import BaseModel
from diabetes_model.processing.validation import DataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    # predictions: Optional[List[int]]
    predictions: Optional[str]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "Pregnancies": 2,
                        "Glucose": 115.863387,
                        "BloodPressure": 56.410731,
                        "SkinThickness": 24.336736,
                        "Insulin": 94.385783,
                        "BMI": 26.455940,
                        "DiabetesPedigreeFunction": 0.272682,
                        "Age": 20.100494,
                    }
                ]
            }
        }
