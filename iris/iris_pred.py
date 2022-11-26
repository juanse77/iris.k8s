"""Implements an API for iris recognition"""
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
iris_classes = ["Setosa", "Versicolour", "Virginica"]

model = joblib.load("iris_classifier.joblib")


class IrisModelInput(BaseModel):
    """Input data type for prediction"""

    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def read_root() -> dict:
    """Default endpoint"""
    return {"Prediction Endpoint": "/predict"}


@app.post("/predict")
async def predict(obj_input: IrisModelInput) -> dict:
    """Predict the type of iris flower based on its petal and sepal measurements

    :param obj_input: object with the measurements of sepal and petal
    :type obj_input: IrisModelInput
    :returns: iris type prediction
    :rtype: dict
    """

    data = np.array(
        [
            [
                obj_input.sepal_length,
                obj_input.sepal_width,
                obj_input.petal_length,
                obj_input.petal_width,
            ]
        ]
    )
    val = model.predict(data)[0]

    return {"Iris type": iris_classes[val]}
