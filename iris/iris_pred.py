import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
iris_classes = ['Setosa', 'Versicolour', 'Virginica']

model = joblib.load('model/iris_classifier.joblib')

class IrisModelInput(BaseModel):
    """Input data type for prediction"""

    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def read_root() -> dict:
    return {"Prediction Endpoint": "/predict"}

@app.post("/predict")
async def predict(input: IrisModelInput) -> dict:
    """Predict the type of iris flower based on its petal and sepal measurements
    
    :param input: object with the measurements of sepal and petal
    :type input: IrisModelInput
    :returns: iris type prediction
    :rtype: dict
    """

    data = np.array([[input.sepal_length, input.sepal_width, input.petal_length, input.petal_width]])
    val = model.predict(data)[0]

    return {"Iris type": iris_classes[val]}
    