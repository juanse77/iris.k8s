"""Module for testing"""
from fastapi.testclient import TestClient
import joblib
from iris.iris_pred import app

client = TestClient(app)


def test_read_main():
    """Test root endpoint"""

    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Prediction Endpoint": "/predict"}


def test_pred_setosa():
    """Test setosa prediction"""

    val = {
        "sepal_length": "5.1",
        "sepal_width": "3.5",
        "petal_length": "1.4",
        "petal_width": "0.2",
    }

    response = client.post("/predict", json=val)
    assert response.status_code == 200
    assert response.json() == {"Iris type": "Setosa"}


def test_pred_versicolour():
    """Test versicolour prediction"""

    val = {
        "sepal_length": "7.0",
        "sepal_width": "3.2",
        "petal_length": "4.7",
        "petal_width": "1.4",
    }

    response = client.post("/predict", json=val)
    assert response.status_code == 200
    assert response.json() == {"Iris type": "Versicolour"}


def test_pred_virginica():
    """Test virginica prediction"""

    val = {
        "sepal_length": "6.3",
        "sepal_width": "3.3",
        "petal_length": "6.0",
        "petal_width": "2.5",
    }

    response = client.post("/predict", json=val)
    assert response.status_code == 200
    assert response.json() == {"Iris type": "Virginica"}
