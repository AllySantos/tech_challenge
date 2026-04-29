from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_app_initialization():
    assert app.title == "churn-prediction"
    assert app.version == "0.1.0"


def test_health_router_is_included():
    response = client.get("/health")
    assert response.status_code == 200


def test_predict_router_is_included():
    response = client.post("/predict")
    assert response.status_code == 200
