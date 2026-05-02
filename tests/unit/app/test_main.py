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


def test_lifespan_loads_model(monkeypatch):
    loaded_model = object()
    called = {}

    def fake_load_machine_learning_model(path):
        called["path"] = path
        return loaded_model

    monkeypatch.setattr("app.main.load_machine_learning_model", fake_load_machine_learning_model)

    with TestClient(app) as lifespan_client:
        assert called["path"] == "models/model.pth"
        response = lifespan_client.get("/health")
        assert response.status_code == 200
