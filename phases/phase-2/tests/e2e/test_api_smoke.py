import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
)

import pytest
from fastapi.testclient import TestClient

from src.app import api
from src.models.train import MLPRecommender


@pytest.fixture()
def client(monkeypatch, tmp_path):
    model_path = tmp_path / "recommender.pt"
    torch = pytest.importorskip("torch")

    model = MLPRecommender(n_users=2, n_items=3, embedding_dim=32)
    torch.save(model.state_dict(), model_path)

    monkeypatch.setattr(api, "model", None)
    monkeypatch.setattr(api, "n_users", 0)
    monkeypatch.setattr(api, "n_items", 0)
    monkeypatch.setattr(api.settings, "models_path", str(tmp_path))

    with TestClient(api.app) as test_client:
        yield test_client


def test_health_endpoint_returns_ok(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_recommend_endpoint_returns_predictions(client):
    response = client.get("/recommend", params={"user_id": 0, "top_k": 3})

    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == 0
    assert payload["top_k"] == 3
    assert len(payload["recommendations"]) == 3


def test_recommend_endpoint_returns_404_for_unknown_user(client):
    response = client.get("/recommend", params={"user_id": 10, "top_k": 3})

    assert response.status_code == 404
    assert "User 10" in response.json()["detail"]
