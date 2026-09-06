"""A API precisa subir mesmo sem modelo, para que o operador enxergue a falha."""

from fastapi.testclient import TestClient


def test_api_starts_degraded_when_no_model_is_available(monkeypatch, tmp_path):
    from src.app import api
    from src.configs import settings as settings_module

    monkeypatch.setattr(settings_module.settings, "models_dir", str(tmp_path / "empty"))
    monkeypatch.setattr(api, "predictor", None)

    with TestClient(api.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {
            "status": "degraded",
            "model_loaded": False,
            "model_version": None,
            "backend": None,
            "labels": [],
        }

        assert client.post("/predict", json={"text": "any report"}).status_code == 503
        assert "triage_model_loaded" in client.get("/metrics").text
