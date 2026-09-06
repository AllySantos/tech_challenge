"""Testes ponta a ponta da API contra um modelo treinado de verdade.

O pipeline completo (treino → ONNX → INT8 → promoção) roda uma vez por sessão
sobre um corpus sintético mínimo, de modo que os testes exercitem o mesmo
caminho de código do serving em produção, e não um dublê.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.models.export import export_all
from src.models.registry import promote, write_metadata
from src.models.train import train

URGENT = "acute myocardial infarction chest pain troponin elevation coronary artery occlusion"
ATTENTION = "colon adenocarcinoma tumor resection lymph node staging adjuvant chemotherapy"
NORMAL = "routine screening stable metabolic parameters without acute inflammatory findings"


@pytest.fixture(scope="session")
def model_root(tmp_path_factory):
    """Treina e promove uma versão completa em um registry temporário."""
    workdir = tmp_path_factory.mktemp("registry")
    root = workdir / "models"
    version_dir = root / "20260101T000000Z"

    rows = []
    for i in range(15):
        rows.append({"text": f"{URGENT} case {i}", "urgency": "urgente"})
        rows.append({"text": f"{ATTENTION} case {i}", "urgency": "atencao"})
        rows.append({"text": f"{NORMAL} case {i}", "urgency": "normal"})

    train_path = workdir / "train.csv"
    pd.DataFrame(rows).to_csv(train_path, index=False)

    corpus = pd.DataFrame(rows)
    pipeline, _ = train(train_path=train_path, output_dir=version_dir)
    export_all(pipeline, version_dir, texts=corpus["text"], labels=corpus["urgency"])

    write_metadata(
        version_dir,
        {
            "version": version_dir.name,
            "labels": [str(label) for label in pipeline.named_steps["classifier"].classes_],
        },
    )
    promote(version_dir, root)
    return root


@pytest.fixture(params=["sklearn", "onnx", "onnx-int8", "onnx-pruned"])
def client(request, monkeypatch, model_root):
    """Sobe a API uma vez por backend, para cobrir os três caminhos."""
    from src.app import api
    from src.configs import settings as settings_module

    monkeypatch.setattr(settings_module.settings, "models_dir", str(model_root))
    monkeypatch.setattr(settings_module.settings, "inference_backend", request.param)
    monkeypatch.setattr(api, "predictor", None)

    with TestClient(api.app) as test_client:
        test_client.backend = request.param
        yield test_client


def test_health_reports_the_loaded_model(client):
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["backend"] == client.backend
    assert sorted(payload["labels"]) == ["atencao", "normal", "urgente"]


def test_predict_classifies_an_urgent_report(client):
    response = client.post("/predict", json={"text": URGENT})

    assert response.status_code == 200
    payload = response.json()
    assert payload["urgency"] == "urgente"
    assert 0.0 < payload["confidence"] <= 1.0
    assert payload["backend"] == client.backend
    assert payload["inference_ms"] >= 0


def test_predict_probabilities_cover_every_class_and_sum_to_one(client):
    payload = client.post("/predict", json={"text": NORMAL}).json()

    assert set(payload["probabilities"]) == {"normal", "atencao", "urgente"}
    assert sum(payload["probabilities"].values()) == pytest.approx(1.0, abs=0.01)


def test_predict_batch_returns_one_result_per_item(client):
    response = client.post("/predict/batch", json={"items": [{"text": URGENT}, {"text": NORMAL}]})

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    assert [r["urgency"] for r in payload["results"]] == ["urgente", "normal"]


def test_predict_rejects_blank_text(client):
    assert client.post("/predict", json={"text": "   "}).status_code == 422


def test_predict_rejects_missing_field(client):
    assert client.post("/predict", json={}).status_code == 422


def test_metrics_endpoint_exposes_the_instrumented_series(client):
    client.post("/predict", json={"text": URGENT})

    body = client.get("/metrics").text

    assert "triage_requests_total" in body
    assert "triage_request_duration_seconds_bucket" in body
    assert "triage_inference_duration_seconds_bucket" in body
    assert "triage_predictions_total" in body
    assert "triage_model_loaded" in body


def test_exported_graphs_match_the_source_pipeline(model_root):
    """ONNX e INT8 servem o mesmo modelo do scikit-learn: divergir aqui é bug de exportação."""
    from src.inference.predictor import load_predictor

    version_dir = model_root / "20260101T000000Z"
    reports = [URGENT, ATTENTION, NORMAL]
    predictions = {
        backend: [p.urgency for p in load_predictor(backend, version_dir).predict(reports)]
        for backend in ("sklearn", "onnx", "onnx-int8")
    }

    assert predictions["onnx"] == predictions["sklearn"]
    assert predictions["onnx-int8"] == predictions["sklearn"]


def test_pruned_model_still_separates_the_three_urgencies(model_root):
    """O pruning descarta features; o que ele não pode fazer é quebrar a triagem."""
    from src.inference.predictor import load_predictor

    predictor = load_predictor("onnx-pruned", model_root / "20260101T000000Z")
    predictions = predictor.predict([URGENT, ATTENTION, NORMAL])

    assert [p.urgency for p in predictions] == ["urgente", "atencao", "normal"]
