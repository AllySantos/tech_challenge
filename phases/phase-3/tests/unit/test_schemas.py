import pytest
from pydantic import ValidationError

from src.app.schemas import BatchTriageRequest, HealthResponse, TriageRequest, TriageResponse


def test_request_strips_surrounding_whitespace():
    assert TriageRequest(text="  chest pain  ").text == "chest pain"


@pytest.mark.parametrize("value", ["", "   ", "\n\t "])
def test_request_rejects_blank_text(value):
    with pytest.raises(ValidationError):
        TriageRequest(text=value)


def test_request_rejects_text_above_the_configured_limit(monkeypatch):
    from src.configs import settings as settings_module

    monkeypatch.setattr(settings_module.settings, "max_text_length", 10)

    with pytest.raises(ValidationError, match="excede o limite"):
        TriageRequest(text="a" * 11)


def test_request_accepts_text_exactly_at_the_limit(monkeypatch):
    from src.configs import settings as settings_module

    monkeypatch.setattr(settings_module.settings, "max_text_length", 10)

    assert len(TriageRequest(text="a" * 10).text) == 10


def test_batch_rejects_an_empty_list():
    with pytest.raises(ValidationError):
        BatchTriageRequest(items=[])


def test_batch_rejects_more_than_one_hundred_items():
    with pytest.raises(ValidationError):
        BatchTriageRequest(items=[{"text": "laudo"}] * 101)


def test_batch_accepts_the_maximum_size():
    assert len(BatchTriageRequest(items=[{"text": "laudo"}] * 100).items) == 100


def test_triage_response_serialises_every_field():
    response = TriageResponse(
        urgency="urgente",
        confidence=0.91,
        probabilities={"normal": 0.03, "atencao": 0.06, "urgente": 0.91},
        model_version="20260101T000000Z",
        backend="onnx-pruned",
        inference_ms=0.12,
    )

    assert response.model_dump()["urgency"] == "urgente"


def test_health_response_defaults_to_no_model():
    health = HealthResponse(status="degraded", model_loaded=False)

    assert health.labels == []
    assert health.model_version is None
