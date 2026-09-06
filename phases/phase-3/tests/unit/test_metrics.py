from prometheus_client import REGISTRY

from src.app.metrics import LATENCY_BUCKETS, record_prediction


def _sample(name: str, labels: dict) -> float:
    value = REGISTRY.get_sample_value(name, labels)
    return 0.0 if value is None else value


def test_record_prediction_increments_the_counter_and_the_histogram():
    labels = {"urgency": "urgente"}
    before = _sample("triage_predictions_total", labels)
    before_count = _sample("triage_prediction_confidence_count", labels)

    record_prediction("urgente", 0.93)

    assert _sample("triage_predictions_total", labels) == before + 1
    assert _sample("triage_prediction_confidence_count", labels) == before_count + 1


def test_latency_buckets_are_ascending_and_cover_the_sub_millisecond_range():
    assert list(LATENCY_BUCKETS) == sorted(LATENCY_BUCKETS)
    assert LATENCY_BUCKETS[0] <= 0.001
    assert LATENCY_BUCKETS[-1] >= 1.0
