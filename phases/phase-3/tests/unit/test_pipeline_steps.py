import pytest

from src.pipeline import QualityGateError, run_quality_gate


def _benchmark(p95_ms, backend=None):
    from src.configs.settings import settings

    return {"backends": {backend or settings.inference_backend: {"p95_ms": p95_ms}}}


def test_quality_gate_passes_when_both_criteria_are_met(monkeypatch):
    from src.configs import settings as settings_module

    monkeypatch.setattr(settings_module.settings, "min_macro_f1", 0.6)
    monkeypatch.setattr(settings_module.settings, "max_p95_latency_ms", 25.0)

    checks = run_quality_gate("models/v1", {"macro_f1": 0.75}, _benchmark(4.0))

    assert checks["macro_f1"]["passed"]
    assert checks["p95_latency_ms"]["passed"]


def test_quality_gate_blocks_promotion_on_low_f1(monkeypatch):
    from src.configs import settings as settings_module

    monkeypatch.setattr(settings_module.settings, "min_macro_f1", 0.8)

    with pytest.raises(QualityGateError, match="macro_f1"):
        run_quality_gate("models/v1", {"macro_f1": 0.5}, _benchmark(4.0))


def test_quality_gate_blocks_promotion_on_slow_p95(monkeypatch):
    from src.configs import settings as settings_module

    monkeypatch.setattr(settings_module.settings, "min_macro_f1", 0.1)
    monkeypatch.setattr(settings_module.settings, "max_p95_latency_ms", 5.0)

    with pytest.raises(QualityGateError, match="p95_latency_ms"):
        run_quality_gate("models/v1", {"macro_f1": 0.9}, _benchmark(50.0))


def test_quality_gate_blocks_when_serving_backend_was_not_measured(monkeypatch):
    from src.configs import settings as settings_module

    monkeypatch.setattr(settings_module.settings, "min_macro_f1", 0.1)
    monkeypatch.setattr(settings_module.settings, "inference_backend", "onnx-pruned")

    with pytest.raises(QualityGateError, match="p95_latency_ms"):
        run_quality_gate("models/v1", {"macro_f1": 0.9}, _benchmark(1.0, backend="sklearn"))
