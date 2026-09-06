from src.evaluation.benchmark import _percentile, _speedups, measure
from src.inference.predictor import Prediction, Predictor


class StubPredictor(Predictor):
    backend = "stub"

    def __init__(self):
        super().__init__(labels=["normal", "atencao", "urgente"], model_version="v-test")
        self.calls = 0

    def predict(self, texts):
        self.calls += len(texts)
        return [Prediction("normal", 0.9, {"normal": 0.9, "atencao": 0.05, "urgente": 0.05})] * len(
            texts
        )


def test_percentile_picks_expected_positions():
    values = [float(i) for i in range(1, 101)]

    assert _percentile(values, 50) == 50.0
    assert _percentile(values, 95) == 95.0
    assert _percentile(values, 99) == 99.0


def test_percentile_is_clamped_to_the_last_element():
    assert _percentile([1.0, 2.0, 3.0], 100) == 3.0


def test_measure_runs_warmup_plus_measured_iterations():
    predictor = StubPredictor()

    result = measure(predictor, ["laudo a", "laudo b"], runs=20, warmup=5)

    assert predictor.calls == 25
    assert result["runs"] == 20
    assert result["p50_ms"] <= result["p95_ms"] <= result["p99_ms"]
    assert result["throughput_rps"] > 0


def test_speedups_are_relative_to_the_sklearn_baseline():
    results = {
        "sklearn": {"p95_ms": 10.0},
        "onnx": {"p95_ms": 5.0},
        "onnx-int8": {"p95_ms": 2.0},
    }

    assert _speedups(results) == {"sklearn": 1.0, "onnx": 2.0, "onnx-int8": 5.0}


def test_speedups_are_empty_without_a_baseline():
    assert _speedups({"onnx": {"p95_ms": 5.0}}) == {}


def test_predictor_converts_probabilities_into_the_winning_label():
    import numpy as np

    predictor = StubPredictor()
    predictions = predictor._to_predictions(np.array([[0.1, 0.2, 0.7]]))

    assert predictions[0].urgency == "urgente"
    assert predictions[0].confidence == 0.7
    assert predictions[0].probabilities == {"normal": 0.1, "atencao": 0.2, "urgente": 0.7}
