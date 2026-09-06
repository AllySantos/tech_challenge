import pandas as pd
import pytest

from src.configs.settings import URGENCY_LABELS
from src.evaluation.evaluate import evaluate_pipeline, save_metrics
from src.models.train import build_pipeline, train

URGENT = "acute myocardial infarction chest pain troponin elevation coronary occlusion"
ATTENTION = "colon adenocarcinoma tumor resection lymph node staging chemotherapy"
NORMAL = "routine screening stable parameters no acute inflammatory findings reported"


def _corpus(repeats: int = 12) -> pd.DataFrame:
    rows = []
    for i in range(repeats):
        rows.append({"text": f"{URGENT} case {i}", "urgency": "urgente"})
        rows.append({"text": f"{ATTENTION} case {i}", "urgency": "atencao"})
        rows.append({"text": f"{NORMAL} case {i}", "urgency": "normal"})
    return pd.DataFrame(rows)


@pytest.fixture()
def corpus_paths(tmp_path):
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"
    _corpus().to_csv(train_path, index=False)
    _corpus(repeats=4).to_csv(validation_path, index=False)
    return train_path, validation_path


def test_build_pipeline_exposes_expected_steps():
    pipeline = build_pipeline()

    assert list(pipeline.named_steps) == ["tfidf", "classifier"]
    assert pipeline.named_steps["tfidf"].ngram_range == (1, 2)


def test_train_persists_artifact_and_learns_known_labels(tmp_path, corpus_paths):
    train_path, _ = corpus_paths
    output_dir = tmp_path / "v1"

    pipeline, artifact_path = train(
        train_path=train_path,
        output_dir=output_dir,
    )

    assert artifact_path.exists()
    assert set(pipeline.named_steps["classifier"].classes_) <= set(URGENCY_LABELS)


def test_train_rejects_unknown_urgency_labels(tmp_path):
    train_path = tmp_path / "train.csv"
    df = _corpus()
    df.loc[0, "urgency"] = "critico"
    df.to_csv(train_path, index=False)

    with pytest.raises(ValueError, match="desconhecidos"):
        train(train_path=train_path, output_dir=tmp_path / "v1")


def test_evaluate_pipeline_reports_per_class_metrics(tmp_path, corpus_paths):
    train_path, validation_path = corpus_paths
    pipeline, _ = train(train_path=train_path, output_dir=tmp_path / "v1")

    metrics = evaluate_pipeline(pipeline, validation_path=validation_path)

    assert metrics["n_samples"] == 12
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert set(metrics["per_class"]) == set(metrics["labels"])
    assert len(metrics["confusion_matrix"]) == len(metrics["labels"])


def test_save_metrics_writes_readable_json(tmp_path):
    path = save_metrics({"macro_f1": 0.9, "nota": "atenção"}, tmp_path / "m.json")

    import json

    assert json.loads(path.read_text(encoding="utf-8"))["nota"] == "atenção"
