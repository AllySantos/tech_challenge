"""Executa o pipeline inteiro sobre um corpus mínimo, sem tocar a rede.

Cobre a orquestração de ponta a ponta — ingestão, split, treino, exportação
das três variantes, avaliação, benchmark, portão de qualidade e promoção — que
é justamente o caminho que a DAG do Airflow percorre em produção.
"""

from __future__ import annotations

import json

import pytest

from src.models.registry import (
    ONNX_ARTIFACT,
    ONNX_INT8_ARTIFACT,
    ONNX_PRUNED_ARTIFACT,
    PIPELINE_ARTIFACT,
    PRUNED_PIPELINE_ARTIFACT,
    resolve_current,
)
from src.pipeline import QualityGateError, run_full_pipeline

ABSTRACTS = {
    1: "colon adenocarcinoma tumor resection lymph node staging adjuvant chemotherapy protocol",
    2: "chronic gastritis helicobacter pylori endoscopic biopsy mild inflammatory infiltrate mucosa",
    3: "cerebral infarction middle artery occlusion sudden hemiparesis aphasia thrombolysis window",
    4: "myocardial infarction segment elevation persistent chest pain troponin coronary angioplasty",
    5: "routine follow up stable metabolic parameters laboratory screening without acute findings",
}


def _corpus_csv(rows: int = 40) -> str:
    lines = ["condition_label,medical_abstract"]
    for i in range(rows):
        for label, text in ABSTRACTS.items():
            lines.append(f'{label},"{text} observation number {i}"')
    return "\n".join(lines) + "\n"


@pytest.fixture()
def isolated_project(tmp_path, monkeypatch):
    """Aponta todos os diretórios do projeto para um sandbox temporário."""
    from src.configs import settings as settings_module

    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("medical_tc_train.csv", "medical_tc_test.csv"):
        (raw / name).write_text(_corpus_csv(), encoding="utf-8")

    settings = settings_module.settings
    monkeypatch.setattr(settings, "data_raw_dir", str(raw))
    monkeypatch.setattr(settings, "data_processed_dir", str(tmp_path / "processed"))
    monkeypatch.setattr(settings, "models_dir", str(tmp_path / "models"))
    monkeypatch.setattr(settings, "metrics_dir", str(tmp_path / "metrics"))
    monkeypatch.setattr(settings, "reports_dir", str(tmp_path / "reports"))
    monkeypatch.setattr(settings, "tfidf_min_df", 1)
    monkeypatch.setattr(settings, "prune_keep_features", 30)
    monkeypatch.setattr(settings, "benchmark_runs", 20)
    monkeypatch.setattr(settings, "benchmark_warmup", 5)
    return tmp_path


def test_full_pipeline_promotes_a_complete_version(isolated_project):
    version = run_full_pipeline()

    version_dir = isolated_project / "models" / version
    for artifact in (
        PIPELINE_ARTIFACT,
        PRUNED_PIPELINE_ARTIFACT,
        ONNX_ARTIFACT,
        ONNX_INT8_ARTIFACT,
        ONNX_PRUNED_ARTIFACT,
    ):
        assert (version_dir / artifact).exists(), f"artefato ausente: {artifact}"

    assert resolve_current(isolated_project / "models") == version_dir


def test_full_pipeline_writes_metadata_describing_the_promoted_model(isolated_project):
    version = run_full_pipeline()

    metadata = json.loads(
        (isolated_project / "models" / version / "metadata.json").read_text(encoding="utf-8")
    )

    assert metadata["status"] == "promoted"
    assert metadata["version"] == version
    assert sorted(metadata["labels"]) == ["atencao", "normal", "urgente"]
    assert metadata["quality_gate"]["macro_f1"]["passed"]
    assert metadata["quality_gate"]["p95_latency_ms"]["passed"]
    assert metadata["metrics"]["served_artifact"] == PRUNED_PIPELINE_ARTIFACT


def test_full_pipeline_benchmarks_every_available_backend(isolated_project):
    run_full_pipeline()

    report = json.loads(
        (isolated_project / "reports" / "latency_benchmark.json").read_text(encoding="utf-8")
    )

    assert set(report["backends"]) == {"sklearn", "onnx", "onnx-int8", "onnx-pruned"}
    for measurement in report["backends"].values():
        assert measurement["p50_ms"] <= measurement["p95_ms"] <= measurement["p99_ms"]
        assert measurement["artifact_mb"] > 0
    assert report["speedup_vs_sklearn"]["sklearn"] == 1.0


def test_full_pipeline_refuses_to_promote_over_the_latency_budget(isolated_project, monkeypatch):
    from src.configs import settings as settings_module

    monkeypatch.setattr(settings_module.settings, "max_p95_latency_ms", 0.0001)

    with pytest.raises(QualityGateError, match="p95_latency_ms"):
        run_full_pipeline()

    # Nada foi promovido: o ponteiro de serving não chegou a ser escrito.
    assert not (isolated_project / "models" / "current.json").exists()
