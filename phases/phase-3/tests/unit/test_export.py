import pandas as pd
import pytest

from src.models.export import export_all, export_onnx, quantize_int8, size_mb
from src.models.registry import (
    ONNX_ARTIFACT,
    ONNX_INT8_ARTIFACT,
    ONNX_PRUNED_ARTIFACT,
    PRUNED_PIPELINE_ARTIFACT,
)
from src.models.train import build_pipeline

URGENT = "acute myocardial infarction chest pain troponin elevation coronary occlusion"
ATTENTION = "colon adenocarcinoma tumor resection lymph node staging chemotherapy"
NORMAL = "routine screening stable metabolic parameters without acute inflammatory findings"


@pytest.fixture()
def fitted():
    rows = []
    for i in range(15):
        rows.append({"text": f"{URGENT} case {i}", "urgency": "urgente"})
        rows.append({"text": f"{ATTENTION} case {i}", "urgency": "atencao"})
        rows.append({"text": f"{NORMAL} case {i}", "urgency": "normal"})
    df = pd.DataFrame(rows)

    pipeline = build_pipeline(min_df=1)
    pipeline.fit(df["text"], df["urgency"])
    return pipeline, df


def test_export_onnx_writes_a_loadable_graph(fitted, tmp_path):
    import onnx

    pipeline, _ = fitted

    path = export_onnx(pipeline, tmp_path)

    assert path.name == ONNX_ARTIFACT
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    assert "TfIdfVectorizer" in {node.op_type for node in graph.graph.node}


def test_export_onnx_honours_a_custom_filename(fitted, tmp_path):
    pipeline, _ = fitted

    path = export_onnx(pipeline, tmp_path, "custom.onnx")

    assert path.name == "custom.onnx"


def test_quantize_int8_produces_a_runnable_graph(fitted, tmp_path):
    import onnxruntime as ort

    pipeline, _ = fitted
    quantized = quantize_int8(export_onnx(pipeline, tmp_path))

    assert quantized.name == ONNX_INT8_ARTIFACT
    assert ort.InferenceSession(str(quantized), providers=["CPUExecutionProvider"])


def test_export_all_skips_pruning_without_a_training_set(fitted, tmp_path):
    pipeline, _ = fitted

    artifacts = export_all(pipeline, tmp_path)

    assert set(artifacts) == {"onnx", "onnx-int8"}
    assert not (tmp_path / ONNX_PRUNED_ARTIFACT).exists()


def test_export_all_produces_every_variant_when_given_the_training_set(
    fitted, tmp_path, monkeypatch
):
    from src.configs import settings as settings_module

    monkeypatch.setattr(settings_module.settings, "prune_keep_features", 20)
    pipeline, df = fitted

    artifacts = export_all(pipeline, tmp_path, texts=df["text"], labels=df["urgency"])

    assert set(artifacts) == {"onnx", "onnx-int8", "onnx-pruned"}
    assert (tmp_path / PRUNED_PIPELINE_ARTIFACT).exists()
    assert size_mb(artifacts["onnx-pruned"]) < size_mb(artifacts["onnx"])
