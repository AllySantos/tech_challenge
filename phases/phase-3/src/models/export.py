"""Exportação do pipeline treinado para ONNX e para ONNX quantizado em INT8.

Três técnicas são aplicadas e medidas:

1. **Conversão para ONNX Runtime** — troca o caminho de inferência do
   scikit-learn (Python puro sobre scipy sparse) por um grafo compilado
   executado em C++, eliminando o overhead do interpretador por requisição.
2. **Pruning do vocabulário** — descarta os termos de menor peso e reexporta,
   atacando o ``TfIdfVectorizer``, que é o nó dominante do grafo.
3. **Quantização dinâmica INT8** — mantida no comparativo, embora não surta
   efeito neste grafo: a saída do classificador é um ``LinearClassifier`` do
   domínio ``ai.onnx.ml``, cujos coeficientes vivem em atributos do nó e não
   em initializers. Sem nenhum ``MatMul`` ou ``Gemm`` para converter, o
   quantizador devolve o grafo intacto. O resultado é reportado assim mesmo,
   em vez de omitido — ver docs/optimization.md.
"""

from __future__ import annotations

import logging
from pathlib import Path

from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType
from sklearn.pipeline import Pipeline

from src.configs.settings import settings
from src.models.registry import (
    ONNX_ARTIFACT,
    ONNX_INT8_ARTIFACT,
    ONNX_PRUNED_ARTIFACT,
    PRUNED_PIPELINE_ARTIFACT,
)

logger = logging.getLogger(__name__)

# opset 15 cobre TfIdfVectorizer e as operações do classificador linear, e é
# suportado por todas as versões do ONNX Runtime em uso.
TARGET_OPSET = 15


def export_onnx(pipeline: Pipeline, output_dir: Path, filename: str = ONNX_ARTIFACT) -> Path:
    """Converte o pipeline scikit-learn para ONNX."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    onnx_model = to_onnx(
        pipeline,
        initial_types=[("input", StringTensorType([None, 1]))],
        target_opset=TARGET_OPSET,
        options={id(pipeline): {"zipmap": False}},
    )
    output_path.write_bytes(onnx_model.SerializeToString())

    logger.info("ONNX exportado: %s (%.2f MB)", output_path, size_mb(output_path))
    return output_path


def quantize_int8(onnx_path: Path, output_path: Path | None = None) -> Path:
    """Aplica quantização dinâmica INT8 sobre o grafo ONNX."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    output_path = output_path or onnx_path.parent / ONNX_INT8_ARTIFACT

    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(output_path),
        weight_type=QuantType.QUInt8,
        extra_options={"MatMulConstBOnly": True},
    )

    logger.info(
        "ONNX INT8 gerado: %s (%.2f MB, %.0f%% do tamanho original)",
        output_path,
        size_mb(output_path),
        100 * size_mb(output_path) / size_mb(onnx_path),
    )
    return output_path


def export_all(pipeline: Pipeline, output_dir: Path, texts=None, labels=None) -> dict[str, Path]:
    """Gera todas as variantes otimizadas e devolve os caminhos por backend.

    O pruning e a quantização são best-effort: se qualquer um falhar, o erro é
    registrado e o serving continua disponível pelas variantes que deram
    certo. ``texts`` e ``labels`` são o conjunto de treino, necessário para
    reajustar o classificador depois do pruning.
    """
    import joblib

    artifacts = {"onnx": export_onnx(pipeline, output_dir)}

    try:
        artifacts["onnx-int8"] = quantize_int8(artifacts["onnx"])
    except Exception:  # noqa: BLE001 - quantização é opcional por design
        logger.exception("Quantização INT8 falhou; seguindo sem essa variante")

    if texts is None or labels is None:
        logger.info("Conjunto de treino não fornecido; pruning ignorado")
        return artifacts

    try:
        from src.models.prune import prune_pipeline

        pruned = prune_pipeline(pipeline, texts, labels, settings.prune_keep_features)
        joblib.dump(pruned, output_dir / PRUNED_PIPELINE_ARTIFACT, compress=3)
        artifacts["onnx-pruned"] = export_onnx(pruned, output_dir, ONNX_PRUNED_ARTIFACT)
    except Exception:  # noqa: BLE001 - pruning é opcional por design
        logger.exception("Pruning do vocabulário falhou; seguindo sem essa variante")

    return artifacts


def size_mb(path: Path) -> float:
    """Tamanho do arquivo em megabytes."""
    return path.stat().st_size / (1024 * 1024)
