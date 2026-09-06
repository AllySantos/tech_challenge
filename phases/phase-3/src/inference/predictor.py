"""Camada de inferência com backends intercambiáveis.

A API e o benchmark de latência conversam com a mesma interface
(``Predictor``), o que permite trocar scikit-learn por ONNX Runtime — ou
comparar os dois — sem tocar em nenhuma outra parte do código.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.models.registry import (
    ONNX_ARTIFACT,
    ONNX_INT8_ARTIFACT,
    ONNX_PRUNED_ARTIFACT,
    PIPELINE_ARTIFACT,
    read_metadata,
    resolve_current,
)

logger = logging.getLogger(__name__)

# Ordem intencional: é a mesma em que o benchmark reporta os backends, do
# menos para o mais otimizado.
BACKENDS = ("sklearn", "onnx", "onnx-int8", "onnx-pruned")

_ARTIFACT_BY_BACKEND = {
    "sklearn": PIPELINE_ARTIFACT,
    "onnx": ONNX_ARTIFACT,
    "onnx-int8": ONNX_INT8_ARTIFACT,
    "onnx-pruned": ONNX_PRUNED_ARTIFACT,
}


@dataclass(frozen=True)
class Prediction:
    """Resultado da classificação de um laudo."""

    urgency: str
    confidence: float
    probabilities: dict[str, float]


class Predictor(ABC):
    """Interface comum a todos os backends de inferência."""

    backend: str

    def __init__(self, labels: list[str], model_version: str) -> None:
        self.labels = labels
        self.model_version = model_version

    @abstractmethod
    def predict(self, texts: list[str]) -> list[Prediction]:
        """Classifica um lote de laudos."""
        ...

    def _to_predictions(self, probabilities: np.ndarray) -> list[Prediction]:
        """Converte a matriz de probabilidades em objetos ``Prediction``."""
        predictions = []
        for row in probabilities:
            index = int(np.argmax(row))
            predictions.append(
                Prediction(
                    urgency=self.labels[index],
                    confidence=round(float(row[index]), 4),
                    probabilities={
                        label: round(float(value), 4)
                        for label, value in zip(self.labels, row, strict=True)
                    },
                )
            )
        return predictions


class SklearnPredictor(Predictor):
    """Inferência direta pelo pipeline scikit-learn — a linha de base."""

    backend = "sklearn"

    def __init__(self, artifact_path: Path, model_version: str) -> None:
        import joblib

        self._pipeline = joblib.load(artifact_path)
        super().__init__(
            labels=[str(label) for label in self._pipeline.named_steps["classifier"].classes_],
            model_version=model_version,
        )

    def predict(self, texts: list[str]) -> list[Prediction]:
        return self._to_predictions(self._pipeline.predict_proba(texts))


class OnnxPredictor(Predictor):
    """Inferência via ONNX Runtime (grafo float32 ou quantizado em INT8)."""

    def __init__(
        self,
        artifact_path: Path,
        labels: list[str],
        model_version: str,
        backend: str = "onnx",
    ) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Uma thread por sessão: o serving escala por réplica, e o paralelismo
        # interno só acrescenta disputa por CPU em lotes deste tamanho.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1

        self._session = ort.InferenceSession(
            str(artifact_path), options, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._probability_output = self._session.get_outputs()[-1].name
        self.backend = backend

        super().__init__(labels=labels, model_version=model_version)

    def predict(self, texts: list[str]) -> list[Prediction]:
        payload = np.array(texts, dtype=object).reshape(-1, 1)
        probabilities = self._session.run([self._probability_output], {self._input_name: payload})[
            0
        ]
        return self._to_predictions(np.asarray(probabilities))


def load_predictor(backend: str, version_dir: Path | None = None) -> Predictor:
    """Instancia o predictor do backend pedido a partir da versão em disco."""
    if backend not in BACKENDS:
        raise ValueError(f"Backend '{backend}' desconhecido. Disponíveis: {list(BACKENDS)}")

    version_dir = version_dir or resolve_current()
    if version_dir is None:
        raise FileNotFoundError(
            "Nenhuma versão de modelo encontrada. Rode o pipeline de treino "
            "(`make train` ou a DAG do Airflow) antes de subir a API."
        )

    artifact_path = version_dir / _ARTIFACT_BY_BACKEND[backend]
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artefato ausente para o backend '{backend}': {artifact_path}")

    metadata = read_metadata(version_dir)
    labels = metadata["labels"]
    version = metadata.get("version", version_dir.name)

    started = time.perf_counter()
    if backend == "sklearn":
        predictor: Predictor = SklearnPredictor(artifact_path, version)
    else:
        predictor = OnnxPredictor(artifact_path, labels, version, backend=backend)

    logger.info(
        "Predictor '%s' carregado da versão %s em %.0f ms",
        backend,
        version,
        (time.perf_counter() - started) * 1000,
    )
    return predictor
