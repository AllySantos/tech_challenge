"""Avaliação de qualidade do classificador de urgência."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.configs.settings import settings

logger = logging.getLogger(__name__)


def evaluate_pipeline(pipeline, validation_path: Path | str | None = None) -> dict:
    """Calcula as métricas do pipeline sobre o conjunto de validação.

    O F1 macro é a métrica de decisão: ele pesa as três urgências igualmente,
    o que evita que o modelo maximize o agregado às custas da classe
    ``urgente`` — justamente a que não pode ser perdida na triagem.
    """
    validation_path = Path(validation_path or settings.processed_dir / "validation.csv")
    df = pd.read_csv(validation_path)

    y_true = df["urgency"]
    y_pred = pipeline.predict(df["text"])
    labels = [str(label) for label in pipeline.named_steps["classifier"].classes_]

    per_class = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    metrics = {
        "n_samples": int(len(df)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
        "per_class": {
            label: {
                "precision": round(per_class[label]["precision"], 4),
                "recall": round(per_class[label]["recall"], 4),
                "f1": round(per_class[label]["f1-score"], 4),
                "support": int(per_class[label]["support"]),
            }
            for label in labels
            if label in per_class
        },
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }

    logger.info(
        "Avaliação: accuracy=%.4f macro_f1=%.4f em %d laudos",
        metrics["accuracy"],
        metrics["macro_f1"],
        metrics["n_samples"],
    )
    return metrics


def save_metrics(metrics: dict, output_path: Path | str | None = None) -> Path:
    """Persiste as métricas em JSON."""
    output_path = Path(output_path or settings.metrics_root / "evaluation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Métricas gravadas em %s", output_path)
    return output_path


def main() -> None:
    import joblib

    from src.models.registry import PIPELINE_ARTIFACT, resolve_current

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    version_dir = resolve_current()
    if version_dir is None:
        raise SystemExit("Nenhuma versão de modelo promovida. Rode o treino antes.")

    pipeline = joblib.load(version_dir / PIPELINE_ARTIFACT)
    save_metrics(evaluate_pipeline(pipeline))


if __name__ == "__main__":
    main()
