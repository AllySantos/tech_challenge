"""Treino do classificador de urgência: TF-IDF + Regressão Logística.

A escolha por um modelo linear esparso é deliberada. O orçamento de latência
da triagem é de poucos milissegundos por laudo, e um pipeline TF-IDF →
regressão logística entrega probabilidades calibráveis, converte limpo para
ONNX e cabe em alguns megabytes — ao contrário de um transformer, que exigiria
GPU ou aceleração dedicada para o mesmo p95.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.configs.settings import URGENCY_LABELS, settings
from src.models.registry import PIPELINE_ARTIFACT

logger = logging.getLogger(__name__)


def set_seeds(seed: int | None = None) -> None:
    """Fixa as seeds para tornar o treino reprodutível."""
    seed = settings.random_seed if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)


def build_pipeline(
    max_features: int | None = None,
    min_df: int | None = None,
    ngram_max: int | None = None,
    c: float | None = None,
    max_iter: int | None = None,
    seed: int | None = None,
) -> Pipeline:
    """Monta o pipeline vetorizador + classificador."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features or settings.tfidf_max_features,
                    min_df=min_df or settings.tfidf_min_df,
                    ngram_range=(1, ngram_max or settings.tfidf_ngram_max),
                    sublinear_tf=True,
                    # `strip_accents` é omitido de propósito: o conversor ONNX
                    # não suporta o parâmetro, e mantê-lo faria o grafo
                    # exportado divergir do pipeline scikit-learn. A
                    # normalização de texto acontece antes, no preprocess.
                    stop_words="english",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=c or settings.logreg_c,
                    max_iter=max_iter or settings.logreg_max_iter,
                    class_weight="balanced",
                    random_state=seed or settings.random_seed,
                ),
            ),
        ]
    )


def train(
    train_path: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> tuple[Pipeline, Path]:
    """Treina o pipeline e serializa em ``output_dir/pipeline.joblib``."""
    set_seeds()

    train_path = Path(train_path or settings.processed_dir / "train.csv")
    df = pd.read_csv(train_path)

    unexpected = set(df["urgency"].unique()) - set(URGENCY_LABELS)
    if unexpected:
        raise ValueError(f"Rótulos de urgência desconhecidos no treino: {sorted(unexpected)}")

    pipeline = build_pipeline()
    pipeline.fit(df["text"], df["urgency"])

    vocabulary_size = len(pipeline.named_steps["tfidf"].vocabulary_)
    logger.info(
        "Treino concluído em %d laudos | vocabulário=%d | classes=%s",
        len(df),
        vocabulary_size,
        list(pipeline.named_steps["classifier"].classes_),
    )

    output_dir = Path(output_dir) if output_dir else None
    if output_dir is None:
        from src.models.registry import new_version

        output_dir = new_version()
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = output_dir / PIPELINE_ARTIFACT
    joblib.dump(pipeline, artifact_path, compress=3)
    logger.info("Pipeline salvo em %s (%.2f MB)", artifact_path, _size_mb(artifact_path))

    return pipeline, artifact_path


def _size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    train()


if __name__ == "__main__":
    main()
