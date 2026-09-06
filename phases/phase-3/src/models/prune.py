"""Pruning do vocabulário por magnitude de peso.

O perfil de latência do modelo mostra que o custo dominante não é o produto
matricial do classificador — é o ``TfIdfVectorizer``, que precisa varrer o
texto contra um vocabulário de dezenas de milhares de termos a cada laudo.
Encolher esse vocabulário é, portanto, a otimização que ataca o gargalo real.

A seleção é a mesma ideia do pruning por magnitude aplicado a redes neurais:
cada termo é pontuado pelo maior peso absoluto que recebeu entre as classes, e
apenas os mais influentes sobrevivem. O classificador é então reajustado sobre
o vocabulário reduzido, para que os pesos se reacomodem à ausência dos termos
descartados em vez de herdarem valores treinados em outro espaço de features.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def rank_features(pipeline: Pipeline) -> np.ndarray:
    """Ordena os índices das features da mais para a menos influente."""
    coefficients = pipeline.named_steps["classifier"].coef_
    importance = np.abs(coefficients).max(axis=0)
    return np.argsort(importance)[::-1]


def prune_pipeline(
    pipeline: Pipeline,
    texts,
    labels,
    keep_features: int,
) -> Pipeline:
    """Devolve um novo pipeline restrito às ``keep_features`` melhores features."""
    tfidf: TfidfVectorizer = pipeline.named_steps["tfidf"]
    classifier: LogisticRegression = pipeline.named_steps["classifier"]

    original_size = len(tfidf.vocabulary_)
    if keep_features >= original_size:
        logger.info(
            "Pruning dispensado: vocabulário já tem %d termos (alvo %d)",
            original_size,
            keep_features,
        )
        return pipeline

    feature_names = tfidf.get_feature_names_out()
    # Os índices são reordenados alfabeticamente para que o vocabulário
    # resultante siga a mesma convenção do scikit-learn.
    kept = sorted(rank_features(pipeline)[:keep_features])
    vocabulary = {str(feature_names[index]): position for position, index in enumerate(kept)}

    pruned = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    vocabulary=vocabulary,
                    ngram_range=tfidf.ngram_range,
                    sublinear_tf=tfidf.sublinear_tf,
                    stop_words=tfidf.stop_words,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=classifier.C,
                    max_iter=classifier.max_iter,
                    class_weight=classifier.class_weight,
                    random_state=classifier.random_state,
                ),
            ),
        ]
    )
    pruned.fit(texts, labels)

    logger.info(
        "Vocabulário reduzido de %d para %d termos (%.0f%% descartado)",
        original_size,
        len(vocabulary),
        100 * (1 - len(vocabulary) / original_size),
    )
    return pruned
