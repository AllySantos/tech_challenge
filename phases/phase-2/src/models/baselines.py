"""Baselines de recomendação para comparação.

Implementam o mesmo contrato `Scorer` usado pelo MLP (ver
`src/evaluation/ranking.py`), permitindo avaliar todos os modelos com a
mesma metodologia e as mesmas métricas
"""

from __future__ import annotations

import numpy as np


class PopularityBaseline:
    """Recomenda sempre os itens mais populares, ignorando o usuário."""

    def __init__(self) -> None:
        self._item_counts: np.ndarray | None = None

    def fit(self, item_idx: np.ndarray, num_items: int) -> PopularityBaseline:
        """Conta as interações de treino por item.

        Args:
            item_idx: Índices de item das interações de treino.
            num_items: Total de itens no catálogo.

        Returns:
            A própria instância
        """
        self._item_counts = np.bincount(item_idx, minlength=num_items).astype(float)
        return self

    def score(self, user_idx: int, item_idx: np.ndarray) -> np.ndarray:
        """Retorna a popularidade (contagem no treino) de cada candidato."""
        return self._item_counts[item_idx]
