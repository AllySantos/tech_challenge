"""Factory

Centraliza a criação de arquiteturas de rede neural, permitindo trocar o
modelo usado no treino sem alterar
"""

from __future__ import annotations

import torch
from torch import nn


class EmbeddingMLPRecommender(nn.Module):
    """MLP embedding-based para recomendação
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        hidden_dims: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        layers: list[nn.Module] = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """Calcula o logit de interação para pares (usuário, item).

        Args:
            user_idx: Tensor de índices de usuário, shape (batch,).
            item_idx: Tensor de índices de item, shape (batch,).

        Returns:
            Tensor de logits, shape (batch,).
        """
        x = torch.cat([self.user_embedding(user_idx), self.item_embedding(item_idx)], dim=1)
        return self.mlp(x).squeeze(-1)


_REGISTRY: dict[str, type[nn.Module]] = {
    "embedding_mlp": EmbeddingMLPRecommender,
}


def create_model(model_type: str, num_users: int, num_items: int, **kwargs: object) -> nn.Module:
    """Cria uma instância de modelo a partir do tipo registrado.

    Args:
        model_type: Chave registrada em _REGISTRY
        num_users: Quantidade de usuários distintos (tamanho do embedding)
        num_items: Quantidade de itens distintos (tamanho do embedding)

    Returns:
        Instância de `nn.Module` pronta para treino.

    Raises:
        ValueError: Se `model_type` não estiver registrado.
    """
    if model_type not in _REGISTRY:
        raise ValueError(
            f"model_type '{model_type}' desconhecido. Opções: {list(_REGISTRY)}"
        )
    return _REGISTRY[model_type](num_users=num_users, num_items=num_items, **kwargs)
