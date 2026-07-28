"""Engenharia de Features

Uso:
    python -m src.features.build_features --params params.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import yaml


# Pode trocar a estratégia de encoding sem afetar demais etapas
class IdEncodingStrategy(Protocol):
    """Codificar IDs
    """

    def fit(self, values: pd.Series) -> None: ...

    def transform(self, values: pd.Series) -> np.ndarray: ...


class LabelEncodingStrategy:
    """Codifica valores categóricos em índices  [0, n)."""

    def __init__(self) -> None:
        self._mapping: dict[int, int] = {}

    def fit(self, values: pd.Series) -> None:
        """Aprende o mapeamento a partir dos valores únicos observados."""
        unique_values = sorted(values.unique())
        self._mapping = {value: idx for idx, value in enumerate(unique_values)}

    def transform(self, values: pd.Series) -> np.ndarray:
        """Aplica o mapeamento aprendido; valores desconhecidos viram -1."""
        return values.map(self._mapping).fillna(-1).astype(int).to_numpy()

    @property
    def size(self) -> int:
        """Quantidade de categorias conhecidas pelo encoder."""
        return len(self._mapping)


def encode_split(
    df: pd.DataFrame, user_encoder: IdEncodingStrategy, item_encoder: IdEncodingStrategy
) -> dict[str, np.ndarray]:
    """Aplica os encoders a um split e descarta interações com IDs não vistos.

    Args:
        df: Split já limpo.
        user_encoder: Encoder de usuários já ajustado (fit) no treino.
        item_encoder: Encoder de itens já ajustado (fit) no treino.

    Returns:
        Dicionário com arrays `user_idx` e `item_idx` prontos para uso.
    """
    user_idx = user_encoder.transform(df["user_id"])
    item_idx = item_encoder.transform(df["item_id"])
    mask = (user_idx >= 0) & (item_idx >= 0)
    return {"user_idx": user_idx[mask], "item_idx": item_idx[mask]}


def main() -> None:
    """Ponto de entrada do stage `feature_eng`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    args = parser.parse_args()

    parent_folder = Path.cwd().parent.parent


    params = yaml.safe_load(args.params.read_text())["feature_eng"]
    input_dir = parent_folder.joinpath(params["input_dir"])
    output_dir = parent_folder.joinpath(params["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(input_dir / "train.csv")
    user_encoder, item_encoder = LabelEncodingStrategy(), LabelEncodingStrategy()
    user_encoder.fit(train_df["user_id"])
    item_encoder.fit(train_df["item_id"])

    for split_name in ("train", "val", "test"):
        split_df = pd.read_csv(input_dir / f"{split_name}.csv")
        arrays = encode_split(split_df, user_encoder, item_encoder)
        np.savez(output_dir / f"{split_name}.npz", **arrays)
        print(f"feature_eng: {split_name} -> {len(arrays['user_idx'])} interações codificadas")

    meta = {"num_users": user_encoder.size, "num_items": item_encoder.size}
    (output_dir / "feature_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"feature_eng: {meta}")


if __name__ == "__main__":
    main()