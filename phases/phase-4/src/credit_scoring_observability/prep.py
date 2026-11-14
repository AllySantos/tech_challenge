"""Limpa os dados e separa referência (treino) de base de produção."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_scoring_observability.data import RAW_CSV, TARGET

REFERENCE_CSV = Path("data/reference/reference.csv")
PRODUCTION_BASE_CSV = Path("data/production/production_base.csv")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa códigos não documentados em 'outros' (regra de preparação)."""
    df = df.copy()
    df["education"] = df["education"].replace({0: 4, 5: 4, 6: 4})
    df["marriage"] = df["marriage"].replace({0: 3})
    return df


def split(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """2/3 referência, 1/3 base de produção, mantendo a taxa de default."""
    reference, production = train_test_split(
        df, test_size=1 / 3, stratify=df[TARGET], random_state=seed
    )
    return reference, production


if __name__ == "__main__":
    data = clean(pd.read_csv(RAW_CSV))
    reference, production = split(data)
    for path, frame in [(REFERENCE_CSV, reference), (PRODUCTION_BASE_CSV, production)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        print(f"{path}: {frame.shape}, default={frame[TARGET].mean():.2%}")
