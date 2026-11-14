"""Simula a passagem do tempo: 4 períodos de produção com drift crescente."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score

from credit_scoring_observability.contracts import BILL, validate_batch
from credit_scoring_observability.data import TARGET
from credit_scoring_observability.prep import PRODUCTION_BASE_CSV
from credit_scoring_observability.train import MODEL_PATH, feature_columns

PRODUCTION_DIR = Path("data/production")

# Intensidade crescente. O período 1 é o controle (sem drift nenhum).
PERIODS = {
    1: dict(inflation=0.00, delay_prob=0.00, younger_years=0, concept_rate=0.00),
    2: dict(inflation=0.10, delay_prob=0.08, younger_years=4, concept_rate=0.05),
    3: dict(inflation=0.25, delay_prob=0.18, younger_years=8, concept_rate=0.10),
    4: dict(inflation=0.45, delay_prob=0.30, younger_years=12, concept_rate=0.18),
}


def period_path(k: int) -> Path:
    return PRODUCTION_DIR / f"period_{k}.csv"


def apply_drift(df, inflation, delay_prob, younger_years, concept_rate, rng):
    df = df.copy()
    n = len(df)

    # DATA DRIFT 1 (inflação): faturas maiores.
    df[BILL] = (df[BILL] * (1 + inflation)).round().astype(int)

    # DATA DRIFT 2 (atrasos): parte dos clientes atrasa mais um mês.
    late = rng.random(n) < delay_prob
    atual = df.loc[late, "pay_1"].to_numpy()
    df.loc[late, "pay_1"] = np.minimum(np.where(atual <= 0, 1, atual + 1), 8)

    # DATA DRIFT 3 (novo perfil): clientes mais jovens (mínimo 21 anos).
    df["age"] = (df["age"] - rng.integers(0, younger_years + 1, n)).clip(lower=21)

    # CONCEPT DRIFT: parte dos bons pagadores passa a dar calote.
    # As entradas continuam com cara de "bom pagador", mas o resultado mudou.
    flip = (df["pay_1"] <= 0) & (df[TARGET] == 0) & (rng.random(n) < concept_rate)
    df.loc[flip, TARGET] = 1
    return df


def main() -> None:
    base = pd.read_csv(PRODUCTION_BASE_CSV)
    model = joblib.load(MODEL_PATH)
    rng = np.random.default_rng(42)
    size = len(base) // len(PERIODS)

    rows = []
    for k, params in PERIODS.items():
        chunk = base.iloc[(k - 1) * size : k * size]
        df = apply_drift(chunk, rng=rng, **params)

        # Drift muda a distribuição, mas os dados continuam VÁLIDOS.
        if not validate_batch(df):
            raise RuntimeError(f"período {k} quebrou o contrato")
        df.to_csv(period_path(k), index=False)

        proba = model.predict_proba(df[feature_columns(df)])[:, 1]
        rows.append(
            {
                "periodo": k,
                "default_real": df[TARGET].mean(),
                "bill_amt1_medio": df["bill_amt1"].mean(),
                "idade_media": df["age"].mean(),
                "pct_atraso": (df["pay_1"] >= 1).mean(),
                "auc": roc_auc_score(df[TARGET], proba),
                "recall": recall_score(df[TARGET], proba >= 0.5),
            }
        )

    print("\nEvolução ao longo do tempo (baseline: AUC 0.791, default 22.1%):")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
