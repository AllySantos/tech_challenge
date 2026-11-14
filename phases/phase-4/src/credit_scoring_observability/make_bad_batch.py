"""Gera um lote com erros propositais para provar que o contrato bloqueia."""

from pathlib import Path

import numpy as np
import pandas as pd

from credit_scoring_observability.prep import PRODUCTION_BASE_CSV

BAD_BATCH_CSV = Path("data/production/bad_batch.csv")


def make_bad_batch(n: int = 500, seed: int = 7) -> pd.DataFrame:
    bad = (
        pd.read_csv(PRODUCTION_BASE_CSV)
        .sample(n, random_state=seed)
        .reset_index(drop=True)
    )
    bad.loc[0:9, "age"] = -5  # idade impossível
    bad.loc[10:19, "age"] = 17  # menor de idade
    bad.loc[20:29, "limit_bal"] = np.nan  # limite nulo
    bad.loc[30:39, "sex"] = 7  # categoria inexistente
    bad.loc[40:49, "education"] = 9  # categoria inexistente
    bad.loc[50:59, "pay_amt1"] = -1000  # pagamento negativo
    bad.loc[60:69, "id"] = bad.loc[70, "id"]  # IDs duplicados
    return bad


if __name__ == "__main__":
    make_bad_batch().to_csv(BAD_BATCH_CSV, index=False)
    print(f"lote ruim salvo em {BAD_BATCH_CSV}")
