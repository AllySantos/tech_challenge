"""Contrato de dados: regras que todo lote precisa cumprir antes de entrar."""

import sys

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

PAY = [f"pay_{i}" for i in range(1, 7)]
BILL = [f"bill_amt{i}" for i in range(1, 7)]
PAY_AMT = [f"pay_amt{i}" for i in range(1, 7)]

CONTRACT = DataFrameSchema(
    {
        "id": Column(int, Check.ge(1), unique=True, nullable=False),
        "limit_bal": Column(int, Check.gt(0), nullable=False),
        "sex": Column(int, Check.isin([1, 2])),
        "education": Column(int, Check.isin([1, 2, 3, 4])),
        "marriage": Column(int, Check.isin([1, 2, 3])),
        "age": Column(int, Check.in_range(18, 100)),
        **{c: Column(int, Check.in_range(-2, 9)) for c in PAY},
        # fatura pode ser negativa (cliente pagou a mais), então sem mínimo
        **{c: Column(int, nullable=False) for c in BILL},
        **{c: Column(int, Check.ge(0), nullable=False) for c in PAY_AMT},
        "default": Column(int, Check.isin([0, 1])),
    },
    strict=True,  # coluna desconhecida ou faltando também reprova
)


def validate_batch(df: pd.DataFrame) -> bool:
    """Valida o lote inteiro e mostra TODAS as violações, não só a primeira."""
    try:
        CONTRACT.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        resumo = err.failure_cases.groupby(["column", "check"]).size()
        print("CONTRATO FALHOU. Lote bloqueado:\n", resumo.to_string())
        return False
    print(f"Contrato OK: {len(df)} linhas aprovadas.")
    return True


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/reference/reference.csv"
    sys.exit(0 if validate_batch(pd.read_csv(path)) else 1)
