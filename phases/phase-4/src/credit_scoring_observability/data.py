"""Carrega o dataset bruto do UCI, padroniza nomes e imprime um perfil."""

from pathlib import Path

import pandas as pd

RAW_XLS = Path("data/raw/default of credit card clients.xls")
RAW_CSV = Path("data/raw/credit_raw.csv")
TARGET = "default"


def load_raw() -> pd.DataFrame:
    """Lê o .xls (cabeçalho real está na 2ª linha) e padroniza as colunas."""
    df = pd.read_excel(RAW_XLS, header=1)
    # O UCI não tem PAY_1: a coluna de setembro se chama PAY_0.
    df = df.rename(columns={"PAY_0": "PAY_1", "default payment next month": TARGET})
    df.columns = [c.lower() for c in df.columns]
    return df


def profile(df: pd.DataFrame) -> None:
    """Imprime o que precisamos saber para escrever o contrato de dados."""
    print("shape:", df.shape)
    print("nulos:", int(df.isna().sum().sum()))
    print("ids duplicados:", int(df["id"].duplicated().sum()))
    print("linhas duplicadas (sem id):", int(df.drop(columns="id").duplicated().sum()))
    print(f"taxa de default: {df[TARGET].mean():.2%}")
    print("idade min/max:", df["age"].min(), df["age"].max())
    print("limit_bal min/max:", df["limit_bal"].min(), df["limit_bal"].max())
    for col in ["sex", "education", "marriage", "pay_1"]:
        print(f"\n{col}:\n{df[col].value_counts().sort_index().to_string()}")


if __name__ == "__main__":
    data = load_raw()
    data.to_csv(RAW_CSV, index=False)
    print(f"salvo em {RAW_CSV}\n")
    profile(data)
