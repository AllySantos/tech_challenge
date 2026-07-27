"""Stage 1 do pipeline DVC: limpeza e split do dataset bruto de interações.

Lê `data/raw`, remove duplicatas/nulos e usuários ou itens
com poucas interações, e gera os splits treino/validação/teste em
`data/processed/`.

"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

REQUIRED_COLUMNS = ("user_id", "item_id", "timestamp")

# Padronização do nome das colunas
RETAILROCKET_COLUMN_MAP = {"visitorid": "user_id", "itemid": "item_id"}


def load_raw_interactions(raw_path: Path) -> pd.DataFrame:
    """Carrega o CSV bruto de interações (events.csv do RetailRocket).

    Args:
        raw_path: Caminho para o CSV bruto.

    Returns:
        DataFrame com as interações carregadas.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    if not raw_path.is_file():
        raise FileNotFoundError(
            f"Dataset bruto não encontrado em '{raw_path}'. "
        )
    return pd.read_csv(raw_path)


def standardize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza colunas do RetailRocket para o schema interno.

    Args:
        df: DataFrame bruto, no schema do RetailRocket ou já padronizado.

    Returns:
        DataFrame com colunas `user_id`, `item_id`, `timestamp`, `event`.
    """
    df = df.rename(columns=RETAILROCKET_COLUMN_MAP)
    return df.drop(columns=["transactionid"], errors="ignore")


def filter_events(df: pd.DataFrame, event_types: list[str]) -> pd.DataFrame:
    """Mantém apenas os tipos de evento configurados.

    Filtra eventos de acordo com o params.yaml

    Args:
        df: DataFrame já padronizado, com coluna `event`.
        event_types: Tipos de evento a manter.

    Returns:
        DataFrame filtrado.
    """
    if "event" not in df.columns:
        return df
    return df[df["event"].isin(event_types)]


def subsample(df: pd.DataFrame, max_interactions: int | None, seed: int) -> pd.DataFrame:
    """Amostra aleatoriamente o dataset para acelerar execuções locais/dev.

    O RetailRocket tem ~2.75M eventos; treinar o pipeline completo em cada
    iteração de desenvolvimento é lento. `max_interactions` no params.yaml
    permite reduzir o volume localmente sem alterar o código — em produção
    (ou na entrega final), basta deixar `null` para usar o dataset inteiro.

    Args:
        df: DataFrame já filtrado.
        max_interactions: Limite de linhas, ou `None` para não amostrar.
        seed: Semente para reprodutibilidade da amostragem.

    Returns:
        DataFrame amostrado (ou o original, se `max_interactions` for `None`
        ou maior que o tamanho do DataFrame).
    """
    if max_interactions is None or len(df) <= max_interactions:
        return df
    return df.sample(n=max_interactions, random_state=seed)


def clean_interactions(df: pd.DataFrame, min_interactions: int) -> pd.DataFrame:
    """Remove duplicatas, nulos e usuários/itens com poucas interações.

    Args:
        df: DataFrame padronizado.
        min_interactions: Mínimo de interações por usuário e por item.

    Returns:
        DataFrame limpo.
    """
    df = df.dropna(subset=list(REQUIRED_COLUMNS)).drop_duplicates()
    for column in ("user_id", "item_id"):
        counts = df[column].value_counts()

        valid_ids = counts[counts >= min_interactions].index

        df = df[df[column].isin(valid_ids)]

    return df.sort_values("timestamp").reset_index(drop=True)


def split_by_time(
    df: pd.DataFrame, val_size: float, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide as interações em treino/validação/teste por ordem temporal.

    Split temporal (e não aleatório) evita vazamento de informação futura
    para o treino, o que é mais realista para recomendação sequencial.

    Args:
        df: DataFrame limpo, ordenado por timestamp.
        val_size: Fração final destinada à validação.
        test_size: Fração final destinada ao teste.

    Returns:
        Tupla (train_df, val_df, test_df).
    """
    n = len(df)
    test_start = int(n * (1 - test_size))
    val_start = int(test_start * (1 - val_size))
    return df.iloc[:val_start], df.iloc[val_start:test_start], df.iloc[test_start:]


def main() -> None:
    """Ponto de entrada do stage `preprocess`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    args = parser.parse_args()

    params = yaml.safe_load(args.params.read_text())["preprocess"]

    parent_folder = Path.cwd().parent.parent

    raw_df = standardize_schema(load_raw_interactions(parent_folder.joinpath(params["raw_path"])))
    raw_df = filter_events(raw_df, params["event_types"])
    raw_df = subsample(raw_df, params.get("max_interactions"), params["seed"])
    clean_df = clean_interactions(raw_df, params["min_interactions"])
    train_df, val_df, test_df = split_by_time(
        clean_df, params["val_size"], params["test_size"]
    )

    out_dir = Path(parent_folder.joinpath(params["output_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    print(
        f"preprocess: {len(clean_df)} interações válidas -> "
        f"train={len(train_df)} val={len(val_df)} test={len(test_df)}"
    )


if __name__ == "__main__":
    main()
