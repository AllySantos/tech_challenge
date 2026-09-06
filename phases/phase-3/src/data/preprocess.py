"""Limpeza dos laudos e split estratificado por urgência."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.configs.settings import URGENCY_LABELS, settings

logger = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normaliza espaçamento e caixa, preservando a pontuação clínica.

    A pontuação é mantida de propósito: valores como ``10.5 mg/dL`` e faixas
    como ``120/80`` carregam sinal e são capturados pelos n-gramas do TF-IDF.
    """
    return _WHITESPACE.sub(" ", str(text)).strip().lower()


def clean_dataframe(df: pd.DataFrame, min_chars: int | None = None) -> pd.DataFrame:
    """Remove laudos vazios, curtos demais ou duplicados."""
    min_chars = settings.min_abstract_chars if min_chars is None else min_chars

    out = df.copy()
    out["text"] = out["text"].astype(str).map(clean_text)

    before = len(out)
    out = out[out["text"].str.len() >= min_chars]
    out = out.drop_duplicates(subset="text")
    out = out[out["urgency"].isin(URGENCY_LABELS)]
    out = out.reset_index(drop=True)

    logger.info("Limpeza: %d → %d laudos (%d descartados)", before, len(out), before - len(out))
    return out


def split_dataset(
    df: pd.DataFrame,
    validation_size: float | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide em treino e validação preservando a proporção entre urgências."""
    validation_size = settings.validation_size if validation_size is None else validation_size
    seed = settings.random_seed if seed is None else seed

    train_df, val_df = train_test_split(
        df,
        test_size=validation_size,
        random_state=seed,
        stratify=df["urgency"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def preprocess(
    input_path: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> tuple[Path, Path]:
    """Executa a etapa completa e grava ``train.csv`` e ``validation.csv``."""
    input_path = Path(input_path or settings.raw_dir / "abstracts_labeled.csv")
    output_dir = Path(output_dir or settings.processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = clean_dataframe(pd.read_csv(input_path))
    train_df, val_df = split_dataset(df)

    train_path = output_dir / "train.csv"
    val_path = output_dir / "validation.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    logger.info(
        "Split gravado: treino=%d validação=%d | distribuição=%s",
        len(train_df),
        len(val_df),
        train_df["urgency"].value_counts(normalize=True).round(3).to_dict(),
    )
    return train_path, val_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    preprocess()


if __name__ == "__main__":
    main()
