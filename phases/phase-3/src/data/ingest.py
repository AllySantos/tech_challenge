"""Ingestão do corpus de laudos e derivação do rótulo de urgência.

O Medical Abstracts TC Corpus rotula cada abstract pelo sistema do corpo
acometido. A triagem hospitalar, porém, precisa de uma prioridade de
atendimento — daí o mapeamento determinístico definido em
``CONDITION_TO_URGENCY``.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import pandas as pd

from src.configs.settings import CONDITION_TO_URGENCY, settings

logger = logging.getLogger(__name__)

RAW_COLUMNS = ("condition_label", "medical_abstract")


def download(url: str, destination: Path) -> Path:
    """Baixa o CSV para ``destination``, reaproveitando o arquivo se já existir."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0:
        logger.info("Reaproveitando arquivo já baixado: %s", destination)
        return destination

    logger.info("Baixando %s", url)
    urllib.request.urlretrieve(url, destination)  # noqa: S310 - URL fixa e pública
    logger.info("Salvo em %s (%d bytes)", destination, destination.stat().st_size)
    return destination


def map_urgency(df: pd.DataFrame) -> pd.DataFrame:
    """Traduz ``condition_label`` numérico para o rótulo de urgência."""
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no CSV bruto: {sorted(missing)}")

    unknown = set(df["condition_label"].unique()) - set(CONDITION_TO_URGENCY)
    if unknown:
        raise ValueError(f"Classes sem mapeamento de urgência: {sorted(unknown)}")

    out = df.rename(columns={"medical_abstract": "text"})
    out["urgency"] = out["condition_label"].map(CONDITION_TO_URGENCY)
    return out[["text", "urgency", "condition_label"]]


def ingest(
    train_url: str | None = None,
    test_url: str | None = None,
    raw_dir: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """Baixa treino e teste, aplica o mapeamento e grava um CSV único.

    Os dois arquivos são concatenados de propósito: a divisão oficial do corpus
    é orientada à taxonomia original, e refazemos o split de forma
    estratificada por urgência em ``src.data.preprocess``.
    """
    raw_dir = raw_dir or settings.raw_dir
    output_path = output_path or (settings.raw_dir / "abstracts_labeled.csv")

    frames = []
    for name, url in (
        ("medical_tc_train.csv", train_url or settings.dataset_train_url),
        ("medical_tc_test.csv", test_url or settings.dataset_test_url),
    ):
        path = download(url, raw_dir / name)
        frames.append(pd.read_csv(path))

    combined = pd.concat(frames, ignore_index=True)
    labeled = map_urgency(combined)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(output_path, index=False)

    distribution = labeled["urgency"].value_counts().to_dict()
    logger.info("Ingestão concluída: %d laudos %s", len(labeled), distribution)
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ingest()


if __name__ == "__main__":
    main()
