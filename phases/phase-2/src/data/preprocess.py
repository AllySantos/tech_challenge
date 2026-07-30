"""RetailRocket dataset preprocessing module."""

import logging
from pathlib import Path

import pandas as pd

from src.configs.settings import settings

logger = logging.getLogger(__name__)


def load_events(path: str) -> pd.DataFrame:
    """Load and validate the events CSV file."""
    df = pd.read_csv(path)
    logger.info("Loaded %d events", len(df))
    return df


def filter_events(df: pd.DataFrame, min_interactions: int = 5) -> pd.DataFrame:
    """Remove users and items with fewer than min_interactions."""
    user_counts = df["visitorid"].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index
    df = df[df["visitorid"].isin(valid_users)]

    item_counts = df["itemid"].value_counts()
    valid_items = item_counts[item_counts >= min_interactions].index
    df = df[df["itemid"].isin(valid_items)]

    logger.info("After filtering: %d events", len(df))
    return df


def encode_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Encode user and item IDs to sequential integers."""
    user_encoder = {uid: idx for idx, uid in enumerate(df["visitorid"].unique())}
    item_encoder = {iid: idx for idx, iid in enumerate(df["itemid"].unique())}

    df = df.copy()
    df["user_idx"] = df["visitorid"].map(user_encoder)
    df["item_idx"] = df["itemid"].map(item_encoder)

    return df, user_encoder, item_encoder


def preprocess(input_path: str, output_path: str) -> None:
    """Run full preprocessing pipeline."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    df = load_events(input_path)
    df = filter_events(df)
    df, user_enc, item_enc = encode_ids(df)

    df.to_csv(output_path, index=False)
    logger.info("Saved preprocessed data to %s", output_path)
    logger.info("Users: %d | Items: %d", len(user_enc), len(item_enc))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print(settings.data_raw_path)

    preprocess(
        input_path=f"{settings.data_raw_path}/events.csv",
        output_path=f"{settings.data_processed_path}/events_processed.csv",
    )
