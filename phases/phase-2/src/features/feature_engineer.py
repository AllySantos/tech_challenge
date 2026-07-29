"""Feature engineering module for RetailRocket dataset."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.configs.settings import settings


logger = logging.getLogger(__name__)


def create_interaction_matrix(df: pd.DataFrame) -> sp.csr_matrix:
    """Create sparse user-item interaction matrix with event weights."""
    event_weights = {"view": 1, "addtocart": 3, "transaction": 5}

    df = df.copy()
    df["weight"] = df["event"].map(event_weights).fillna(1)

    n_users = df["user_idx"].max() + 1
    n_items = df["item_idx"].max() + 1

    matrix = sp.csr_matrix(
        (df["weight"].values, (df["user_idx"].values, df["item_idx"].values)),
        shape=(n_users, n_items),
        dtype=np.float32,
    )

    logger.info("Sparse matrix shape: %s | nnz: %d", matrix.shape, matrix.nnz)
    return matrix


def create_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create user-level aggregated features."""
    features = (
        df.groupby("user_idx")
        .agg(
            total_events=("event", "count"),
            unique_items=("item_idx", "nunique"),
            transactions=("event", lambda x: (x == "transaction").sum()),
            add_to_cart=("event", lambda x: (x == "addtocart").sum()),
        )
        .reset_index()
    )
    logger.info("User features shape: %s", features.shape)
    return features


def run_feature_engineering(input_path: str, output_dir: str) -> None:
    """Run full feature engineering pipeline."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    logger.info("Loaded %d preprocessed events", len(df))

    user_features = create_user_features(df)
    user_features.to_csv(f"{output_dir}/user_features.csv", index=False)

    matrix = create_interaction_matrix(df)
    sp.save_npz(f"{output_dir}/interaction_matrix.npz", matrix)

    logger.info("Feature engineering complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_feature_engineering(
        input_path=f"{settings.data_processed_path}/events_processed.csv",
        output_dir=f"{settings.data_processed_path}",
    )
