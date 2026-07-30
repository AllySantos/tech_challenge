"""Evaluation module for recommendation models."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.configs.settings import settings
from src.models.train import MLPRecommender

logger = logging.getLogger(__name__)


def precision_at_k(relevant: set, recommended: list, k: int) -> float:
    """Calculate Precision@K."""
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & relevant)
    return hits / k


def recall_at_k(relevant: set, recommended: list, k: int) -> float:
    """Calculate Recall@K."""
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & relevant)
    return hits / len(relevant) if relevant else 0.0


def ndcg_at_k(relevant: set, recommended: list, k: int) -> float:
    """Calculate NDCG@K."""
    recommended_k = recommended[:k]
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(recommended_k) if item in relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(relevant: set, recommended: list, k: int) -> float:
    """Calculate Hit Rate@K."""
    return float(len(set(recommended[:k]) & relevant) > 0)


def evaluate(
    model_path: str,
    events_path: str,
    metrics_path: str,
    k: int = 10,
    n_users: int = 500,
) -> None:
    """Evaluate recommendation model on held-out interactions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(events_path)
    n_users_total = df["user_idx"].max() + 1
    n_items_total = df["item_idx"].max() + 1

    model = MLPRecommender(n_users_total, n_items_total)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    sample_users = np.random.choice(df["user_idx"].unique(), size=n_users, replace=False)

    precision_scores, recall_scores, ndcg_scores, hr_scores = [], [], [], []

    with torch.no_grad():
        for user_id in sample_users:
            relevant = set(df[df["user_idx"] == user_id]["item_idx"].values)
            if len(relevant) < 2:
                continue

            all_items = torch.arange(n_items_total, dtype=torch.long).to(device)
            user_tensor = torch.tensor([user_id] * n_items_total, dtype=torch.long).to(device)
            scores = model(user_tensor, all_items).cpu().numpy()
            recommended = np.argsort(scores)[::-1].tolist()

            precision_scores.append(precision_at_k(relevant, recommended, k))
            recall_scores.append(recall_at_k(relevant, recommended, k))
            ndcg_scores.append(ndcg_at_k(relevant, recommended, k))
            hr_scores.append(hit_rate_at_k(relevant, recommended, k))

    metrics = {
        f"precision@{k}": float(np.mean(precision_scores)),
        f"recall@{k}": float(np.mean(recall_scores)),
        f"ndcg@{k}": float(np.mean(ndcg_scores)),
        f"hit_rate@{k}": float(np.mean(hr_scores)),
    }

    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Evaluation metrics: %s", metrics)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluate(
        model_path="models/recommender.pt",
        events_path=f"{settings.data_processed_path}/events_processed.csv",
        metrics_path="metrics/eval_metrics.json",
    )
