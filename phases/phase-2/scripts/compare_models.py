"""Compare MLP recommender against popularity baseline."""

import json
import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.configs.settings import settings
from src.evaluation.evaluator import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.models.factory import ModelFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_baseline(
    df: pd.DataFrame,
    matrix: sp.csr_matrix,
    n_users: int = 500,
    k: int = 10,
) -> dict:
    """Evaluate popularity baseline model."""
    model = ModelFactory.create("popularity")
    model.fit(matrix.toarray())

    sample_users = np.random.default_rng(42).choice(
        df["user_idx"].unique(), size=n_users, replace=False
    )

    precision_scores, recall_scores, ndcg_scores, hr_scores = [], [], [], []

    for user_id in sample_users:
        relevant = set(df[df["user_idx"] == user_id]["item_idx"].values)
        if len(relevant) < 2:
            continue
        recommended = model.recommend(user_id=int(user_id), top_k=k)
        precision_scores.append(precision_at_k(relevant, recommended, k))
        recall_scores.append(recall_at_k(relevant, recommended, k))
        ndcg_scores.append(ndcg_at_k(relevant, recommended, k))
        hr_scores.append(hit_rate_at_k(relevant, recommended, k))

    return {
        f"precision@{k}": float(np.mean(precision_scores)),
        f"recall@{k}": float(np.mean(recall_scores)),
        f"ndcg@{k}": float(np.mean(ndcg_scores)),
        f"hit_rate@{k}": float(np.mean(hr_scores)),
    }


def load_mlp_metrics(path: str) -> dict:
    """Load MLP evaluation metrics from file."""
    with open(path) as f:
        return json.load(f)


def main() -> None:
    """Run model comparison and save results."""
    df = pd.read_csv(f"{settings.data_processed_path}/events_processed.csv")
    matrix = sp.load_npz(f"{settings.data_processed_path}/interaction_matrix.npz")

    logger.info("Evaluating baseline...")
    baseline_metrics = evaluate_baseline(df, matrix)

    logger.info("Loading MLP metrics...")
    mlp_metrics = load_mlp_metrics("metrics/eval_metrics.json")

    print("\n" + "=" * 55)
    print(f"{'Metric':<20} {'Baseline':>15} {'MLP':>15}")
    print("=" * 55)
    for metric in baseline_metrics:
        b = baseline_metrics[metric]
        m = mlp_metrics.get(metric, 0)
        winner = "✓ MLP" if m > b else "✓ BASE"
        print(f"{metric:<20} {b:>15.4f} {m:>15.4f}  {winner}")
    print("=" * 55)

    comparison = {
        "baseline": baseline_metrics,
        "mlp": mlp_metrics,
    }
    with open("metrics/comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info("Comparison saved to metrics/comparison.json")


if __name__ == "__main__":
    main()
