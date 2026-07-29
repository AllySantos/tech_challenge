"""Treina e avalia os baselines para comparação com o MLP."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlflow
import numpy as np
import yaml

from src.evaluation.ranking import evaluate_scorer, load_test_positives
from src.models.baselines import PopularityBaseline


def load_split_arrays(features_dir: Path, split_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Carrega os arrays codificados (`user_idx`, `item_idx`) de um split."""
    data = np.load(features_dir / f"{split_name}.npz")
    return data["user_idx"], data["item_idx"]


def main() -> None:
    """Treina, avalia e loga cada baseline como um run separado no MLflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    args = parser.parse_args()

    all_params = yaml.safe_load(args.params.read_text())
    eval_params = all_params["evaluate"]
    features_dir = Path(eval_params["input_dir"])
    meta = json.loads((features_dir / "feature_meta.json").read_text())

    train_users, train_items = load_split_arrays(features_dir, "train")
    positives_by_user = load_test_positives(features_dir)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "recsys-phase-2"))

    baselines = {
        "popularity_baseline": PopularityBaseline().fit(
            train_users, train_items, meta["num_items"]
        ),
    }

    all_metrics: dict[str, dict[str, float]] = {}
    for name, model in baselines.items():
        metrics = evaluate_scorer(
            scorer=model,
            positives_by_user=positives_by_user,
            num_items=meta["num_items"],
            k=eval_params["k"],
            num_negatives=eval_params["num_negative_candidates"],
        )
        all_metrics[name] = metrics
        with mlflow.start_run(run_name=name):
            mlflow.set_tag("model_family", name)
            mlflow.log_params({"num_users": meta["num_users"], "num_items": meta["num_items"]})
            mlflow.log_metrics(metrics)
        print(f"{name}: {metrics}")

    Path("baseline_metrics.json").write_text(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
