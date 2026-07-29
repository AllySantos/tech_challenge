"""Register the best model in MLflow Model Registry."""

import logging

import mlflow
from mlflow import MlflowClient
from src.configs.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_best_run(experiment_name: str) -> str:
    """Return the run_id with lowest best_loss."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="metrics.best_loss > 0",
        order_by=["metrics.best_loss ASC"],
        max_results=1,
    )

    if not runs:
        raise ValueError("No runs found with best_loss metric.")

    best_run = runs[0]
    logger.info(
        "Best run: %s | best_loss: %.4f",
        best_run.info.run_id,
        best_run.data.metrics["best_loss"],
    )
    return best_run.info.run_id


def register_model(run_id: str, model_name: str) -> None:
    """Register model and promote to Production."""
    client = MlflowClient()

    model_uri = f"mlflow-artifacts:/1/{run_id}/artifacts/recommender.pt"
    mv = mlflow.register_model(model_uri, model_name)
    logger.info("Registered model '%s' version %s", model_name, mv.version)

    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=mv.version,
    )
    logger.info("Model promoted to 'production' alias")


def main() -> None:
    """Run model registration pipeline."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    run_id = get_best_run(settings.mlflow_experiment_name)
    register_model(run_id, model_name="recsys-mlp")


if __name__ == "__main__":
    main()
