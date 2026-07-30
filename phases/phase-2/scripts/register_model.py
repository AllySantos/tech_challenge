"""Register the best model in MLflow Model Registry."""

import logging

import mlflow
from mlflow import MlflowClient

from src.configs.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_best_run(experiment_name: str):
    """Return the run with lowest best_loss."""
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
    return best_run


def register_model(run, model_name: str) -> str:
    """Register the model logged by train.py and return the new version.

    train.py logs the model via mlflow.pytorch.log_model() and stores the
    resulting model_uri as a run tag, so this avoids hardcoding artifact
    filenames or MLflow's internal 'runs:/' path conventions.
    """
    model_uri = run.data.tags.get("model_uri")
    if not model_uri:
        raise ValueError(
            f"Run {run.info.run_id} has no 'model_uri' tag. "
            "Was it trained with the updated train.py that logs the model "
            "via mlflow.pytorch.log_model()?"
        )

    mv = mlflow.register_model(model_uri, model_name)
    logger.info("Registered model '%s' version %s", model_name, mv.version)
    return mv.version


def promote_to_staging(model_name: str, version: str) -> None:
    """Assign the 'staging' alias to a newly registered model version."""
    client = MlflowClient()
    client.set_registered_model_alias(
        name=model_name,
        alias="staging",
        version=version,
    )
    logger.info("Model '%s' v%s promoted to 'staging'", model_name, version)


def promote_to_production(model_name: str, version: str) -> None:
    """Assign the 'production' alias to a model version already in staging."""
    client = MlflowClient()
    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=version,
    )
    logger.info("Model '%s' v%s promoted to 'production'", model_name, version)


def main() -> None:
    """Run model registration pipeline: register -> staging -> production.

    Production promotion only happens automatically when explicitly
    requested (PROMOTE_TO_PRODUCTION=true). This mirrors a real MLOps
    gate: staging is where validation happens before a human/CI decides
    to promote further.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    run = get_best_run(settings.mlflow_experiment_name)
    version = register_model(run, model_name="recsys-mlp")
    promote_to_staging(model_name="recsys-mlp", version=version)

    if settings.promote_to_production.lower() == "true":
        promote_to_production(model_name="recsys-mlp", version=version)
    else:
        logger.info("Skipping production promotion (set PROMOTE_TO_PRODUCTION=true to promote)")


if __name__ == "__main__":
    main()
