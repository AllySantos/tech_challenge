import logging
import os
import warnings
from pathlib import Path
from typing import Optional

import boto3
import mlflow
import mlflow.pytorch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_URI = f"sqlite:///{_PROJECT_ROOT / 'mlflow.db'}"

logger = logging.getLogger(__name__)


class MLFlowService:
    @staticmethod
    def _normalize_optional(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None

        normalized = value.strip()
        if normalized == "" or normalized.lower() in {"none", "null"}:
            return None

        return normalized

    @staticmethod
    def _describe_tracking_server(tracking_server_name: Optional[str]) -> Optional[dict]:
        if not tracking_server_name:
            return None

        try:
            sm_client = boto3.client("sagemaker")
            return sm_client.describe_mlflow_tracking_server(
                TrackingServerName=tracking_server_name
            )
        except Exception as e:
            logger.warning(f"Failed to describe tracking server {tracking_server_name}: {e}")
            return None

    @classmethod
    def resolve_tracking_configuration(
        cls,
        tracking_uri: Optional[str] = None,
        tracking_arn: Optional[str] = None,
        tracking_server_name: Optional[str] = None,
    ) -> tuple[Optional[str], Optional[str]]:

        resolved_uri = cls._normalize_optional(tracking_uri or os.getenv("MLFLOW_TRACKING_URI"))
        resolved_arn = cls._normalize_optional(
            tracking_arn
            or os.getenv("MLFLOW_TRACKING_ARN")
            or os.getenv("MLFLOW_TRACKING_SERVER_ARN")
        )

        server_name = cls._normalize_optional(
            tracking_server_name
            or os.getenv("MLFLOW_TRACKING_SERVER_NAME")
            or f"{os.getenv('APP_NAME', 'churn-prediction')}-tracking-server"
        )

        if not resolved_uri or not resolved_arn:
            server_description = cls._describe_tracking_server(server_name)
            if server_description:
                resolved_uri = resolved_uri or cls._normalize_optional(
                    server_description.get("TrackingServerUrl")
                )
                resolved_arn = resolved_arn or cls._normalize_optional(
                    server_description.get("TrackingServerArn")
                )

        if resolved_arn:
            resolved_uri = resolved_arn
            os.environ["MLFLOW_TRACKING_ARN"] = resolved_arn
            os.environ["MLFLOW_TRACKING_SERVER_ARN"] = resolved_arn

        if resolved_uri:
            os.environ["MLFLOW_TRACKING_URI"] = resolved_uri

        return resolved_uri, resolved_arn

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        enable_metrics: bool = False,
    ):
        resolved_uri, tracking_arn = self.resolve_tracking_configuration(tracking_uri=tracking_uri)

        if not resolved_uri:
            resolved_uri = _DEFAULT_URI

        if resolved_uri != _DEFAULT_URI and not tracking_arn:
            logger.warning("Missing MLFLOW tracking ARN; SageMaker MLflow auth will likely fail.")

        mlflow.set_tracking_uri(resolved_uri)
        mlflow.set_experiment(experiment_name)
        warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")

        if enable_metrics:
            mlflow.config.enable_system_metrics_logging()
            mlflow.config.set_system_metrics_sampling_interval(1)

        self._run = None

    def start_run(self, run_name: str):
        self._run = mlflow.start_run(run_name=run_name)
        return self._run

    def end_run(self) -> None:
        if mlflow.active_run():
            mlflow.end_run()
        self._run = None

    def __enter__(self):
        # We return self, but the user still needs to call start_run() explicitly.
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()

    def log_params(self, params: dict) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        numeric = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        if numeric:
            mlflow.log_metrics(numeric, step=step)

    def log_artifact(self, local_path: str, name: str | None = None) -> None:
        mlflow.log_artifact(local_path, name)

    def log_sklearn_model(self, model, name) -> None:
        mlflow.sklearn.log_model(model, name)

    def log_pytorch_model(self, model, name, export_model=False, **kwargs) -> None:
        mlflow.pytorch.log_model(model, name=name, export_model=export_model, **kwargs)
