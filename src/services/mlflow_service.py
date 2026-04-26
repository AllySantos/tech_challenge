import mlflow
import mlflow.pytorch

class MLFlow:
    def __init__(self, experiment_name: str, tracking_uri: str = "mlruns", enable_metrics = True):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        if enable_metrics:
            mlflow.config.enable_system_metrics_logging()
            mlflow.config.set_system_metrics_sampling_interval(1)
        
        self._run = None

    # ── Ciclo de vida do run ──────────────────────────────────────────────────

    def start_run(self, run_name: str) -> None:
        self._run = mlflow.start_run(run_name=run_name)

    def end_run(self) -> None:
        mlflow.end_run()
        self._run = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.end_run()

    def log_params(self, params: dict) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        """Filtra automaticamente valores não numéricos antes de logar."""
        numeric = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}

        mlflow.log_metrics(numeric, step=step)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        mlflow.log_artifact(local_path, artifact_path)

    # ── Modelos ───────────────────────────────────────────────────────────────

    def log_sklearn_model(self, model, artifact_path: str = "model") -> None:
        mlflow.sklearn.log_model(model, artifact_path)

    def log_pytorch_model(self, model, artifact_path: str = "model") -> None:
        mlflow.pytorch.log_model(model, artifact_path)