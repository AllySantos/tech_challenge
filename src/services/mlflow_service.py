import warnings
from pathlib import Path

import mlflow
import mlflow.pytorch

# Raiz do projeto: src/services/../../ = project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_URI = f"sqlite:///{_PROJECT_ROOT / 'mlflow.db'}"

class MLFlowService:
    def __init__(self, experiment_name: str, tracking_uri: str = None, enable_metrics = False):
        mlflow.set_tracking_uri(tracking_uri or _DEFAULT_URI)
        mlflow.set_experiment(experiment_name)
        warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")

        if enable_metrics:
            mlflow.config.enable_system_metrics_logging()
            mlflow.config.set_system_metrics_sampling_interval(1)
        
        self._run = None

    # ── Ciclo de vida do run ──────────────────────────────────────────────────

    def start_run(self, run_name: str) -> None:
        self._run = mlflow.start_run(run_name=run_name)
        return self._run  # Retorna o gerenciador de contexto

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

    def log_artifact(self, local_path: str, name: str | None = None) -> None:
        mlflow.log_artifact(local_path, name)

    # ── Modelos ───────────────────────────────────────────────────────────────

    def log_sklearn_model(self, model, name) -> None:
        mlflow.sklearn.log_model(model, name)

    def log_pytorch_model(self, model, name, export_model = False, **kwargs) -> None:
        mlflow.pytorch.log_model(model, name=name, export_model=export_model, **kwargs)