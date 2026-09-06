"""Central project configuration, loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Raiz do projeto, deduzida da posição deste arquivo (<raiz>/src/configs/).
# Todos os caminhos relativos da configuração são resolvidos contra ela, e não
# contra o diretório de trabalho do processo — o Airflow executa as tasks a
# partir de AIRFLOW_HOME, e sem esta âncora os artefatos de treino cairiam
# dentro do volume do scheduler, invisíveis para a API.
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Rótulos de urgência, do menos para o mais crítico. A ordem importa: é ela
# que define o índice de cada classe nos artefatos do modelo.
URGENCY_LABELS: tuple[str, ...] = ("normal", "atencao", "urgente")

# O corpus de origem classifica abstracts por sistema do corpo, não por
# urgência. O mapeamento abaixo deriva a prioridade de triagem a partir dessa
# taxonomia; a justificativa clínica está documentada em docs/model_card.md.
CONDITION_TO_URGENCY: dict[int, str] = {
    1: "atencao",  # neoplasms
    2: "atencao",  # digestive system diseases
    3: "urgente",  # nervous system diseases
    4: "urgente",  # cardiovascular diseases
    5: "normal",  # general pathological conditions
}


class Settings(BaseSettings):
    """Configuração do projeto lida do ambiente (ver .env.example)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Dataset -----------------------------------------------------------
    dataset_train_url: str = (
        "https://raw.githubusercontent.com/sebischair/"
        "Medical-Abstracts-TC-Corpus/main/medical_tc_train.csv"
    )
    dataset_test_url: str = (
        "https://raw.githubusercontent.com/sebischair/"
        "Medical-Abstracts-TC-Corpus/main/medical_tc_test.csv"
    )

    # --- Paths -------------------------------------------------------------
    project_root: str = str(DEFAULT_PROJECT_ROOT)
    data_raw_dir: str = "data/raw"
    data_processed_dir: str = "data/processed"
    models_dir: str = "models"
    metrics_dir: str = "metrics"
    reports_dir: str = "reports"

    # --- Treino ------------------------------------------------------------
    random_seed: int = 42
    validation_size: float = 0.15
    min_abstract_chars: int = 50
    tfidf_max_features: int = 30_000
    tfidf_min_df: int = 3
    tfidf_ngram_max: int = 2
    logreg_c: float = 4.0
    logreg_max_iter: int = 1000

    # Termos mantidos após o pruning do vocabulário. 10 mil é o joelho da
    # curva medida em docs/optimization.md: abaixo disso o F1 cai rápido, e
    # acima o ganho de latência já se esgotou.
    prune_keep_features: int = 10_000

    # --- Portões de qualidade do retreino ----------------------------------
    min_macro_f1: float = 0.60
    max_p95_latency_ms: float = 25.0

    # --- Serving -----------------------------------------------------------
    inference_backend: str = "onnx-pruned"
    max_text_length: int = 20_000
    api_title: str = "Triagem de Laudos Médicos"
    api_version: str = "1.0.0"

    # --- Benchmark ---------------------------------------------------------
    benchmark_runs: int = 500
    benchmark_warmup: int = 50

    def resolve_path(self, value: str) -> Path:
        """Resolve um caminho configurado contra a raiz do projeto.

        Caminhos absolutos passam intactos, o que permite apontar volumes
        montados em outro lugar (é o que o container da API faz com
        ``MODELS_DIR``).
        """
        path = Path(value)
        return path if path.is_absolute() else Path(self.project_root) / path

    @property
    def raw_dir(self) -> Path:
        return self.resolve_path(self.data_raw_dir)

    @property
    def processed_dir(self) -> Path:
        return self.resolve_path(self.data_processed_dir)

    @property
    def models_root(self) -> Path:
        return self.resolve_path(self.models_dir)

    @property
    def metrics_root(self) -> Path:
        return self.resolve_path(self.metrics_dir)

    @property
    def reports_root(self) -> Path:
        return self.resolve_path(self.reports_dir)


settings = Settings()
