from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    """Central project configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "recsys-retailrocket"

    # Paths

    
    data_raw_path: str = "data/raw"
    data_processed_path: str = "data/processed"
    models_path: str = "models"

    # Training
    random_seed: int = 42
    batch_size: int = 256
    learning_rate: float = 0.001
    num_epochs: int = 50

    promote_to_production : str = "true"

settings = Settings()
