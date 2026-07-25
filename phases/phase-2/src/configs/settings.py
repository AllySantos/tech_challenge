"""Configurações da aplicação, externalizadas via variáveis de ambiente.

Usa ``pydantic-settings`` para validar e tipar tudo que hoje vive no
``.env`` (caminhos, hiperparâmetros default, URIs do MLflow etc.), em vez
de espalhar ``os.getenv`` pelo código. Ver ``.env.example`` para a lista
de variáveis suportadas.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configurações tipadas do projeto, lidas de variáveis de ambiente/.env.

    Attributes:
        seed: Seed global para numpy, torch, sklearn e DataLoader.
        data_raw_dir: Diretório com os dados brutos (versionado via DVC).
        data_processed_dir: Diretório com os dados já pré-processados.
        models_dir: Diretório onde artefatos de modelo são salvos localmente.
        mlflow_tracking_uri: URI do servidor de tracking do MLflow.
        mlflow_experiment_name: Nome do experimento no MLflow.
        batch_size: Tamanho de batch default para treino.
        learning_rate: Taxa de aprendizado default do otimizador.
        num_epochs: Número máximo de épocas de treino.
        embedding_dim: Dimensão dos embeddings de usuário/item.
        early_stopping_patience: Épocas sem melhora antes de parar o treino.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    seed: int = Field(default=42, ge=0)

    data_raw_dir: Path = Field(default=Path("data/raw"))
    data_processed_dir: Path = Field(default=Path("data/processed"))
    models_dir: Path = Field(default=Path("models"))

    mlflow_tracking_uri: str = Field(default="http://localhost:5000")
    mlflow_experiment_name: str = Field(default="recsys-phase-2")

    batch_size: int = Field(default=256, gt=0)
    learning_rate: float = Field(default=0.001, gt=0)
    num_epochs: int = Field(default=20, gt=0)
    embedding_dim: int = Field(default=64, gt=0)
    early_stopping_patience: int = Field(default=3, ge=0)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retorna a instância singleton de :class:`Settings`.

    Usar ``lru_cache`` evita reler e revalidar o ``.env`` a cada chamada,
    e permite fazer override em testes com ``get_settings.cache_clear()``.

    Returns:
        Instância cacheada de ``Settings``.
    """
    return Settings()
