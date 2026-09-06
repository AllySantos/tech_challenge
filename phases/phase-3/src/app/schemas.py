"""Contratos de entrada e saída da API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from src.configs.settings import settings


class TriageRequest(BaseModel):
    """Laudo submetido para triagem."""

    text: str = Field(
        ...,
        min_length=1,
        description="Texto integral do laudo médico.",
        examples=["Patient presents with acute chest pain radiating to the left arm..."],
    )

    @field_validator("text")
    @classmethod
    def reject_blank_or_oversized(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("O laudo não pode ser vazio ou conter apenas espaços.")
        if len(stripped) > settings.max_text_length:
            raise ValueError(
                f"Laudo excede o limite de {settings.max_text_length} caracteres "
                f"({len(stripped)} recebidos)."
            )
        return stripped


class BatchTriageRequest(BaseModel):
    """Lote de laudos submetidos em uma única chamada."""

    items: list[TriageRequest] = Field(..., min_length=1, max_length=100)


class TriageResponse(BaseModel):
    """Classificação de urgência de um laudo."""

    urgency: str = Field(..., description="Nível de urgência previsto.")
    confidence: float = Field(..., description="Probabilidade da classe vencedora.")
    probabilities: dict[str, float] = Field(..., description="Probabilidade por classe.")
    model_version: str = Field(..., description="Versão do modelo que atendeu a requisição.")
    backend: str = Field(..., description="Backend de inferência em uso.")
    inference_ms: float = Field(..., description="Tempo de inferência, em milissegundos.")


class BatchTriageResponse(BaseModel):
    """Resultado de uma triagem em lote."""

    results: list[TriageResponse]
    count: int


class HealthResponse(BaseModel):
    """Estado do serviço."""

    status: str
    model_loaded: bool
    model_version: str | None = None
    backend: str | None = None
    labels: list[str] = []
