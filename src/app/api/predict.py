import os
from typing import Literal

import structlog
from fastapi import APIRouter, Body, Request
from pydantic import BaseModel, Field, TypeAdapter, ValidationError, field_validator

from app.utils.machine_learning import predict_churn_class

router = APIRouter(prefix="/predict", tags=["Prediction"])
logger = structlog.get_logger()


class PredictRequestItem(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    @field_validator("MonthlyCharges", "TotalCharges", mode="before")
    @classmethod
    def normalize_charge_fields(cls, value):
        if isinstance(value, str):
            stripped = value.strip()
            if stripped == "":
                return 0.0
            value = stripped
        return float(value)


class PredictSingleResponse(BaseModel):
    prediction: Literal["Yes", "No"]


class PredictBatchResponse(BaseModel):
    predictions: list[Literal["Yes", "No"]]


class PredictErrorResponse(BaseModel):
    error: str


predict_request_adapter = TypeAdapter(PredictRequestItem | list[PredictRequestItem])


@router.post("")
async def predict(
    request: Request,
    payload: dict | list[dict] | None = Body(default=None),
) -> PredictSingleResponse | PredictBatchResponse | PredictErrorResponse:
    if payload is None:
        return PredictErrorResponse(error="Request body is required")

    try:
        validated_payload = predict_request_adapter.validate_python(payload)
    except ValidationError as exc:
        return PredictErrorResponse(error=str(exc.errors()[0]["msg"]))

    threshold = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))
    is_batch = isinstance(validated_payload, list)
    records = (
        [item.model_dump() for item in validated_payload]
        if is_batch
        else [validated_payload.model_dump()]
    )

    model = getattr(request.state, "model", None)
    if model is None:
        return PredictErrorResponse(error="Model is not available")

    try:
        predictions = predict_churn_class(
            model=model,
            records=records,
            threshold=threshold,
        )
    except Exception as exc:
        logger.exception("prediction_failed", error=str(exc))
        return PredictErrorResponse(error="Prediction failed")

    if is_batch:
        return PredictBatchResponse(predictions=predictions)
    return PredictSingleResponse(prediction=predictions[0])
