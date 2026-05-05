from unittest.mock import patch

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.api.predict import router

app = FastAPI()


@app.middleware("http")
async def inject_model(request: Request, call_next):
    request.state.model = object()
    return await call_next(request)


app.include_router(router)

client = TestClient(app)


SAMPLE_RECORD = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 10,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 1397.5,
}


def test_predict_single_record_success():
    with patch("app.api.predict.predict_churn_class", return_value=["Yes"]) as mock_predict:
        response = client.post("/predict", json=SAMPLE_RECORD)

    assert response.status_code == 200
    assert response.json() == {"prediction": "Yes"}
    mock_predict.assert_called_once()


def test_predict_batch_success():
    payload = [SAMPLE_RECORD, {**SAMPLE_RECORD, "tenure": 24}]

    with patch("app.api.predict.predict_churn_class", return_value=["No", "Yes"]) as mock_predict:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == {"predictions": ["No", "Yes"]}
    mock_predict.assert_called_once()


def test_predict_validation_error_for_invalid_value():
    invalid_payload = {**SAMPLE_RECORD, "gender": "Unknown"}

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 400
    assert response.json() == {"error": "gender: Input should be 'Female' or 'Male'"}


def test_predict_returns_reason_for_malformed_json():
    response = client.post(
        "/predict",
        data='{"gender": ',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json()["error"].startswith("Malformed JSON:")


def test_predict_returns_reason_for_missing_field():
    payload = dict(SAMPLE_RECORD)
    payload.pop("Contract")

    response = client.post("/predict", json=payload)

    assert response.status_code == 400
    assert response.json() == {"error": "Contract: Field required"}


def test_predict_returns_reason_for_unknown_field():
    response = client.post("/predict", json={**SAMPLE_RECORD, "invalidField": "value"})

    assert response.status_code == 400
    assert response.json() == {"error": "invalidField: Extra inputs are not permitted"}


def test_predict_accepts_blank_total_charges_as_zero():
    payload = {**SAMPLE_RECORD, "TotalCharges": ""}

    with patch("app.api.predict.predict_churn_class", return_value=["No"]):
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == {"prediction": "No"}


def test_predict_returns_500_on_inference_error():
    with patch("app.api.predict.predict_churn_class", side_effect=RuntimeError("boom")):
        response = client.post("/predict", json=SAMPLE_RECORD)

    assert response.status_code == 500
    assert response.json() == {"error": "Prediction failed"}


def test_predict_returns_500_when_model_missing():
    app_without_model = FastAPI()

    @app_without_model.middleware("http")
    async def inject_empty_model(request: Request, call_next):
        request.state.model = None
        return await call_next(request)

    app_without_model.include_router(router)
    local_client = TestClient(app_without_model)

    response = local_client.post("/predict", json=SAMPLE_RECORD)

    assert response.status_code == 500
    assert response.json() == {"error": "Model is not available"}
