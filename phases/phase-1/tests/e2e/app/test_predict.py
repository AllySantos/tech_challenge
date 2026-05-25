import io
import tarfile

import boto3
import joblib
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from moto import mock_aws

from app.main import app
from ml.utils.architecture import ChurnMLP

SAMPLE_RECORD = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 7,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.1,
    "TotalCharges": 623.7,
}

LOW_RISK_RECORD = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 48,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 59.4,
    "TotalCharges": 2851.2,
}

NO_INTERNET_RECORD = {
    **LOW_RISK_RECORD,
    "InternetService": "No",
    "OnlineSecurity": "No internet service",
    "OnlineBackup": "No internet service",
    "DeviceProtection": "No internet service",
    "TechSupport": "No internet service",
    "StreamingTV": "No internet service",
    "StreamingMovies": "No internet service",
    "MonthlyCharges": 20.0,
    "TotalCharges": 960.0,
}

NO_PHONE_SERVICE_RECORD = {
    **LOW_RISK_RECORD,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "MonthlyCharges": 44.8,
    "TotalCharges": 2150.4,
}

SENIOR_RECORD = {
    **SAMPLE_RECORD,
    "SeniorCitizen": 1,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 3,
}

AUTOMATIC_PAYMENT_RECORD = {
    **LOW_RISK_RECORD,
    "PaymentMethod": "Credit card (automatic)",
    "PaperlessBilling": "Yes",
}

NEW_CUSTOMER_RECORD = {
    **SAMPLE_RECORD,
    "gender": "Female",
    "tenure": 1,
    "MonthlyCharges": 95.5,
    "TotalCharges": 95.5,
}


class TelcoRulePipeline:
    def transform(self, dataframe):
        rows = []

        for _, row in dataframe.iterrows():
            rows.append(
                [
                    1.0 if row["Contract"] == "Month-to-month" else 0.0,
                    1.0 if row["PaymentMethod"] == "Electronic check" else 0.0,
                    float(row["SeniorCitizen"]),
                    max(0.0, 72.0 - float(row["tenure"])) / 72.0,
                    float(row["MonthlyCharges"]) / 100.0,
                    1.0 if row["OnlineSecurity"] == "No" else 0.0,
                ]
            )

        return np.array(rows, dtype=np.float32)


def _build_model_state_bytes() -> bytes:
    model = ChurnMLP(input_dim=6, dropout=0.0)
    model.eval()

    for parameter in model.parameters():
        parameter.data.zero_()

    first_linear = model.net[0]
    first_linear.weight.data[0, :6] = torch.tensor([1.2, 1.0, 0.9, 0.8, 0.7, 0.6])

    for batch_norm in (model.net[1], model.net[5], model.net[9]):
        batch_norm.weight.data.fill_(1.0)
        batch_norm.bias.data.zero_()
        batch_norm.running_mean.zero_()
        batch_norm.running_var.fill_(1.0)

    model.net[4].weight.data[0, 0] = 1.0
    model.net[8].weight.data[0, 0] = 1.0
    model.net[12].weight.data[0, 0] = 1.0
    model.net[12].bias.data[0] = -2.0

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()


def _build_pipeline_bytes() -> bytes:
    buffer = io.BytesIO()
    joblib.dump(TelcoRulePipeline(), buffer)
    return buffer.getvalue()


def _build_model_package_bytes() -> bytes:
    model_bytes = _build_model_state_bytes()
    pipeline_bytes = _build_pipeline_bytes()

    package_buffer = io.BytesIO()
    with tarfile.open(fileobj=package_buffer, mode="w:gz") as tar:
        model_info = tarfile.TarInfo(name="model.pth")
        model_info.size = len(model_bytes)
        tar.addfile(model_info, io.BytesIO(model_bytes))

        pipeline_info = tarfile.TarInfo(name="pipeline.pkl")
        pipeline_info.size = len(pipeline_bytes)
        tar.addfile(pipeline_info, io.BytesIO(pipeline_bytes))

    return package_buffer.getvalue()


@pytest.fixture
def predict_client(monkeypatch):
    with mock_aws():
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
        monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

        s3_client = boto3.client("s3", region_name="us-east-1")
        bucket_name = "predict-e2e-bucket"
        object_key = "models/model.tar.gz"
        s3_client.create_bucket(Bucket=bucket_name)
        s3_client.put_object(
            Bucket=bucket_name,
            Key=object_key,
            Body=_build_model_package_bytes(),
        )

        monkeypatch.setenv("MODEL_S3_URI", f"s3://{bucket_name}/{object_key}")
        monkeypatch.delenv("PIPELINE_S3_URI", raising=False)

        with TestClient(app) as client:
            yield client


def test_e2e_predict_endpoint_single_record(predict_client):
    response = predict_client.post("/predict", json=SAMPLE_RECORD)

    assert response.status_code == 200
    assert response.json() == {"prediction": "Yes"}


def test_e2e_predict_endpoint_low_risk_customer(predict_client):
    response = predict_client.post("/predict", json=LOW_RISK_RECORD)

    assert response.status_code == 200
    assert response.json() == {"prediction": "No"}


def test_e2e_predict_endpoint_senior_customer_case(predict_client):
    response = predict_client.post("/predict", json=SENIOR_RECORD)

    assert response.status_code == 200
    assert response.json() == {"prediction": "Yes"}


def test_e2e_predict_endpoint_no_internet_service_case(predict_client):
    response = predict_client.post("/predict", json=NO_INTERNET_RECORD)

    assert response.status_code == 200
    assert response.json() == {"prediction": "No"}


def test_e2e_predict_endpoint_no_phone_service_case(predict_client):
    response = predict_client.post("/predict", json=NO_PHONE_SERVICE_RECORD)

    assert response.status_code == 200
    assert response.json() == {"prediction": "No"}


def test_e2e_predict_endpoint_automatic_payment_case(predict_client):
    response = predict_client.post("/predict", json=AUTOMATIC_PAYMENT_RECORD)

    assert response.status_code == 200
    assert response.json() == {"prediction": "No"}


def test_e2e_predict_endpoint_batch_records(predict_client):
    response = predict_client.post("/predict", json=[LOW_RISK_RECORD, SAMPLE_RECORD])

    assert response.status_code == 200
    assert response.json() == {"predictions": ["No", "Yes"]}


def test_e2e_predict_endpoint_blank_total_charges_is_accepted(predict_client):
    response = predict_client.post("/predict", json={**SAMPLE_RECORD, "TotalCharges": ""})

    assert response.status_code == 200
    assert response.json() == {"prediction": "Yes"}


def test_e2e_predict_endpoint_new_customer_case(predict_client):
    response = predict_client.post("/predict", json=NEW_CUSTOMER_RECORD)

    assert response.status_code == 200
    assert response.json() == {"prediction": "Yes"}


def test_e2e_predict_endpoint_invalid_enum_returns_error_payload(predict_client):
    response = predict_client.post("/predict", json={**SAMPLE_RECORD, "gender": "Other"})

    assert response.status_code == 400
    assert response.json() == {"error": "gender: Input should be 'Female' or 'Male'"}


def test_e2e_predict_endpoint_missing_field_returns_error_payload(predict_client):
    payload = dict(SAMPLE_RECORD)
    payload.pop("Contract")
    response = predict_client.post("/predict", json=payload)

    assert response.status_code == 400
    assert response.json() == {"error": "Contract: Field required"}


def test_e2e_predict_endpoint_unknown_field_returns_error_payload(predict_client):
    response = predict_client.post("/predict", json={**SAMPLE_RECORD, "invalidField": "value"})

    assert response.status_code == 400
    assert response.json() == {"error": "invalidField: Extra inputs are not permitted"}


def test_e2e_predict_endpoint_malformed_json_returns_error_payload(predict_client):
    response = predict_client.post(
        "/predict",
        data='{"gender": ',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json()["error"].startswith("Malformed JSON:")
