import io
import os
import tarfile
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import boto3
import joblib
import pandas as pd
import structlog
import torch

from ml.utils.architecture import ChurnMLP

logger = structlog.get_logger()


TELCO_FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


@dataclass
class ModelArtifacts:
    model: torch.nn.Module
    pipeline: Any


def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    parsed_uri = urlparse(s3_uri)
    bucket_name = parsed_uri.netloc
    key = parsed_uri.path.lstrip("/")
    if parsed_uri.scheme != "s3" or not bucket_name or not key:
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    return bucket_name, key


def _download_s3_object_bytes(s3_client, s3_uri: str) -> bytes:
    bucket_name, key = _parse_s3_uri(s3_uri)
    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    return response["Body"].read()


def _infer_input_dim_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int:
    for tensor in state_dict.values():
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
            return int(tensor.shape[1])
    raise RuntimeError("Could not infer input dimension from model state dict")


def _load_model_from_bytes(model_bytes: bytes) -> torch.nn.Module:
    loaded = torch.load(io.BytesIO(model_bytes), map_location="cpu")

    if isinstance(loaded, torch.nn.Module):
        model = loaded
    elif isinstance(loaded, dict):
        input_dim = _infer_input_dim_from_state_dict(loaded)
        model = ChurnMLP(input_dim=input_dim)
        model.load_state_dict(loaded)
    else:
        raise RuntimeError("Unsupported model artifact format")

    model.eval()
    return model


def _extract_tar_member(tar: tarfile.TarFile, member_name: str) -> bytes:
    for member in tar.getmembers():
        if member.name.endswith(member_name):
            extracted = tar.extractfile(member)
            if extracted is None:
                break
            return extracted.read()
    raise RuntimeError(f"Missing '{member_name}' in model package")


def _load_artifacts_from_tar_bytes(package_bytes: bytes) -> ModelArtifacts:
    with tarfile.open(fileobj=io.BytesIO(package_bytes), mode="r:gz") as tar:
        model_bytes = _extract_tar_member(tar, "model.pth")
        pipeline_bytes = _extract_tar_member(tar, "pipeline.pkl")

    model = _load_model_from_bytes(model_bytes)
    pipeline = joblib.load(io.BytesIO(pipeline_bytes))
    return ModelArtifacts(model=model, pipeline=pipeline)


def load_machine_learning_model():
    model_path = os.getenv("MODEL_S3_URI")
    pipeline_path = os.getenv("PIPELINE_S3_URI")

    if not model_path:
        logger.warning("machine_learning_load_skipped", reason="MODEL_S3_URI_not_set")
        return None

    logger.info("machine_learning_start_load", model_path=model_path)

    try:
        start_time = time.perf_counter()

        s3_client = boto3.client("s3")
        model_bytes = _download_s3_object_bytes(s3_client, model_path)

        if model_path.endswith(".tar.gz"):
            model_artifacts = _load_artifacts_from_tar_bytes(model_bytes)
        else:
            if not pipeline_path:
                raise RuntimeError(
                    "PIPELINE_S3_URI must be set when MODEL_S3_URI is not a .tar.gz package"
                )

            pipeline_bytes = _download_s3_object_bytes(s3_client, pipeline_path)
            model = _load_model_from_bytes(model_bytes)
            pipeline = joblib.load(io.BytesIO(pipeline_bytes))
            model_artifacts = ModelArtifacts(model=model, pipeline=pipeline)

        process_time = time.perf_counter() - start_time
        logger.info(
            "machine_learning_loaded",
            model_path=str(model_path),
            duration_ms=round(process_time * 1000, 2),
        )
        return model_artifacts

    except Exception as e:
        logger.exception("machine_learning_error_load", model_path=model_path, error=str(e))
        raise


def predict_churn_class(model: ModelArtifacts, records: list[dict], threshold: float) -> list[str]:
    if model is None:
        raise RuntimeError("Model artifacts are not loaded")

    payload_df = pd.DataFrame(records, columns=TELCO_FEATURE_COLUMNS)
    transformed = model.pipeline.transform(payload_df)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    model_input = torch.tensor(transformed, dtype=torch.float32)

    model.model.eval()
    with torch.no_grad():
        logits = model.model(model_input)
        probabilities = torch.sigmoid(logits)

    predictions = (probabilities >= threshold).view(-1).tolist()

    return ["Yes" if is_churn else "No" for is_churn in predictions]
