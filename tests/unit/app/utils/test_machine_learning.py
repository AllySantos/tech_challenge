import io
import tarfile
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pytest
import torch

from app.utils.machine_learning import (
    ModelArtifacts,
    _infer_input_dim_from_state_dict,
    _load_artifacts_from_tar_bytes,
    _load_model_from_bytes,
    _parse_s3_uri,
    load_machine_learning_model,
    predict_churn_class,
)
from ml.utils.architecture import ChurnMLP


def _fake_getenv(values: dict[str, str | None]):
    def _reader(key, default=None):
        return values.get(key, default)

    return _reader


class FakePipeline:
    def __init__(self, output):
        self.output = output

    def transform(self, _dataframe):
        return self.output


class DummyModel(torch.nn.Module):
    def forward(self, x):
        return torch.ones(x.shape[0])


def _build_model_state_bytes(input_dim: int = 19) -> bytes:
    model = ChurnMLP(input_dim=input_dim)
    model_buffer = io.BytesIO()
    torch.save(model.state_dict(), model_buffer)
    return model_buffer.getvalue()


def _build_pipeline_bytes() -> bytes:
    pipeline_buffer = io.BytesIO()
    joblib.dump(FakePipeline(np.ones((1, 19), dtype=np.float32)), pipeline_buffer)
    return pipeline_buffer.getvalue()


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


def test_parse_s3_uri_success():
    bucket, key = _parse_s3_uri("s3://my-bucket/models/model.tar.gz")
    assert bucket == "my-bucket"
    assert key == "models/model.tar.gz"


def test_parse_s3_uri_invalid():
    with pytest.raises(ValueError):
        _parse_s3_uri("https://example.com/model.tar.gz")


def test_infer_input_dim_from_state_dict_success():
    state_dict = ChurnMLP(input_dim=19).state_dict()
    assert _infer_input_dim_from_state_dict(state_dict) == 19


def test_load_model_from_bytes_with_state_dict():
    model = _load_model_from_bytes(_build_model_state_bytes(input_dim=19))
    assert isinstance(model, ChurnMLP)


def test_load_artifacts_from_tar_bytes_success():
    artifacts = _load_artifacts_from_tar_bytes(_build_model_package_bytes())
    assert isinstance(artifacts, ModelArtifacts)
    assert isinstance(artifacts.model, torch.nn.Module)
    assert hasattr(artifacts.pipeline, "transform")


def test_predict_churn_class_success():
    artifacts = ModelArtifacts(
        model=DummyModel(),
        pipeline=FakePipeline(np.zeros((2, 19), dtype=np.float32)),
    )
    records = [{"gender": "Female"}, {"gender": "Male"}]

    result = predict_churn_class(model=artifacts, records=records, threshold=0.5)

    assert result == ["Yes", "Yes"]


def test_predict_churn_class_without_artifacts():
    with pytest.raises(RuntimeError):
        predict_churn_class(model=None, records=[], threshold=0.5)


@patch("app.utils.machine_learning.os.getenv")
def test_load_machine_learning_model_no_env_var(mock_getenv):
    mock_getenv.side_effect = _fake_getenv({"MODEL_S3_URI": None})

    result = load_machine_learning_model()

    assert result is None
    assert mock_getenv.call_count >= 1


@patch("app.utils.machine_learning.boto3.client")
@patch("app.utils.machine_learning.os.getenv")
def test_load_machine_learning_model_success_with_tar_package(mock_getenv, mock_boto_client):
    mock_getenv.side_effect = _fake_getenv(
        {
            "MODEL_S3_URI": "s3://my-bucket/models/model.tar.gz",
        }
    )

    mock_s3_instance = MagicMock()
    mock_boto_client.return_value = mock_s3_instance

    mock_body = MagicMock()
    mock_body.read.return_value = _build_model_package_bytes()
    mock_s3_instance.get_object.return_value = {"Body": mock_body}

    result = load_machine_learning_model()

    assert isinstance(result, ModelArtifacts)
    mock_boto_client.assert_called_once_with("s3")
    mock_s3_instance.get_object.assert_called_once_with(
        Bucket="my-bucket", Key="models/model.tar.gz"
    )


@patch("app.utils.machine_learning.boto3.client")
@patch("app.utils.machine_learning.os.getenv")
def test_load_machine_learning_model_non_package_requires_pipeline(mock_getenv, mock_boto_client):
    mock_getenv.side_effect = _fake_getenv({"MODEL_S3_URI": "s3://my-bucket/models/model.pth"})

    with pytest.raises(RuntimeError) as excinfo:
        load_machine_learning_model()

    assert "MODEL_S3_URI must point to a .tar.gz package" in str(excinfo.value)


@patch("app.utils.machine_learning.boto3.client")
@patch("app.utils.machine_learning.os.getenv")
def test_load_machine_learning_model_s3_error(mock_getenv, mock_boto_client):
    mock_getenv.side_effect = _fake_getenv(
        {
            "MODEL_S3_URI": "s3://my-bucket/model.tar.gz",
        }
    )

    mock_s3_instance = MagicMock()
    mock_boto_client.return_value = mock_s3_instance
    mock_s3_instance.get_object.side_effect = Exception("S3 Access Denied")

    with pytest.raises(Exception) as excinfo:
        load_machine_learning_model()

    assert "S3 Access Denied" in str(excinfo.value)
