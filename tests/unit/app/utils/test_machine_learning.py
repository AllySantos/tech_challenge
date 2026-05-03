from unittest.mock import MagicMock, patch

import pytest

from app.utils.machine_learning import load_machine_learning_model


@patch("app.utils.machine_learning.os.getenv")
def test_load_machine_learning_model_no_env_var(mock_getenv):
    mock_getenv.return_value = None

    result = load_machine_learning_model()

    assert result is None
    mock_getenv.assert_called_once_with("MODEL_S3_URI")


@patch("app.utils.machine_learning.torch.load")
@patch("app.utils.machine_learning.boto3.client")
@patch("app.utils.machine_learning.os.getenv")
def test_load_machine_learning_model_success(mock_getenv, mock_boto_client, mock_torch_load):
    mock_getenv.return_value = "s3://my-bucket/models/model.pth"

    mock_s3_instance = MagicMock()
    mock_boto_client.return_value = mock_s3_instance

    mock_body = MagicMock()
    mock_body.read.return_value = b"fake-model-data"
    mock_s3_instance.get_object.return_value = {"Body": mock_body}

    mock_model_object = MagicMock()
    mock_torch_load.return_value = mock_model_object

    result = load_machine_learning_model()

    assert result == mock_model_object
    mock_boto_client.assert_called_once_with("s3")
    mock_s3_instance.get_object.assert_called_once_with(Bucket="my-bucket", Key="models/model.pth")
    mock_torch_load.assert_called_once()
    args, kwargs = mock_torch_load.call_args
    assert kwargs["map_location"] == "cpu"


@patch("app.utils.machine_learning.boto3.client")
@patch("app.utils.machine_learning.os.getenv")
def test_load_machine_learning_model_s3_error(mock_getenv, mock_boto_client):
    mock_getenv.return_value = "s3://my-bucket/model.pth"

    mock_s3_instance = MagicMock()
    mock_boto_client.return_value = mock_s3_instance
    mock_s3_instance.get_object.side_effect = Exception("S3 Access Denied")

    with pytest.raises(Exception) as excinfo:
        load_machine_learning_model()

    assert "S3 Access Denied" in str(excinfo.value)


@patch("app.utils.machine_learning.torch.load")
@patch("app.utils.machine_learning.boto3.client")
@patch("app.utils.machine_learning.os.getenv")
def test_load_machine_learning_model_corrupt_data(mock_getenv, mock_boto_client, mock_torch_load):
    mock_getenv.return_value = "s3://my-bucket/model.pth"

    mock_s3_instance = MagicMock()
    mock_boto_client.return_value = mock_s3_instance
    mock_body = MagicMock()
    mock_body.read.return_value = b"corrupt-data"
    mock_s3_instance.get_object.return_value = {"Body": mock_body}

    mock_torch_load.side_effect = RuntimeError("Invalid file format")

    with pytest.raises(RuntimeError) as excinfo:
        load_machine_learning_model()

    assert "Invalid file format" in str(excinfo.value)
