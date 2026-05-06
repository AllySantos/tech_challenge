import pickle

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

import ml.utils.loaders
from ml.utils.architecture import ChurnMLP
from ml.utils.loaders import load_model, load_scaler, make_loader


def test_make_loader_output_validation():
    X = pd.DataFrame({"feat1": [0.5, 1.5, 2.5], "feat2": [3.5, 4.5, 5.5]})
    y = pd.Series([1, 0, 1])
    batch_size = 2

    loader = make_loader(X, y, batch_size=batch_size, shuffle=False)

    assert isinstance(loader, DataLoader)
    assert len(loader.dataset) == 3

    batch_X, batch_y = next(iter(loader))
    assert batch_X.shape == (batch_size, 2)
    assert batch_y.shape == (batch_size, 1)
    assert batch_X.dtype == torch.float32
    assert batch_y.dtype == torch.float32


def test_load_scaler_integration(tmp_path):
    mock_data = {"mean": [10.0], "scale": [1.0]}
    file_name = "test_scaler.pkl"
    file_path = tmp_path / file_name

    with open(file_path, "wb") as f:
        pickle.dump(mock_data, f)

    original_dir = ml.utils.loaders.ARTIFACTS_DIR
    ml.utils.loaders.ARTIFACTS_DIR = tmp_path

    try:
        result = load_scaler(file_name)
        assert result == mock_data
    finally:
        ml.utils.loaders.ARTIFACTS_DIR = original_dir


def test_load_model_state_integration(tmp_path):
    input_dim = 4
    model = ChurnMLP(input_dim=input_dim)
    file_name = "test_checkpoint.pt"
    file_path = tmp_path / file_name

    torch.save(model.state_dict(), file_path)

    original_dir = ml.utils.loaders.ARTIFACTS_DIR
    ml.utils.loaders.ARTIFACTS_DIR = tmp_path

    try:
        loaded_model = load_model(input_dim=input_dim, checkpoint_name=file_name)
        assert isinstance(loaded_model, ChurnMLP)
        assert loaded_model.training is False
    finally:
        ml.utils.loaders.ARTIFACTS_DIR = original_dir


def test_load_scaler_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_scaler("missing_file.pkl")


def test_load_model_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_model(input_dim=5, checkpoint_name="missing_model.pt")
