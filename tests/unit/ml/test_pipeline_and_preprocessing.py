import numpy as np
import pandas as pd
import pytest
import torch

from ml.enums.dataset_type import DatasetType
from ml.pipeline.builder import PipelineBuilder
from ml.services.preprocessing_service import PreprocessingService
from ml.utils.architecture import EarlyStopping
from ml.utils.feature_identifier import FeatureIdentifier
from ml.utils.loaders import make_loader


def test_pipeline_builder_build_returns_working_pipeline():
    builder = PipelineBuilder()
    binary_features = ["gender"]
    categorical_features = ["contract"]

    pipeline = builder.build(binary_features, categorical_features)

    df = pd.DataFrame(
        {
            "gender": ["Male", "Female"],
            "contract": ["Month-to-month", "Two year"],
            "tenure": [1, 12],
        }
    )

    transformed = pipeline.fit_transform(df)

    assert transformed.shape[0] == 2
    assert transformed.shape[1] >= 1
    assert np.all(np.isfinite(transformed))


class FakeFeatureIdentifier:
    def get_features(self, df):
        return ["gender"], ["MultipleLines"], ["MonthlyCharges"]


def test_preprocessing_service_train_then_validation_runs_pipeline():
    service = PreprocessingService(
        feature_identifier=FakeFeatureIdentifier(), pipeline_builder=PipelineBuilder()
    )

    df_train = pd.DataFrame(
        {
            "customerID": ["0001", "0002"],
            "gender": ["Male", "Female"],
            "MultipleLines": ["No", "Yes"],
            "MonthlyCharges": [70.0, 99.0],
        }
    )

    transformed_train = service.run_pipeline(df_train, type=DatasetType.TRAIN)
    assert transformed_train.shape[0] == 2
    assert service.pipeline is not None

    df_val = pd.DataFrame(
        {
            "customerID": ["0003"],
            "gender": ["Female"],
            "MultipleLines": ["Yes"],
            "MonthlyCharges": [80.0],
        }
    )

    transformed_val = service.run_pipeline(df_val, type=DatasetType.VALIDATION)
    assert transformed_val.shape[0] == 1
    assert isinstance(transformed_val, pd.DataFrame)


def test_preprocessing_service_requires_train_before_validation():
    service = PreprocessingService(
        feature_identifier=FeatureIdentifier(), pipeline_builder=PipelineBuilder()
    )

    with pytest.raises(RuntimeError, match="Pipeline não foi ajustado"):
        service.run_pipeline(pd.DataFrame({"gender": ["Male"]}), type=DatasetType.TEST)


def test_make_loader_returns_expected_tensor_shapes():
    df = pd.DataFrame(np.arange(12).reshape(4, 3), columns=["a", "b", "c"])
    y = pd.Series([0, 1, 0, 1])

    loader = make_loader(df, y, batch_size=2)
    batch = next(iter(loader))

    assert batch[0].shape == (2, 3)
    assert batch[1].shape == (2, 1)
    assert batch[0].dtype == torch.float32
    assert batch[1].dtype == torch.float32


def test_early_stopping_triggers_after_patience():
    model = torch.nn.Linear(1, 1)
    early_stopping = EarlyStopping(patience=2)

    assert early_stopping.step(0.5, model) is False
    assert early_stopping.step(0.4, model) is False
    assert early_stopping.step(0.6, model) is False
    assert early_stopping.step(0.7, model) is True
    assert isinstance(early_stopping.best_model, dict)
