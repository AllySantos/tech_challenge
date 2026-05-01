import pandas as pd

from ml.enums.dataset_type import DatasetType
from ml.pipeline.builder import PipelineBuilder
from ml.utils.feature_identifier import FeatureIdentifier


class PreprocessingService:
    def __init__(self, feature_identifier: FeatureIdentifier, pipeline_builder: PipelineBuilder):
        self.feature_identifier = feature_identifier
        self.pipeline_builder = pipeline_builder
        self.pipeline = None

    def run_pipeline(
        self, df: pd.DataFrame, type: DatasetType = DatasetType.TRAIN, target=None
    ) -> pd.DataFrame:
        df = self.__remove_non_predictable_features(df)

        if type == DatasetType.TRAIN:
            # fit_transform APENAS no treino — ajusta os encoders e o scaler
            # com as estatísticas do treino e já transforma.

            binary_features, categorical_features, _ = self.feature_identifier.get_features(df)
            self.pipeline = self.pipeline_builder.build(binary_features, categorical_features)
            transformed = self.pipeline.fit_transform(df)

        else:
            if self.pipeline is None:
                raise RuntimeError(
                    "Pipeline não foi ajustado. "
                    "Chame run_pipeline com DatasetType.TRAIN antes de val/test."
                )
            transformed = self.pipeline.transform(df)

        return pd.DataFrame(transformed, index=df.index)

    def __remove_non_predictable_features(self, df: pd.DataFrame, features=None) -> pd.DataFrame:
        if features is None:
            features = ["customerID"]

        return df.drop(columns=features, errors="ignore")
