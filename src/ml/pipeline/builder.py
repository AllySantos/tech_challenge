from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.utils.encoders import CustomLabelEncoder


class PipelineBuilder:
    def __init__(self, label_encoder=None, one_hot_encoder=None, scaler=None):
        self.custom_label_encoder = label_encoder or CustomLabelEncoder()

        self.one_hot_encoder = one_hot_encoder or OneHotEncoder(
            drop="first",
            handle_unknown="ignore",
            sparse_output=False,
        )
        self.scaler = scaler or StandardScaler(with_mean=False)

    def build(self, binary_features, categorical_features) -> Pipeline:
        return Pipeline(
            [
                (
                    "preprocessor",
                    ColumnTransformer(
                        transformers=[
                            ("binary", self.custom_label_encoder, binary_features),
                            ("categorical", self.one_hot_encoder, categorical_features),
                        ],
                        remainder="passthrough",
                    ),
                ),
                ("scaler", self.scaler),
            ]
        )
