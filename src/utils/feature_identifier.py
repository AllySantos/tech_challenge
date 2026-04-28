import pandas as pd


class FeatureIdentifier:
    @staticmethod
    def get_features(df: pd.DataFrame) -> tuple:
        binary_features = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
        categorical_features = ["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"]
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        return binary_features, categorical_features, numeric_features

