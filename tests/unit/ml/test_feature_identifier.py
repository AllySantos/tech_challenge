import pandas as pd

from ml.utils.feature_identifier import FeatureIdentifier


def test_get_features_standard_input():
    df = pd.DataFrame(
        {
            "gender": ["Male"],
            "Partner": ["Yes"],
            "tenure": [10],
            "MonthlyCharges": [50.0],
            "Contract": ["Month-to-month"],
            "SeniorCitizen": [0],
        }
    )

    bin_feat, cat_feat, num_feat = FeatureIdentifier.get_features(df)

    assert "gender" in bin_feat
    assert "Partner" in bin_feat
    assert "Contract" in cat_feat
    assert "tenure" in num_feat
    assert "MonthlyCharges" in num_feat
    assert "SeniorCitizen" in num_feat
    assert len(num_feat) == 3


def test_get_features_no_numeric_columns():
    df = pd.DataFrame({"gender": ["Female"], "Contract": ["One year"]})

    bin_feat, cat_feat, num_feat = FeatureIdentifier.get_features(df)

    assert len(num_feat) == 0
    assert "gender" in bin_feat
    assert "Contract" in cat_feat


def test_get_features_numeric_types_identification():
    df = pd.DataFrame(
        {"int_col": [1, 2, 3], "float_col": [1.1, 2.2, 3.3], "str_col": ["a", "b", "c"]}
    )

    _, _, num_feat = FeatureIdentifier.get_features(df)

    assert "int_col" in num_feat
    assert "float_col" in num_feat
    assert "str_col" not in num_feat


def test_get_features_returns_tuple_of_lists():
    df = pd.DataFrame({"dummy": [1]})
    result = FeatureIdentifier.get_features(df)

    assert isinstance(result, tuple)
    assert len(result) == 3
    assert all(isinstance(x, list) for x in result)


def test_get_features_contains_all_defined_binary_keys():
    df = pd.DataFrame({"a": [1]})
    bin_feat, _, _ = FeatureIdentifier.get_features(df)

    expected = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    assert bin_feat == expected


def test_get_features_contains_all_defined_categorical_keys():
    df = pd.DataFrame({"a": [1]})
    _, cat_feat, _ = FeatureIdentifier.get_features(df)

    expected = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
    ]
    assert cat_feat == expected
