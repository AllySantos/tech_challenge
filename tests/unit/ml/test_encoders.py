import pytest

from ml.utils.encoders import CustomLabelEncoder, CustomOneHotEncoder


def test_custom_label_encoder_transforms_binary_values():
    encoder = CustomLabelEncoder()

    values = [
        ["Male", "Yes"],
        ["female", "no"],
    ]

    transformed = encoder.transform(values)

    assert transformed.shape == (2, 2)
    assert transformed.tolist() == [[1, 1], [0, 0]]


def test_custom_label_encoder_raises_on_unrecognized_value():
    encoder = CustomLabelEncoder()

    with pytest.raises(ValueError, match="Valor 'maybe' nao reconhecido como binario"):
        encoder.transform(["maybe"])


def test_custom_one_hot_encoder_fit_transform_and_feature_names():
    encoder = CustomOneHotEncoder(drop_first=True)
    data = [["red"], ["blue"], ["red"], ["green"]]

    encoder.fit(data)
    assert encoder.categories_ == ["green", "red"]
    assert encoder.get_feature_names_out(["color"]).tolist() == ["color_green", "color_red"]

    transformed = encoder.transform(data)
    assert transformed.shape == (4, 2)
    assert transformed.tolist() == [[0, 1], [0, 0], [0, 1], [1, 0]]


def test_custom_one_hot_encoder_rejects_unseen_category():
    encoder = CustomOneHotEncoder(drop_first=False)
    encoder.fit(["a", "b"])

    with pytest.raises(ValueError, match="Valor 'c' nao visto no fit"):
        encoder.transform(["c"])
