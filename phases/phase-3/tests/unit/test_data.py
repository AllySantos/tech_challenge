import pandas as pd
import pytest

from src.configs.settings import CONDITION_TO_URGENCY, URGENCY_LABELS
from src.data.ingest import map_urgency
from src.data.preprocess import clean_dataframe, clean_text, split_dataset


def test_urgency_mapping_covers_every_source_class():
    assert set(CONDITION_TO_URGENCY) == {1, 2, 3, 4, 5}
    assert set(CONDITION_TO_URGENCY.values()) == set(URGENCY_LABELS)


def test_map_urgency_translates_condition_labels():
    df = pd.DataFrame(
        {
            "condition_label": [4, 1, 5],
            "medical_abstract": ["cardiac arrest", "tumor resection", "routine screening"],
        }
    )

    mapped = map_urgency(df)

    assert mapped["urgency"].tolist() == ["urgente", "atencao", "normal"]
    assert list(mapped.columns) == ["text", "urgency", "condition_label"]


def test_map_urgency_rejects_unmapped_class():
    df = pd.DataFrame({"condition_label": [9], "medical_abstract": ["unknown"]})

    with pytest.raises(ValueError, match="sem mapeamento"):
        map_urgency(df)


def test_map_urgency_rejects_missing_columns():
    with pytest.raises(ValueError, match="Colunas ausentes"):
        map_urgency(pd.DataFrame({"condition_label": [1]}))


def test_clean_text_collapses_whitespace_and_lowercases():
    assert clean_text("  Acute\n\tChest   PAIN  ") == "acute chest pain"


def test_clean_text_preserves_clinical_punctuation():
    assert clean_text("Troponin 10.5 mg/dL, BP 120/80") == "troponin 10.5 mg/dl, bp 120/80"


def test_clean_dataframe_drops_short_and_duplicated_records():
    df = pd.DataFrame(
        {
            "text": ["short", "a" * 60, "a" * 60, "b" * 60],
            "urgency": ["normal", "urgente", "urgente", "atencao"],
        }
    )

    cleaned = clean_dataframe(df, min_chars=50)

    assert len(cleaned) == 2
    assert cleaned["text"].is_unique


def test_clean_dataframe_drops_unknown_urgency_labels():
    df = pd.DataFrame({"text": ["a" * 60, "b" * 60], "urgency": ["urgente", "critico"]})

    cleaned = clean_dataframe(df, min_chars=50)

    assert cleaned["urgency"].tolist() == ["urgente"]


def test_split_dataset_preserves_class_proportions():
    df = pd.DataFrame(
        {
            "text": [f"laudo numero {i}" for i in range(300)],
            "urgency": ["normal", "atencao", "urgente"] * 100,
        }
    )

    train_df, val_df = split_dataset(df, validation_size=0.2, seed=42)

    assert len(train_df) == 240
    assert len(val_df) == 60
    assert val_df["urgency"].value_counts().to_dict() == {
        "normal": 20,
        "atencao": 20,
        "urgente": 20,
    }
