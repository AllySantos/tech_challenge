from credit_scoring_observability.contracts import validate_batch


def test_valid_batch_passes(valid_df):
    assert validate_batch(valid_df) is True


def test_negative_age_fails(valid_df):
    valid_df.loc[0, "age"] = -5
    assert validate_batch(valid_df) is False


def test_invalid_sex_fails(valid_df):
    valid_df.loc[0, "sex"] = 7
    assert validate_batch(valid_df) is False


def test_duplicate_ids_fail(valid_df):
    valid_df.loc[1, "id"] = valid_df.loc[0, "id"]
    assert validate_batch(valid_df) is False


def test_extra_column_fails_because_schema_is_strict(valid_df):
    valid_df["unexpected"] = 1
    assert validate_batch(valid_df) is False
