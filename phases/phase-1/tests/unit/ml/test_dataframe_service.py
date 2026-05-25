import pandas as pd
import pytest

from ml.services.dataframe_service import DataFrameService


def test_load_dataframe_numeric_cleaning(tmp_path):
    csv_file = tmp_path / "data.csv"
    content = "customerID,TotalCharges\n1,10.5\n2, \n3,invalid"
    csv_file.write_text(content)

    service = DataFrameService()
    df = service.load_dataframe(str(csv_file.absolute()))

    assert df.loc[0, "TotalCharges"] == 10.5
    assert df.loc[1, "TotalCharges"] == 0.0
    assert df.loc[2, "TotalCharges"] == 0.0
    assert pd.api.types.is_float_dtype(df["TotalCharges"])


def test_load_dataframe_no_processing_needed(tmp_path):
    csv_file = tmp_path / "simple.csv"
    pd.DataFrame({"col_a": [1], "col_b": [2]}).to_csv(csv_file, index=False)

    service = DataFrameService()
    df = service.load_dataframe(str(csv_file.absolute()))

    assert list(df.columns) == ["col_a", "col_b"]
    assert df.iloc[0]["col_a"] == 1


def test_load_dataframe_raises_exception():
    service = DataFrameService()
    with pytest.raises(Exception):
        service.load_dataframe("invalid/path/to/nothing.csv")


def test_load_dataframe_handles_absolute_path(tmp_path):
    csv_file = tmp_path / "abs_test.csv"
    pd.DataFrame({"val": [100]}).to_csv(csv_file, index=False)

    service = DataFrameService()
    df = service.load_dataframe(str(csv_file.resolve()))

    assert df["val"].iloc[0] == 100
