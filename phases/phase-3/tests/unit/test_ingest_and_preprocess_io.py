import pandas as pd
import pytest

from src.data.ingest import download, ingest
from src.data.preprocess import preprocess

CSV = "condition_label,medical_abstract\n" + "".join(
    f'{label},"{"clinical narrative " * 6}variant {i}"\n'
    for i, label in enumerate([1, 2, 3, 4, 5] * 6)
)


def test_download_reuses_an_existing_non_empty_file(tmp_path):
    destination = tmp_path / "already-there.csv"
    destination.write_text(CSV, encoding="utf-8")

    result = download("https://example.invalid/never-fetched.csv", destination)

    assert result == destination
    assert result.read_text(encoding="utf-8") == CSV


def test_download_fetches_when_the_file_is_missing(tmp_path, monkeypatch):
    calls = []

    def fake_retrieve(url, path):
        calls.append(url)
        Pathlike = type(path)  # noqa: N806 - urlretrieve aceita str ou Path
        assert Pathlike is not None
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(CSV)

    monkeypatch.setattr("urllib.request.urlretrieve", fake_retrieve)

    result = download("https://example.invalid/data.csv", tmp_path / "nested" / "data.csv")

    assert calls == ["https://example.invalid/data.csv"]
    assert result.exists()


def test_ingest_combines_both_splits_and_writes_the_labeled_csv(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "medical_tc_train.csv").write_text(CSV, encoding="utf-8")
    (raw_dir / "medical_tc_test.csv").write_text(CSV, encoding="utf-8")

    output = ingest(
        train_url="https://example.invalid/train.csv",
        test_url="https://example.invalid/test.csv",
        raw_dir=raw_dir,
        output_path=tmp_path / "labeled.csv",
    )

    df = pd.read_csv(output)
    assert len(df) == 60
    assert set(df.columns) == {"text", "urgency", "condition_label"}
    assert set(df["urgency"]) == {"normal", "atencao", "urgente"}


def test_preprocess_writes_both_splits(tmp_path):
    labeled = tmp_path / "labeled.csv"
    # O texto precisa ser único por linha: a limpeza descarta duplicatas.
    rows = [
        {"text": f"{'clinical narrative ' * 6} {urgency} variant {i}", "urgency": urgency}
        for i in range(20)
        for urgency in ("normal", "atencao", "urgente")
    ]
    pd.DataFrame(rows).to_csv(labeled, index=False)

    train_path, validation_path = preprocess(input_path=labeled, output_dir=tmp_path / "processed")

    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)

    assert len(train_df) + len(validation_df) == 60
    assert set(train_df["urgency"]) == {"normal", "atencao", "urgente"}


def test_preprocess_fails_loudly_on_a_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        preprocess(input_path=tmp_path / "absent.csv", output_dir=tmp_path)
