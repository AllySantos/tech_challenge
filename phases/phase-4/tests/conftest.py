import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def valid_df():
    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "limit_bal": rng.integers(10_000, 500_000, n),
            "sex": rng.integers(1, 3, n),
            "education": rng.integers(1, 5, n),
            "marriage": rng.integers(1, 4, n),
            "age": rng.integers(25, 60, n),
            "default": rng.integers(0, 2, n),
        }
    )
    for i in range(1, 7):
        df[f"pay_{i}"] = rng.integers(-1, 3, n)
        df[f"bill_amt{i}"] = rng.integers(1_000, 100_000, n)
        df[f"pay_amt{i}"] = rng.integers(0, 10_000, n)
    return df
