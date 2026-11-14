import numpy as np

from credit_scoring_observability.simulate_drift import apply_drift


def drift(df, **kw):
    params = dict(inflation=0.0, delay_prob=0.0, younger_years=0, concept_rate=0.0)
    params.update(kw)
    return apply_drift(df.copy(), rng=np.random.default_rng(42), **params)


def test_zero_drift_changes_nothing(valid_df):
    out = drift(valid_df)
    for col in ["age", "bill_amt1", "pay_1"]:
        assert np.allclose(out[col], valid_df[col])


def test_inflation_raises_bills(valid_df):
    out = drift(valid_df, inflation=0.4)
    assert out["bill_amt1"].mean() > valid_df["bill_amt1"].mean()


def test_delays_raise_late_payments(valid_df):
    out = drift(valid_df, delay_prob=0.3)
    assert out["pay_1"].mean() > valid_df["pay_1"].mean()


def test_younger_lowers_mean_age(valid_df):
    out = drift(valid_df, younger_years=10)
    assert out["age"].mean() < valid_df["age"].mean()


def test_same_seed_is_reproducible(valid_df):
    a = drift(valid_df, inflation=0.2, delay_prob=0.1)
    b = drift(valid_df, inflation=0.2, delay_prob=0.1)
    assert a.equals(b)
