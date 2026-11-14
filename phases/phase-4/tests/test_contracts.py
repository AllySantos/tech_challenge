from pathlib import Path

import pandas as pd
import pytest

from credit_scoring_observability.contracts import validate_batch
from credit_scoring_observability.make_bad_batch import make_bad_batch
from credit_scoring_observability.prep import PRODUCTION_BASE_CSV, REFERENCE_CSV

needs_reference = pytest.mark.skipif(
    not Path(REFERENCE_CSV).exists(),
    reason="dados locais ausentes (rode data.py e prep.py)",
)
needs_production = pytest.mark.skipif(
    not Path(PRODUCTION_BASE_CSV).exists(),
    reason="dados locais ausentes (rode data.py e prep.py)",
)


@needs_reference
def test_reference_passes_contract():
    assert validate_batch(pd.read_csv(REFERENCE_CSV))


@needs_production
def test_bad_batch_is_blocked():
    assert not validate_batch(make_bad_batch())
