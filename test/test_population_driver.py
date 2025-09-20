import pandas as pd
import numpy as np
from cbc_population_distributions.population_driver import _get_map_sample

class DummyResult:
    def __init__(self, posterior):
        self.posterior = posterior

def test_get_map_sample_ml():
    # Prior constant -> should select ML
    df = pd.DataFrame({
        "log_likelihood": [0.1, 2.0, 1.0],
        "log_prior": [0.0, 0.0, 0.0],
    })
    res = DummyResult(df)
    sample = _get_map_sample(res)
    assert sample["log_likelihood"] == 2.0

def test_get_map_sample_map():
    # Prior informative -> should select MAP (ll + lp)
    df = pd.DataFrame({
        "log_likelihood": [1.0, 1.0, 1.0],
        "log_prior": [0.1, 5.0, 0.2],
    })
    res = DummyResult(df)
    sample = _get_map_sample(res)
    assert sample["log_prior"] == 5.0
