import pandas as pd
import pytest
from data_generating_process import data_generating_process


@pytest.fixture
def setup_data_generating_process():
    out = {}

    out["n"] = 1000
    out["M"] = 100
    out["model"] = "linear"
    out["distribution"] = "normal"
    out["discrete"] = False
    out["cutoff"] = 10
    out["tau"] = 10
    out["noise_var"] = 2

    out = {"out": out}

    return out


def test_data_generating_process_return_val(setup_data_generating_process):
    data = data_generating_process(params=setup_data_generating_process["out"])
    assert isinstance(data, pd.DataFrame)
