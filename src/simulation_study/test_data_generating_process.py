import pandas as pd
import pytest
from data_generating_process import data_generating_process


@pytest.fixture
def setup_data_generating_process():
    out = {}

    out["n"] = 500
    out["M"] = 100
    out["model"] = "linear"
    out["discrete"] = False
    out["cutoff"] = 0
    out["tau"] = 0.75
    out["noise_var"] = 1

    out = {"out": out}

    return out


def test_data_generating_process_return_val(setup_data_generating_process):
    data = data_generating_process(params=setup_data_generating_process["out"])
    assert isinstance(data, pd.DataFrame)
