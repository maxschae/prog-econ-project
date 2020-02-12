import pytest
from data_generating_process import data_generating_process


@pytest.fixture
def setup_data_generating_process():
    out = {}
    out["n"] = 100.1
    out["model"] = "lin"

    return out


def test_data_generating_process_arguments(setup_data_generating_process):
    with pytest.raises(ValueError):
        data_generating_process(
            n=setup_data_generating_process["n"],
            model=setup_data_generating_process["model"],
        )
