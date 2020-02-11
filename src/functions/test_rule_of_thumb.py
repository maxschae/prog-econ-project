import pandas as pd
import pytest
from rule_of_thumb import rule_of_thumb


@pytest.fixture
def setup_rule_of_thumb():
    out = {}
    out["data"] = pd.DataFrame(
        data=[
            [2, 3],
            [1, 4],
            [1.75, 3],
            [1.5, 3.5],
            [3, 6],
            [3.4, 5],
            [3.5, 6],
            [3.8, 5.5],
            [4, 7],
        ],
        columns=["run_var", "y"],
        index=range(9),
    )
    out["cutoff"] = 2.5

    return out


def test_rule_of_thumb_positive_h_opt(setup_rule_of_thumb):
    calc_h_opt = rule_of_thumb(
        data=setup_rule_of_thumb["data"], cutoff=setup_rule_of_thumb["cutoff"]
    )
    assert calc_h_opt > 0
