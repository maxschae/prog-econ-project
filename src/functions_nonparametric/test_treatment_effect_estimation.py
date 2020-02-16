import pandas as pd
import pytest
from treatment_effect_estimation import estimate_treatment_effect_nonparametric


@pytest.fixture
def setup_treatment_effect_estimation():
    out = {}
    out["data"] = pd.DataFrame(
        [
            [2.0, 3.0, 1.0],
            [1.0, 4.0, 0.0],
            [1.75, 3.0, 0.0],
            [1.5, 3.5, 0.0],
            [3.0, 6.0, 1.0],
            [3.5, 6.0, 1.0],
            [4.0, 7.0, 1.0],
        ],
        columns=["r", "y", "d"],
        index=range(7),
    )
    out["cutoff"] = 1.8
    out["bandwidth"] = 1.0
    out["alpha"] = 0.05

    return out


def test_estimate_treatment_effect_nonparametric_positive_bandwidth(
    setup_treatment_effect_estimation,
):
    with pytest.raises(ValueError):
        estimate_treatment_effect_nonparametric(
            data=setup_treatment_effect_estimation["data"],
            cutoff=setup_treatment_effect_estimation["cutoff"],
            bandwidth=-1.0,
            alpha=setup_treatment_effect_estimation["alpha"],
        )


def test_estimate_treatment_effect_nonparametric_empty_kernel(
    setup_treatment_effect_estimation,
):
    with pytest.raises(ValueError):
        estimate_treatment_effect_nonparametric(
            data=setup_treatment_effect_estimation["data"],
            cutoff=-5.0,
            bandwidth=setup_treatment_effect_estimation["bandwidth"],
            alpha=setup_treatment_effect_estimation["alpha"],
        )
