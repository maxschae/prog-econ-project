import pandas as pd
import pytest
from treatment_effect_estimation import estimate_treatment_effect_nonparametric


@pytest.fixture
def setup_treatment_effect_estimation():
    out = {}
    out["data"] = pd.DataFrame(
        [
            [2, 3, 1],
            [1, 4, 0],
            [1.75, 3, 0],
            [1.5, 3.5, 0],
            [3, 6, 1],
            [3.5, 6, 1],
            [4, 7, 1],
        ],
        columns=["r", "y", "d"],
        index=range(7),
    )
    out["cutoff"] = 1.8
    out["bandwidth"] = 1
    out["degree"] = 1
    out["alpha"] = 0.05

    return out


def test_estimate_treatment_effect_nonparametric_positive_bandwidth(
    setup_treatment_effect_estimation,
):
    with pytest.raises(ValueError):
        estimate_treatment_effect_nonparametric(
            data=setup_treatment_effect_estimation["data"],
            cutoff=setup_treatment_effect_estimation["cutoff"],
            bandwidth=-1,
            degree=setup_treatment_effect_estimation["degree"],
            alpha=setup_treatment_effect_estimation["alpha"],
        )


def test_estimate_treatment_effect_nonparametric_positive_degree(
    setup_treatment_effect_estimation,
):
    with pytest.raises(ValueError):
        estimate_treatment_effect_nonparametric(
            data=setup_treatment_effect_estimation["data"],
            cutoff=setup_treatment_effect_estimation["cutoff"],
            bandwidth=setup_treatment_effect_estimation["bandwidth"],
            degree=-2,
            alpha=setup_treatment_effect_estimation["alpha"],
        )


def test_estimate_treatment_effect_nonparametric_empty_kernel(
    setup_treatment_effect_estimation,
):
    with pytest.raises(ValueError):
        estimate_treatment_effect_nonparametric(
            data=setup_treatment_effect_estimation["data"],
            cutoff=-5,
            bandwidth=setup_treatment_effect_estimation["bandwidth"],
            degree=setup_treatment_effect_estimation["degree"],
            alpha=setup_treatment_effect_estimation["alpha"],
        )
