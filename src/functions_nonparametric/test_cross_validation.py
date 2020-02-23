import numpy as np
import pandas as pd
import pytest
from cross_validation import cross_validation


@pytest.fixture
def setup_cross_validation():
    out = {}
    out["data"] = pd.DataFrame(
        data=[
            [2.0, 3.0],
            [1.0, 4.0],
            [1.75, 3.0],
            [1.5, 3.5],
            [3.0, 6.0],
            [3.4, 5.0],
            [3.5, 6.0],
            [3.8, 5.5],
            [4.0, 7.0],
        ],
        columns=["r", "y"],
        index=range(9),
    )
    out["cutoff"] = 2.5
    out["h_grid"] = np.array([1.0, 0.5], dtype=float)
    out["min_num_obs"] = 2

    return out


def test_cross_validation_positive_h_grid(setup_cross_validation):
    with pytest.raises(ValueError):
        cross_validation(
            data=setup_cross_validation["data"],
            cutoff=setup_cross_validation["cutoff"],
            h_grid=np.array([-1.0, 1.0], dtype=float),
            min_num_obs=setup_cross_validation["min_num_obs"],
        )


def test_cross_validation_cutoff_within_range(setup_cross_validation):
    with pytest.raises(ValueError):
        cross_validation(
            data=setup_cross_validation["data"],
            cutoff=-3,
            h_grid=setup_cross_validation["h_grid"],
            min_num_obs=setup_cross_validation["min_num_obs"],
        )


def test_cross_validation_positive_h_opt(setup_cross_validation):
    calc_h_opt = cross_validation(
        data=setup_cross_validation["data"],
        cutoff=setup_cross_validation["cutoff"],
        h_grid=setup_cross_validation["h_grid"],
        min_num_obs=setup_cross_validation["min_num_obs"],
    )
    assert calc_h_opt > 0


def test_cross_validation_h_opt_in_h_grid(setup_cross_validation):
    calc_h_opt = cross_validation(
        data=setup_cross_validation["data"],
        cutoff=setup_cross_validation["cutoff"],
        h_grid=setup_cross_validation["h_grid"],
        min_num_obs=setup_cross_validation["min_num_obs"],
    )
    assert np.any(setup_cross_validation["h_grid"] == calc_h_opt)


def test_cross_validation_empty_kernel(setup_cross_validation):
    with pytest.raises(ValueError):
        cross_validation(
            data=setup_cross_validation["data"],
            cutoff=setup_cross_validation["cutoff"],
            h_grid=np.array([0.2]),
            min_num_obs=setup_cross_validation["min_num_obs"],
        )
