import numpy as np
import pandas as pd
import pytest
from cross_validation import cross_validation
from cross_validation import y_hat_local_linear


@pytest.fixture
def setup_local_linear():
    out = {}
    out["x"] = np.array([2.0, 1.0, 1.75, 1.5, 3.0, 3.5, 4.0], dtype=np.float64)
    out["y"] = np.array([3.0, 4.0, 3.0, 3.5, 6.0, 6.0, 7.0], dtype=np.float64)
    out["x0"] = 1.4
    out["bandwidth"] = 1.0

    return out


@pytest.fixture
def expected_local_linear():
    out = {}
    out["y0_hat"] = 3.552056

    return out


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


def test_local_linear_y0_hat(setup_local_linear, expected_local_linear):
    calc_y0_hat = y_hat_local_linear(
        x=setup_local_linear["x"],
        y=setup_local_linear["y"],
        x0=setup_local_linear["x0"],
        bandwidth=setup_local_linear["bandwidth"],
    )
    assert np.isclose(calc_y0_hat, expected_local_linear["y0_hat"])


def test_local_linear_positive_bandwidth(setup_local_linear):
    with pytest.raises(ValueError):
        y_hat_local_linear(
            x=setup_local_linear["x"],
            y=setup_local_linear["y"],
            x0=setup_local_linear["x0"],
            bandwidth=-1.0,
        )


def test_local_linear_empty_kernel(setup_local_linear):
    calc_y0_hat = y_hat_local_linear(
        x=setup_local_linear["x"],
        y=setup_local_linear["y"],
        x0=6.0,
        bandwidth=setup_local_linear["bandwidth"],
    )
    assert np.isnan(calc_y0_hat)


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


def test_cross_validation_empty_kernel(setup_cross_validation):
    with pytest.raises(ValueError):
        cross_validation(
            data=setup_cross_validation["data"],
            cutoff=setup_cross_validation["cutoff"],
            h_grid=np.array([0.2]),
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
