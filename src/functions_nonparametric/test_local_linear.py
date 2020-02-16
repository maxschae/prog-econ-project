import numpy as np
import pytest
from local_linear import y_hat_local_linear


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
    with pytest.raises(ValueError):
        y_hat_local_linear(
            x=setup_local_linear["x"],
            y=setup_local_linear["y"],
            x0=6.0,
            bandwidth=setup_local_linear["bandwidth"],
        )
