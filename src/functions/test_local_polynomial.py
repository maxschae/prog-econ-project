import numpy as np
import pytest
from local_polynomial import y_hat_local_polynomial


@pytest.fixture
def setup_local_polynomial():
    out = {}
    out["x"] = np.array([2, 1, 1.75, 1.5, 3, 3.5, 4], dtype=np.float64)
    out["y"] = np.array([3, 4, 3, 3.5, 6, 6, 7], dtype=np.float64)
    out["x0"] = 1.4
    out["degree"] = 1
    out["bandwidth"] = 1

    return out


@pytest.fixture
def expected_local_polynomial():
    out = {}
    out["y0_hat"] = 3.552056

    return out


def test_local_polynomial_y0_hat(setup_local_polynomial, expected_local_polynomial):
    calc_y0_hat = y_hat_local_polynomial(
        x=setup_local_polynomial["x"],
        y=setup_local_polynomial["y"],
        x0=setup_local_polynomial["x0"],
        degree=setup_local_polynomial["degree"],
        bandwidth=setup_local_polynomial["bandwidth"],
    )
    assert np.isclose(calc_y0_hat, expected_local_polynomial["y0_hat"])


def test_local_polynomial_positive_bandwidth(
    setup_local_polynomial, expected_local_polynomial
):
    with pytest.raises(ValueError):
        y_hat_local_polynomial(
            x=setup_local_polynomial["x"],
            y=setup_local_polynomial["y"],
            x0=setup_local_polynomial["x0"],
            degree=setup_local_polynomial["degree"],
            bandwidth=-1,
        )


def test_local_polynomial_positive_degree(
    setup_local_polynomial, expected_local_polynomial
):
    with pytest.raises(ValueError):
        y_hat_local_polynomial(
            x=setup_local_polynomial["x"],
            y=setup_local_polynomial["y"],
            x0=setup_local_polynomial["x0"],
            degree=-2,
            bandwidth=setup_local_polynomial["bandwidth"],
        )
