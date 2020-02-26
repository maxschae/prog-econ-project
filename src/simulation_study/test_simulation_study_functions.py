import pytest

from src.simulation_study.simulation_study_functions import (
    simulate_estimator_performance,
)


@pytest.fixture
def setup_simulate_estimator_performance():
    out = {}

    sim_params = {}
    sim_params["n"] = 500
    sim_params["M"] = 100
    sim_params["model"] = "linear"
    sim_params["discrete"] = False
    sim_params["cutoff"] = 0
    sim_params["tau"] = 0.75
    sim_params["noise_var"] = 2

    out["params"] = sim_params
    out["degree"] = 1
    out["parametric"] = True
    out["bandwidth"] = "rot"

    return out


def test_simulate_estimator_performance_arguments(setup_simulate_estimator_performance):
    with pytest.raises(ValueError):
        simulate_estimator_performance(
            params=setup_simulate_estimator_performance["params"],
            degree=(-1.5),
            parametric=setup_simulate_estimator_performance["parametric"],
            bandwidth=setup_simulate_estimator_performance["bandwidth"],
        )
    with pytest.raises(ValueError):
        simulate_estimator_performance(
            params=setup_simulate_estimator_performance["params"],
            degree=setup_simulate_estimator_performance["degree"],
            parametric=False,
            bandwidth="yes",
        )
    with pytest.raises(TypeError):
        simulate_estimator_performance(
            params=setup_simulate_estimator_performance["params"],
            degree=setup_simulate_estimator_performance["degree"],
            parametric="Yes",
            bandwidth=setup_simulate_estimator_performance["bandwidth"],
        )
