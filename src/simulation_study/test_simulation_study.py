import pytest
from simulation_study import fix_simulation_params
from simulation_study import simulate_estimator_performance


@pytest.fixture
def setup_fix_simulation_params():
    out = {}
    out["n"] = 1000
    out["M"] = 100
    out["model"] = "linear"
    out["distribution"] = "normal"
    out["discrete"] = False
    out["cutoff"] = 10
    out["tau"] = 10
    out["noise_var"] = 2

    return out


@pytest.fixture
def setup_simulate_estimator_performance():
    out = {}

    sim_params = {}
    sim_params["n"] = 1000
    sim_params["M"] = 100
    sim_params["model"] = "linear"
    sim_params["distribution"] = "normal"
    sim_params["discrete"] = False
    sim_params["cutoff"] = 10
    sim_params["tau"] = 10
    sim_params["noise_var"] = 2

    out["params"] = sim_params
    out["degree"] = 1
    out["parametric"] = True

    return out


def test_fix_simulation_params_n(setup_fix_simulation_params):
    with pytest.raises(TypeError):
        fix_simulation_params(
            n=100.1,
            M=setup_fix_simulation_params["M"],
            model=setup_fix_simulation_params["model"],
            distribution=setup_fix_simulation_params["distribution"],
            discrete=setup_fix_simulation_params["discrete"],
            cutoff=setup_fix_simulation_params["cutoff"],
            tau=setup_fix_simulation_params["tau"],
            noise_var=setup_fix_simulation_params["noise_var"],
        )


def test_fix_simulation_params_M(setup_fix_simulation_params):
    with pytest.raises(TypeError):
        fix_simulation_params(
            n=setup_fix_simulation_params["n"],
            M=(-1.5),
            model=setup_fix_simulation_params["model"],
            distribution=setup_fix_simulation_params["distribution"],
            discrete=setup_fix_simulation_params["discrete"],
            cutoff=setup_fix_simulation_params["cutoff"],
            tau=setup_fix_simulation_params["tau"],
            noise_var=setup_fix_simulation_params["noise_var"],
        )


def test_fix_simulation_params_model(setup_fix_simulation_params):
    with pytest.raises(ValueError):
        fix_simulation_params(
            n=setup_fix_simulation_params["n"],
            M=setup_fix_simulation_params["M"],
            model="Gaussian",
            distribution=setup_fix_simulation_params["distribution"],
            discrete=setup_fix_simulation_params["discrete"],
            cutoff=setup_fix_simulation_params["cutoff"],
            tau=setup_fix_simulation_params["tau"],
            noise_var=setup_fix_simulation_params["noise_var"],
        )


def test_fix_simulation_params_distribution(setup_fix_simulation_params):
    with pytest.raises(ValueError):
        fix_simulation_params(
            n=setup_fix_simulation_params["n"],
            M=setup_fix_simulation_params["M"],
            model=setup_fix_simulation_params["model"],
            distribution=100,
            discrete=setup_fix_simulation_params["discrete"],
            cutoff=setup_fix_simulation_params["cutoff"],
            tau=setup_fix_simulation_params["tau"],
            noise_var=setup_fix_simulation_params["noise_var"],
        )


def test_fix_simulation_params_discrete(setup_fix_simulation_params):
    with pytest.raises(TypeError):
        fix_simulation_params(
            n=setup_fix_simulation_params["n"],
            M=setup_fix_simulation_params["M"],
            model=setup_fix_simulation_params["model"],
            distribution=setup_fix_simulation_params["distribution"],
            discrete="Yes",
            cutoff=setup_fix_simulation_params["cutoff"],
            tau=setup_fix_simulation_params["tau"],
            noise_var=setup_fix_simulation_params["noise_var"],
        )


def test_simulate_estimator_performance_arguments(setup_simulate_estimator_performance):
    with pytest.raises(ValueError):
        simulate_estimator_performance(
            params=setup_simulate_estimator_performance["params"],
            degree=(-1.5),
            parametric=setup_simulate_estimator_performance["parametric"],
        )
    with pytest.raises(TypeError):
        simulate_estimator_performance(
            params=setup_simulate_estimator_performance["params"],
            degree=setup_simulate_estimator_performance["degree"],
            parametric="Yes",
        )
