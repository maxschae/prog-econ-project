import pytest
from simulation_study import fix_simulation_params


@pytest.fixture
def setup_fix_simulation_params():
    out = {}
    out["n"] = 500
    out["M"] = 100
    out["model"] = "linear"
    out["discrete"] = False
    out["cutoff"] = 0
    out["tau"] = 0.75
    out["noise_var"] = 1

    return out


def test_fix_simulation_params_n(setup_fix_simulation_params):
    with pytest.raises(TypeError):
        fix_simulation_params(
            n=100.1,
            M=setup_fix_simulation_params["M"],
            model=setup_fix_simulation_params["model"],
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
            discrete="Yes",
            cutoff=setup_fix_simulation_params["cutoff"],
            tau=setup_fix_simulation_params["tau"],
            noise_var=setup_fix_simulation_params["noise_var"],
        )
