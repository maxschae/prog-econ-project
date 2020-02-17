import pytest
from data_generating_process import data_generating_process


@pytest.fixture
def setup_data_generating_process_params_distr():
    my_dict = {}
    my_dict["M"] = 100
    my_dict["n"] = 1000
    my_dict["distribution"] = "Gaussian"
    my_dict["discrete"] = False
    my_dict["model"] = "linear"
    my_dict["cutoff"] = 0
    my_dict["tau"] = 10
    my_dict["alpha"] = 10
    my_dict["beta_l"] = 1
    my_dict["beta_r"] = 0.5
    my_dict["noise_var"] = 3
    out = {"my_dict": my_dict}

    return out


@pytest.fixture
def setup_data_generating_process_params_discr():
    my_dict = {}
    my_dict["M"] = 100
    my_dict["n"] = 1000
    my_dict["distribution"] = "normal"
    my_dict["discrete"] = 1
    my_dict["model"] = "linear"
    my_dict["cutoff"] = 0
    my_dict["tau"] = 10
    my_dict["alpha"] = 10
    my_dict["beta_l"] = 1
    my_dict["beta_r"] = 0.5
    my_dict["noise_var"] = 3
    out = {"my_dict": my_dict}

    return out


def test_data_generating_process_params_distr(
    setup_data_generating_process_params_distr,
):
    with pytest.raises(ValueError):
        data_generating_process(
            params=setup_data_generating_process_params_distr["my_dict"]
        )


def test_data_generating_process_params_discr(
    setup_data_generating_process_params_discr,
):
    with pytest.raises(TypeError):
        data_generating_process(
            params=setup_data_generating_process_params_discr["my_dict"]
        )
