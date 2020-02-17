import pandas as pd
import pytest
from treatment_effect_estimation import estimate_treatment_effect_parametric


@pytest.fixture
def setup_treatment_effect_estimation():
    out = {}
    out["data"] = pd.DataFrame(
        {"y": [15, 20, 13.5, 7], "x": [7, 12.7, 11, 0.5], "d": [0, 1, 1, 0]}
    )
    out["data_clean"] = pd.DataFrame(
        {"y": [15, 20, 13.5, 7], "r": [7, 12.7, 11, 0.5], "d": [0, 1, 1, 0]}
    )
    out["degree"] = -1
    out["degree_clean"] = 1
    out["out"] = 1
    out["out_clean"] = "coef"

    return out


def test_treatment_effect_estimation_model_input(setup_treatment_effect_estimation):
    with pytest.raises(IndexError):
        estimate_treatment_effect_parametric(
            data=setup_treatment_effect_estimation["data"],
            degree=setup_treatment_effect_estimation["degree_clean"],
            out=setup_treatment_effect_estimation["out_clean"],
        )
    with pytest.raises(TypeError):
        estimate_treatment_effect_parametric(
            data=setup_treatment_effect_estimation["data_clean"],
            degree=setup_treatment_effect_estimation["degree"],
            out=setup_treatment_effect_estimation["out_clean"],
        )
    with pytest.raises(ValueError):
        estimate_treatment_effect_parametric(
            data=setup_treatment_effect_estimation["data_clean"],
            degree=setup_treatment_effect_estimation["degree_clean"],
            out=setup_treatment_effect_estimation["out"],
        )
