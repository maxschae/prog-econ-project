import random
import time

import numpy as np
import pandas as pd
from data_generating_process import data_generating_process

from bld.project_paths import project_paths_join as ppj
from src.functions_parametric.treatment_effect_estimation import (
    estimate_treatment_effect_parametric,
)


def fix_simulation_params(model="linear"):
    """Initialize parameters for simulating potential outcome model.

    Args:
        model (str): Specify general functional form of model
                     that potential outcomes underlie.

    Returns:
        sim_params (dict): Contains all parameters for simulation study
                           and data generating process.
    """
    sim_params = {}

    # Set number of Monte Carlo repitions.
    sim_params["M"] = 100

    # Fix distribution of running variable, "normal" or "uniform".
    sim_params["distribution"] = "normal"
    sim_params["n"] = 1000

    # Choose continuous or discrete data.
    sim_params["discrete"] = False

    # Fix parameters of model that describes potential outcomes.
    sim_params["model"] = model
    sim_params["cutoff"] = 0
    sim_params["tau"] = 10
    sim_params["alpha"] = 10
    sim_params["beta_l"] = 1
    sim_params["beta_r"] = 0.5
    sim_params["noise_var"] = 3

    return sim_params


def simulate_estimator_performance(params, degree=1):
    """Collect performance measures on treatment effect estimator
    based on parametric OLS. Data stems from data generating process.

    Args:
        params (dict): Contains all parameters for simulating
                       artificial data in a regression discontinuity
                       setting.
        degree (int): Specifies degree of polynomial model fitted to
                      data. Degree of '0' corresponds to comparison
                      in means, '1' to the linear model. Default is 1.

    Returns:
        performance_measure (dict): Holds coverage probability,
                                    mean, standard deviation and
                                    mean squared error of
                                    estimator across all M repitions.
    """

    tau_hats = []
    tau_in_conf_int = []

    for _ in range(params["M"]):
        out_reg = estimate_treatment_effect_parametric(
            data=data_generating_process(params=sim_params), degree=degree,
        )
        tau_hat = out_reg["coef"]
        ci_lower, ci_upper = out_reg["coef_ci"]

        # Collect estimates for subsequent investigation.
        tau_hats.append(tau_hat)

        if ci_lower < params["tau"] and params["tau"] < ci_upper:
            # Count if true value falls into its estimate's confidence region.
            tau_in_conf_int.append(1)
        else:
            tau_in_conf_int.append(0)

    performance_measure = {}
    performance_measure["coverage_prob"] = np.mean(tau_in_conf_int)
    performance_measure["tau_hat"] = np.mean(tau_hats)
    performance_measure["stdev_tau_hat"] = np.std(tau_hats)
    performance_measure["mse_tau_hat"] = np.square(
        np.subtract(tau_hats, params["tau"])
    ).mean()

    return performance_measure


# Run simulation study.
random.seed(1234)

start = time.time()
model = "linear"
sim_params = fix_simulation_params(model=model)

performance_measures = []
degrees = list(range(0, 10, 1))
for degree in degrees:
    performance_measures.append(
        simulate_estimator_performance(params=sim_params, degree=degree,)
    )

df_performance_measures = pd.DataFrame.from_dict(performance_measures)
df_performance_measures["degree"] = degrees

# TODO Move to "OUT_TABLES", instead of "OUT_FIGURES"
# f = open(ppj("OUT_TABLES", "perform_meas_table_{:}.tex".format(model)), "w")
f = open(ppj("OUT_FIGURES", f"perform_meas_table_{model}.tex"), "w")
f.write(df_performance_measures.to_latex(index=False))
f.close()

end = time.time()
print("Run took {:5.3f}s.".format(end - start))
