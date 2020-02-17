import random
import time

import numpy as np
import pandas as pd
from data_generating_process import data_generating_process

from bld.project_paths import project_paths_join as ppj
from src.functions_nonparametric.rule_of_thumb import rule_of_thumb
from src.functions_nonparametric.treatment_effect_estimation import (
    estimate_treatment_effect_nonparametric,
)
from src.functions_parametric.treatment_effect_estimation import (
    estimate_treatment_effect_parametric,
)


def fix_simulation_params(
    model="linear", distribution="normal", discrete=False, cutoff=10,
):
    """Initialize parameters for simulating potential outcome model.

    Args:
        model (str): Specify general functional form of model
                     that potential outcomes underlie.
        distribution (str): Specify running variable's distribution.
        discrete (bool): Specify if data is discretized or not.
        cutoff (float): RDD cutoff.

    Returns:
        sim_params (dict): Contains all parameters for simulation study
                           and data generating process.
    """
    sim_params = {}

    # Set number of Monte Carlo repitions.
    sim_params["M"] = 100

    # Fix distribution of running variable, "normal" or "uniform".
    sim_params["distribution"] = distribution
    sim_params["n"] = 1000

    # Choose continuous or discrete data.
    sim_params["discrete"] = discrete

    # Fix parameters of model that describes potential outcomes.
    sim_params["model"] = model
    sim_params["cutoff"] = cutoff
    sim_params["tau"] = 10
    sim_params["alpha"] = 10
    sim_params["beta_l"] = 1
    sim_params["beta_r"] = 0.5
    sim_params["noise_var"] = 2

    return sim_params


def simulate_estimator_performance(params, degree=1, parametric=True):
    """Collect performance measures on treatment effect estimator
    based on parametric OLS. Data stems from data generating process.

    Args:
        params (dict): Contains all parameters for simulating
                       artificial data in a regression discontinuity
                       setting.
        degree (int): Specifies degree of polynomial model fitted to
                      data. Degree of '0' corresponds to comparison
                      in means, '1' to the linear model. Default is 1.
        parametric (bool): Specify whether treatment effect is estimated
                           using 'parametric' or 'non-parametric' methods.
                           Defaults to 'parametric' of degree 1.

    Returns:
        performance_measure (dict): Holds coverage probability,
                                    mean, standard deviation and
                                    mean squared error of
                                    estimator across all M repitions.
    """

    tau_hats = []
    tau_in_conf_int = []

    if parametric is True:
        for _ in range(params["M"]):
            out_reg = estimate_treatment_effect_parametric(
                data=data_generating_process(params=sim_params), degree=degree,
            )
            tau_hat = out_reg["coef"]
            ci_lower, ci_upper = out_reg["conf_int"]

            # Collect estimates for subsequent investigation.
            tau_hats.append(tau_hat)

            # TODO
            if np.isnan(tau_hat):
                raise ValueError("Estimate is nan.")

            if ci_lower <= params["tau"] and params["tau"] <= ci_upper:
                # Count if true value falls into its estimate's confidence region.
                tau_in_conf_int.append(1)
            else:
                tau_in_conf_int.append(0)

    elif parametric is False:
        for _ in range(params["M"]):
            data = data_generating_process(params=sim_params)
            out_reg = estimate_treatment_effect_nonparametric(
                data=data,
                cutoff=sim_params["cutoff"],
                bandwidth=rule_of_thumb(data, sim_params["cutoff"]),
            )
            tau_hat = out_reg["coef"]
            ci_lower, ci_upper = out_reg["conf_int"]
            tau_hats.append(tau_hat)
            if ci_lower <= params["tau"] and params["tau"] <= ci_upper:
                tau_in_conf_int.append(1)
            else:
                tau_in_conf_int.append(0)
    else:
        raise TypeError("argument 'parametric' must be boolean.")

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

for model in ["linear", "poly"]:
    sim_params = fix_simulation_params(model=model, discrete=False,)

    for parametric in [True, False]:
        performance_measures = []

        if parametric is True:
            degrees = list(range(0, 10, 1))

            for degree in degrees:
                performance_measures.append(
                    simulate_estimator_performance(
                        params=sim_params, degree=degree, parametric=parametric
                    )
                )

            # Convert dictionary to pd.DataFrame format to allow table construction.
            df_performance_measures = pd.DataFrame.from_dict(performance_measures)
            df_performance_measures["degree"] = degrees

            f = open(
                ppj("OUT_FIGURES", f"perform_meas_table_{model}_parametric.tex"), "w"
            )
            # Construct table from dataframe holding performance measures.
            f.write(df_performance_measures.to_latex(index=False))
            f.close()

        elif parametric is False:
            performance_measures.append(
                simulate_estimator_performance(params=sim_params, parametric=parametric)
            )

            df_performance_measures = pd.DataFrame.from_dict(performance_measures)
            f = open(
                ppj("OUT_FIGURES", f"perform_meas_table_{model}_nonparametric.tex"), "w"
            )
            f.write(df_performance_measures.to_latex(index=False))
            f.close()


end = time.time()
print("Run took {:5.3f}s.".format(end - start))
