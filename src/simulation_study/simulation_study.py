import random
import time

import numpy as np
import pandas as pd
from data_generating_process import data_generating_process

from bld.project_paths import project_paths_join as ppj
from src.functions_nonparametric.cross_validation import cross_validation
from src.functions_nonparametric.rule_of_thumb import rule_of_thumb
from src.functions_nonparametric.treatment_effect_estimation import (
    estimate_treatment_effect_nonparametric,
)
from src.functions_parametric.treatment_effect_estimation import (
    estimate_treatment_effect_parametric,
)


def fix_simulation_params(
    n=1000,
    M=100,
    model="linear",
    distribution="normal",
    discrete=False,
    cutoff=0,
    tau=0.75,
    noise_var=1,
):
    """Initialize parameters for simulating potential outcome model.

    Args:
        n (int): Number of observations.
        M (int): Number of Monte Carlo repetions.
        model (str): Specify general functional form of model
                     that potential outcomes underlie.
        distribution (str): Specify running variable's distribution.
        discrete (bool): Specify if data is discretized or not.
        cutoff (float): RDD cutoff.
        noise_var (float): Variance of error term.

    Returns:
        sim_params (dict): Contains all parameters for simulation study
                           and data generating process.
    """

    # Check model parameters.
    if (isinstance(n, int) and isinstance(M, int)) is False:
        raise TypeError("'n' and 'M' must be integer.")
    if isinstance(n, int) is False:
        raise TypeError("'n' must be integer.")
    if model not in ["linear", "poly"]:
        raise ValueError("'model' takes 'linear' or 'poly' only.")
    if distribution not in ["normal", "uniform"]:
        raise ValueError("'distribution' must be 'normal' or 'uniform'.")
    if isinstance(discrete, bool) is False:
        raise TypeError("'discrete' must be type boolean.")

    sim_params = {}

    # Set number of Monte Carlo repetions.
    sim_params["M"] = M

    # Fix distribution of running variable, "normal" or "uniform".
    sim_params["distribution"] = distribution
    sim_params["n"] = n

    # Choose continuous or discrete data.
    sim_params["discrete"] = discrete

    # Fix parameters of model that describes potential outcomes.
    sim_params["model"] = model
    sim_params["cutoff"] = cutoff
    sim_params["tau"] = tau
    sim_params["noise_var"] = noise_var

    return sim_params


def simulate_estimator_performance(params, degree=1, parametric=True, bandwidth=None):
    """Collect performance measures on treatment effect estimator.
        Data stems from data generating process.

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
        bandwidth: Bandwidth used in local linear regression. Either 'cv' or'rot'
                   indicating use of cross-validation or rule-of-thumb bandwidth
                   selection procedure or a float directly specifying the bandwidth.

    Returns:
        performance_measure (dict): Holds coverage probability,
                                    mean, standard deviation and
                                    mean squared error of
                                    estimator across all M repetions.
    """

    tau_hats = []
    tau_in_conf_int = []

    if parametric is True:
        for _ in range(params["M"]):
            out_reg = estimate_treatment_effect_parametric(
                data=data_generating_process(params=params), degree=degree,
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
            data = data_generating_process(params=params)

            if bandwidth == "cv":
                # Specify largest bandwidth taken into consideration.
                h_pilot = rule_of_thumb(data, params["cutoff"])
                h = cross_validation(
                    data=data,
                    cutoff=params["cutoff"],
                    h_grid=np.linspace(start=0.5 * h_pilot, stop=2 * h_pilot, num=50),
                    min_num_obs=10,
                )
            elif bandwidth == "rot":
                h = rule_of_thumb(data, params["cutoff"])
            elif isinstance(bandwidth, (float, int)):
                h = bandwidth
            else:
                raise ValueError("Specified bandwidth is incorrect.")

            out_reg = estimate_treatment_effect_nonparametric(
                data=data, cutoff=params["cutoff"], bandwidth=h,
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
    performance_measure["tau_hat"] = np.mean(tau_hats)
    performance_measure["coverage_prob"] = np.mean(tau_in_conf_int)
    performance_measure["stdev_tau_hat"] = np.std(tau_hats)
    performance_measure["mse_tau_hat"] = np.square(
        np.subtract(tau_hats, params["tau"])
    ).mean()

    return performance_measure


if __name__ == "__main__":
    # Run simulation study.
    start = time.time()

    # Vary simulation along potential outcome models.
    for model in ["linear"]:
        # Run simulation for continuous and discrete data.
        for discrete in [False]:
            sim_params = fix_simulation_params(model=model, discrete=discrete)

            # Estimate treatment effect parametrically and non-parametrically.
            for parametric in [True, False]:
                performance_measures = []

                if parametric is True:
                    # Estimate parametric model with different polyonomial degrees.
                    degrees = list(range(0, 10, 1))
                    for degree in degrees:
                        random.seed(123)
                        performance_measures.append(
                            simulate_estimator_performance(
                                params=sim_params, degree=degree, parametric=parametric,
                            )
                        )

                    # Convert dictionary to pd.DataFrame format to allow table construction.
                    df_performance_measures = pd.DataFrame.from_dict(
                        performance_measures
                    )
                    df_performance_measures["degree"] = degrees

                    # Round all measures for representation purposes.
                    df_performance_measures = df_performance_measures.round(5)
                    # Place 'degree' in first column for representation purposes.
                    cols = df_performance_measures.columns.tolist()
                    cols = cols[-1:] + cols[:-1]
                    df_performance_measures = df_performance_measures[cols]
                    # Rename columns for LaTex table.
                    df_performance_measures = df_performance_measures.rename(
                        columns={
                            "coverage_prob": "Cov. Prob.",
                            "degree": "Polynomials",
                            "mse_tau_hat": "MSE",
                            "tau_hat": "Estimate",
                            "stdev_tau_hat": "Std. Dev.",
                        },
                    )

                    # Construct table from dataframe holding performance measures.
                    with open(
                        ppj(
                            "OUT_TABLES",
                            "simulation_study",
                            f"perf_meas_table_{model}_p_d_{discrete}.tex",
                        ),
                        "w",
                    ) as j:
                        j.write(df_performance_measures.to_latex(index=False))

                elif parametric is False:
                    # Estimate non-parametric model with different bandwidths.
                    bandwidths = ["rot", "cv"]
                    for bandwidth in bandwidths:
                        random.seed(123)
                        performance_measures.append(
                            simulate_estimator_performance(
                                params=sim_params,
                                parametric=parametric,
                                bandwidth=bandwidth,
                            )
                        )

                    # Convert dictionary to pd.DataFrame format to allow table construction.
                    df_performance_measures = pd.DataFrame.from_dict(
                        performance_measures
                    )
                    df_performance_measures["bandwidth"] = bandwidths

                    # Round all measures for representation purposes.
                    df_performance_measures = df_performance_measures.round(5)
                    # Place 'degree' in first column for representation purposes.
                    cols = df_performance_measures.columns.tolist()
                    cols = cols[-1:] + cols[:-1]
                    df_performance_measures = df_performance_measures[cols]
                    # Rename columns for LaTex table.
                    df_performance_measures = df_performance_measures.rename(
                        columns={
                            "coverage_prob": "Cov. Prob.",
                            "bandwidth": "Bandwidth",
                            "mse_tau_hat": "MSE",
                            "tau_hat": "Estimate",
                            "stdev_tau_hat": "Std. Dev.",
                        },
                    )

                    # Construct table from dataframe holding performance measures.
                    with open(
                        ppj(
                            "OUT_TABLES",
                            "simulation_study",
                            f"perf_meas_table_{model}_np_d_{discrete}.tex",
                        ),
                        "w",
                    ) as j:
                        j.write(df_performance_measures.to_latex(index=False))

    end = time.time()
    print("Run took {:5.3f}s.".format(end - start))
