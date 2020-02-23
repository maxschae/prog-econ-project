import time

import numpy as np
import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.simulation_study.simulation_study_functions import (
    simulate_estimator_performance,
)


def fix_simulation_params(
    n=500,
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
    if model not in ["linear", "poly", "nonparametric"]:
        raise ValueError("'model' takes 'linear', 'poly' or 'nonparametric' only.")
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


if __name__ == "__main__":
    start = time.time()

    # Vary simulation along potential outcome models.
    for model in ["linear", "poly", "nonparametric"]:
        # Run simulation for continuous and discrete data.
        for discrete in [False, True]:
            sim_params = fix_simulation_params(model=model, discrete=discrete)

            # Estimate treatment effect parametrically and non-parametrically.
            for parametric in [True, False]:
                performance_measures = []

                if parametric is True:
                    # Estimate parametric model with different polyonomial degrees.
                    degrees = list(range(0, 10, 1))
                    for degree in degrees:
                        np.random.seed(123)
                        performance_measures.append(
                            simulate_estimator_performance(
                                params=sim_params, degree=degree, parametric=parametric,
                            )
                        )

                    # Convert dictionary to pd.DataFrame format to allow table construction.
                    df_performance_measures = pd.DataFrame.from_dict(
                        performance_measures
                    )
                    # Restrict interest to first four measures.
                    df_performance_measures = df_performance_measures.iloc[:, :4]
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
                    bandwidths = ["rot", "rot_under", "rot_over", "cv"]
                    for bandwidth in bandwidths:
                        np.random.seed(123)
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
                    # Restrict interest to first four measures.
                    df_performance_measures = df_performance_measures.iloc[:, :4]
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
