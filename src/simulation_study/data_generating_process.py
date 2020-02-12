import numpy as np
import pandas as pd


def data_generating_process(params, model="linear"):
    """Implement data generating process for simulation study.
    Obtain artificial data on individual-level potential outcomes
    given a sharp regression discontinuity setup.

    Args:
        params (dict): Specifies simulation parameters.
        model (str): Specifies basic relationship between outcome
                     and running variable. Options are 'linear',
                     'polynomial', 'nonparametric'. Default is 'linear'.

    Returns:
        data (pd.DataFrame): Dataframe with data on (observed)
                           potential outcome, treatmend status,
                           running variable.
    """

    # Check function arguments.
    if model not in ["linear", "polynomial", "nonparametric"]:
        raise ValueError(
            "model (argument) must be 'linear', 'polynomial', or 'nonparametric'."
        )

    n = params["n"]
    if isinstance(n, int) is False:
        raise ValueError("n (argument) must be integer.")

    # Obtain model parameters.
    cutoff = params["cutoff"]
    tau = params["tau"]
    alpha = params["alpha"]
    beta_l = params["beta_l"]
    beta_r = params["beta_r"]
    noise_var = params["noise_var"]

    data = pd.DataFrame()

    if params["distribution"] == "normal":
        # Draw running variable from Gaussian distribution.
        data["r"] = np.random.normal(loc=0, scale=5, size=n)
    elif params["distribution"] == "uniform":
        # Draw running variable from uniform distribution.
        data["r"] = np.random.uniform(low=-10, high=10, size=n)

    if cutoff < np.min(data["r"]) or cutoff > np.max(data["r"]):
        raise ValueError("cutoff out of bounds.")

    # Assign binary treatment status.
    data["d"] = 0
    data.loc[data["r"] >= cutoff, "d"] = 1

    if model == "linear":
        # Obtain potential outcomes through linear model.
        data["y"] = (
            alpha
            + tau * data["d"]
            + beta_l * data["r"]
            + (beta_l + beta_r) * data["d"] * data["r"]
            + np.random.normal(loc=0, scale=noise_var, size=n)
        )

    elif model == "polynomial":
        pass

    elif model == "nonparametric":
        pass

    return data
