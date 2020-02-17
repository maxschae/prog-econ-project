import numpy as np
import pandas as pd


def data_generating_process(params):
    """Implement data generating process for simulation study.
    Obtain artificial data on individual-level potential outcomes
    given a sharp regression discontinuity setup.

    Args:
        params (dict): Specifies simulation parameters.

    Returns:
        data (pd.DataFrame): Dataframe with data on (observed)
                           potential outcome, treatment status,
                           running variable.
    """

    # Obtain model parameters.
    model = params["model"]
    cutoff = params["cutoff"]
    tau = params["tau"]
    alpha = params["alpha"]
    beta_l, beta_r = params["beta_l"], params["beta_r"]
    noise_var = params["noise_var"]
    n = params["n"]

    # Check model parameters.
    if params["distribution"] not in ["normal", "uniform"]:
        raise ValueError("run. var. is drawn from 'normal' or 'uniform' only.")
    if isinstance(params["discrete"], bool) is False:
        raise TypeError("must be type boolean.")
    if params["model"] not in ["linear", "poly"]:
        raise ValueError("'model' takes 'linear' or 'poly' only.")

    data = pd.DataFrame()

    if params["distribution"] == "normal":
        # Draw running variable from Gaussian distribution.
        data["r"] = np.random.normal(loc=10, scale=5, size=n)
    elif params["distribution"] == "uniform":
        # Draw running variable from uniform distribution.
        data["r"] = np.random.uniform(low=0, high=20, size=n)

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

    elif model == "poly":
        # Obtain potential outcomes through 'poly' model.
        data["y"] = (
            alpha
            + tau * data["d"]
            + 1.1 * data["r"]
            + 2 * np.cos(data["r"])
            + np.random.normal(loc=0, scale=noise_var, size=n)
        )

    elif model == "nonparametric":
        pass
    else:
        pass

    if params["discrete"] is False:
        return data

    elif params["discrete"] is True:
        rmin = min(data["r"])
        binsize = 2 * np.std(data["r"]) * n ** (-1 / 2)

        # Calculate midpoint of lowest bin.
        binmp_lowest = (
            np.floor((rmin - cutoff) / binsize) * binsize + binsize / 2 + cutoff
        )

        # Assign each running variable observation its bin.
        data["binnum"] = round(
            (
                (
                    np.floor((data["r"] - cutoff) / binsize) * binsize
                    + binsize / 2
                    + cutoff
                )
                - binmp_lowest
            )
            / binsize
        )

        # Calculate mean of outcome and running variable for each discrete value.
        data_discrete = data.groupby(["binnum"], as_index=False).mean()

        # Mean of treatment indicator across bins must be zero or one.
        d_means = np.array(data_discrete["d"].dropna().drop_duplicates())
        if len(d_means) > 2:
            raise ValueError("a bin contains both treatment and control observations.")
        if np.array_equal(d_means, np.array([0, 1])) is False:
            raise ValueError("a bin contains both treatment and control observations.")

        return data_discrete

    else:
        return np.nan
