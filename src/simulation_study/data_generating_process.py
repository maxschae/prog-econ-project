import numpy as np
import pandas as pd


def data_generating_process(params):
    """
    Implementation of the data generating process in the simulation study.
    Obtain artificial data on individual-level variables given a sharp Regression
    Discontinuity setup for one of the following three model specifications:
    a linear relationship between outcome and running variable, a fourth-order
    polynomial one or a non-polynomial relationship.

    Args:
        params (dict): Dictionary holding the simulation parameters.

    Returns:
        pd.DataFrame: Dataframe with data on "r", "d" and "y" -
                the running variable, treatment status and observed outcome for
                each individual.
    """

    # Obtain model parameters.
    model = params["model"]
    cutoff = params["cutoff"]
    tau = params["tau"]
    noise_var = params["noise_var"]
    n = params["n"]

    data = pd.DataFrame()

    # Draw running variable from Gaussian distribution.
    data["r"] = np.random.normal(loc=0, scale=1, size=n)

    if cutoff < np.min(data["r"]) or cutoff > np.max(data["r"]):
        raise AssertionError("Cutoff out of bounds.")
    else:
        pass

    # Assign binary treatment status.
    data["d"] = 0
    data.loc[data["r"] >= cutoff, "d"] = 1

    if model == "linear":
        # Obtain potential outcomes through linear model.
        data["y"] = (
            10
            + tau * data["d"]
            + 1 * data["r"]
            + (1 + 0.5) * data["d"] * data["r"]
            + np.random.normal(loc=0, scale=noise_var, size=n)
        )

    elif model == "poly":
        # Obtain potential outcomes through 'poly' model.
        data["y"] = (
            2
            + tau * data["d"]
            + 0.8 * (data["r"])
            - 0.8 * (data["r"]) ** 2
            - 0.2 * (data["r"]) ** 3
            + 0.2 * (data["r"]) ** 4
            + np.random.normal(loc=0, scale=noise_var, size=n)
        )

    elif model == "nonpolynomial":
        # Obtain potential outcomes through 'nonparametric' model.
        data["y"] = (
            tau * data["d"]
            + data["r"] * np.sin(4 * data["r"]) * np.cos(data["r"] * data["d"])
            + (1 / (1 + data["r"] * data["d"])) * 1.5
            + np.random.normal(loc=0, scale=noise_var, size=n)
        )
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
            raise ValueError("A bin contains both treatment and control observations.")
        if np.array_equal(d_means, np.array([0, 1])) is False:
            raise ValueError("A bin contains both treatment and control observations.")
        else:
            pass

        return data_discrete

    else:
        return np.nan
