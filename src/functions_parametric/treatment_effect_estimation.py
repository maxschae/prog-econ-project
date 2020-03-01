import numpy as np
import statsmodels.api as sm


def estimate_treatment_effect_parametric(data, cutoff, degree=1, alpha=0.05):
    """
    Estimate treatment effect parametrically with global polynomial fitting of a
    specified degree. Allow varying coefficients on either side of the cutoff and
    center the running variable by subtracting the cutoff before estimation.

    Args:
        data (pd.DataFrame): Dataframe with data on the running variable in a
                            column called "r", data on the dependent variable
                            in a column called "y" and data on the treatment
                            status in a column called "d".
        cutoff (float): Cutpoint in the range of the running variable used to
                        distinguish between treatment and control groups.
        degree (int): Degree of polynomial used for fitting. Default is linear model.
        alpha (float): Significance level used to construct confidence intervals.
                        Default is 0.05.

    Returns:
        dict: Dictionary containing estimation results.
    """

    if {"y", "d", "r"}.issubset(data.columns) is False:
        raise IndexError("'y', 'd' or 'r' not in index.")
    if (isinstance(degree, int) and degree >= 0) is False:
        raise ValueError("polynomial order must be weakly positive integer.")
    else:
        pass

    r = np.array(data["r"])
    d = np.array(data["d"])
    y = np.array(data["y"])

    # Construct running variable polynomials of flexible degree,
    # and interactions thereof with treatment indicator.
    # Center running variable by subtracting cutoff.
    r_polys = (r[:, np.newaxis] - cutoff) ** np.arange(degree + 1)
    r_polys_interact = r_polys[:, 1:] * d[:, np.newaxis]
    X = np.column_stack((d, r_polys, r_polys_interact))

    reg_out = {}
    results = sm.OLS(y, X).fit()
    reg_out["coef"] = results.params[0]
    reg_out["se"] = results.bse[0]
    reg_out["conf_int_lower"] = results.conf_int(alpha=alpha)[0, 0]
    reg_out["conf_int_upper"] = results.conf_int(alpha=alpha)[0, 1]

    return reg_out
