import numpy as np
import statsmodels.api as sm


def estimate_treatment_effect_parametric(data, cutoff, degree=1, alpha=0.05):
    """Estimate parametric model and return treatment effect
    estimate using the package statsmodels. Allow varying
    coefficients on either side of cutoff.

    Args:
        data (pd.DataFrame): Dataframe holds regressand and regressors.
                             Column names must be 'y', 'd', 'r'.
        cutoff (float): RDD cutoff.
        degree (int): Specify degree of polynomial model estimated.
                      Default is linear model, i.e. degree = 1.

    Returns reg_out (dict):
        coef (float): Coefficient of treatment indicator.
        conf_int (np.ndarray): Respective 95% confidence interval.
    """

    if {"y", "d", "r"}.issubset(data.columns) is False:
        raise IndexError("'y', 'd' or 'r' not in index.")
    if (isinstance(degree, int) and degree >= 0) is False:
        raise ValueError("polynomial order must be weakly positive integer.")

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
