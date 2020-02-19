import numpy as np
import statsmodels.api as sm


def estimate_treatment_effect_parametric(data, degree=1):
    """Estimate parametric model and return treatment effect
    estimate using the package statsmodels. Allow varying
    coefficients on either side of cutoff.

    Args:
        data (pd.DataFrame): Dataframe holds regressand and regressors.
                             Column names must be 'y', 'd', 'r'.
        degree (int): Specify degree of polynomial model estimated.
                      Default is linear model, i.e. degree = 1.

    Returns reg_out (dict):
        coef (float): Coefficient of treatment indicator.
        conf_int (np.ndarray): Respective 95% confidence interval.
    """

    if {"y", "d", "r"}.issubset(data.columns) is False:
        raise IndexError("y, d or r not in index. Cannot run regression.")
    if (isinstance(degree, int) and degree >= 0) is False:
        raise TypeError("polynomial order must be weakly positive integer.")

    r = np.array(data["r"])
    d = np.array(data["d"])
    y = np.array(data["y"])

    # Construct running variable polynomials of flexible degree,
    # and interactions thereof with treatment indicator.
    r_polys = r[:, np.newaxis] ** np.arange(degree + 1)
    r_polys_interact = r_polys[:, 1:] * d[:, np.newaxis]
    X = np.column_stack((d, r_polys, r_polys_interact))

    reg_out = {}
    results = sm.OLS(y, X).fit()
    reg_out["coef"] = results.params[0]
    reg_out["conf_int"] = results.conf_int(alpha=0.05)[0, :]

    return reg_out
