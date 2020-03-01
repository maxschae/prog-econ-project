import numpy as np
import statsmodels.api as sm


def estimate_treatment_effect_nonparametric(data, cutoff, bandwidth, alpha=0.05):
    """
    Estimate treatment effect non-parametrically with local linear regression
    using the boundary optimal triangle kernel and a specified bandwidth. Following
    Lee and Lemieux (2010), we estimate a pooled regression including a treatment
    indicator and an interaction term using weighted data within the bandwidth only.
    Center the running variable by subtracting the cutoff before estimation.

    Args:
        data (pd.DataFrame): Dataframe with data on the running variable in a
                            column called "r", data on the dependent variable
                            in a column called "y" and data on the treatment
                            status in a column called "d".
        cutoff (float): Cutpoint in the range of the running variable used to
                        distinguish between treatment and control groups.
        bandwidth (float): Bandwidth used in local linear regression.
        alpha (float): Significance level used to construct confidence intervals.
                        Default is 0.05.

    Returns:
        dict: Dictionary containing estimation results.
    """

    if bandwidth <= 0:
        raise ValueError("The specified bandwidth must be positive.")
    else:
        pass

    r = np.array(data["r"])
    y = np.array(data["y"])
    d = np.array(data["d"])

    # Compute weights with triangle kernel.
    data_points = np.abs(r - cutoff) / bandwidth
    weights = np.zeros_like(data_points)
    index_pos_weight = np.where(data_points <= 1)
    weights[index_pos_weight] = 1 - data_points[index_pos_weight]

    if np.all(weights == 0):
        raise ValueError("The Kernel does not include any data.")
    else:
        pass

    # Consider only datapoints with positive weight.
    r = r[np.where(weights > 0)]
    y = y[np.where(weights > 0)]
    d = d[np.where(weights > 0)]
    weights = weights[np.where(weights > 0)]
    sqrt_weights = np.sqrt(weights)

    # Perform local linear regression with centered running variable.
    r = r - cutoff
    r_powers = r[:, None] ** np.arange(2)
    r_powers_interact = r_powers[:, 1:] * d[:, None]
    regressors = np.column_stack((d, r_powers, r_powers_interact))
    regressors_weighted = regressors * sqrt_weights[:, None]
    y_weighted = y * sqrt_weights
    reg_results = sm.OLS(endog=y_weighted, exog=regressors_weighted).fit()

    # Store results in a dictionary.
    reg_out = {}
    reg_out["coef"] = reg_results.params[0]
    reg_out["se"] = reg_results.bse[0]
    reg_out["conf_int_lower"] = reg_results.conf_int(alpha=alpha)[0, 0]
    reg_out["conf_int_upper"] = reg_results.conf_int(alpha=alpha)[0, 1]

    return reg_out
