import numpy as np
import statsmodels.api as sm


def estimate_treatment_effect_nonparametric(data, cutoff, bandwidth, alpha=0.05):
    """Estimate treatment effect non-parametrically with local linear
        regression using the triangle kernel and a specified bandwidth.

    Args:
        data (pd.DataFrame): Dataframe with data on the running variable in a
                            column called "r", data on the dependent variable
                            in a column called "y" and data on the treatment
                            status in a column called "d".
        cutoff (float): Cutpoint in the range of the running variable used to
                        distinguish between treatment and control groups.
        bandwidth (float): Bandwidth used in local linear regression.
        alpha (float): Significance level used to construct the confidence interval.

    Returns:
        reg_out (dict):
            coef (float): Treatment effect estimate.
            se (float): Standard error of the treatment effect estimate.
            conf_int (np.array): (1-alpha)-confidence interval of the treatment
                                    effect estimate.
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

    # Perform weighted least squares regression with centered running variable.
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
    reg_out["se"] = reg_results.normalized_cov_params[0, 0]
    reg_out["conf_int"] = reg_results.conf_int(alpha=alpha)[0, :]

    return reg_out
