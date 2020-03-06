import numpy as np
import pandas as pd


def rule_of_thumb(data, cutoff):
    """
    Calculate the mean squared error optimal bandwidth to be used in local
    linear regression with a rule-of-thumb procedure developed by
    Fan and Gijbels (1996) and modified for the context of Regression
    Discontinuity Design by Imbens and Kalyanaraman (2009). The procedure plugs
    parameter estimates from the data into a formula for the mean squared error
    optimal bandwidth and is tailored for local linear regression using the
    boundary optimal triangle kernel.

    Args:
        data (pd.DataFrame): Dataframe with data on the running variable in a
                            column called "r" and data on the dependent variable
                            in a column called "y".
        cutoff (float): Cutpoint in range of the running variable used to
                        distinguish between treatment and control groups.

    Returns:
        float: Mean squared error optimal rule-of-thumb bandwidth.
    """

    # Split data at the cutoff.
    data = data[["r", "y"]]
    data_left = np.array(data[data["r"] < cutoff])
    data_right = np.array(data[data["r"] >= cutoff])
    n = data.shape[0]
    n_left = data_left.shape[0]
    n_right = data_right.shape[0]

    if n_left == 0 or n_right == 0:
        raise ValueError("Cutoff must lie within range of the running variable.")
    else:
        pass

    # STEP 1: Estimation of density and conditional variance.
    # Compute bandwidth used for estimation of density and conditional variance
    # with Silverman's rule of thumb.
    h_1 = 1.84 * np.sqrt(pd.DataFrame.var(data, axis=0)[0]) * n ** (-1 / 5)

    if h_1 <= 0:
        raise ValueError("The computed bandwidth h_1 is not positive.")
    else:
        pass

    # Select data within the pilot bandwidth at the cutoff and compute mean of outcome variable.
    data_h_1_left = data_left[data_left[:, 0] >= cutoff - h_1]
    data_h_1_right = data_right[data_right[:, 0] <= cutoff + h_1]
    n_h_1_left = data_h_1_left.shape[0]
    n_h_1_right = data_h_1_right.shape[0]
    y_mean_h_1_left = np.sum(data_h_1_left, axis=0)[1] / n_h_1_left
    y_mean_h_1_right = np.sum(data_h_1_right, axis=0)[1] / n_h_1_right

    # Compute estimate of the running variable's density function at the cutoff.
    f_hat = (n_h_1_left + n_h_1_right) / (2 * n * h_1)

    if f_hat <= 0:
        raise ValueError("The computed density function estimate is not positive.")
    else:
        pass

    # Compute estimate of outcome given the running variable at the cutoff.
    sigma_hat = (
        np.sum((data_h_1_left[:, 1] - y_mean_h_1_left) ** 2)
        + np.sum((data_h_1_right[:, 1] - y_mean_h_1_right) ** 2)
    ) / (n_h_1_left + n_h_1_right)

    if sigma_hat <= 0:
        raise ValueError(
            "The computed conditional variance of the running variable is not positive."
        )
    else:
        pass

    # STEP 2: Estimation of second derivatives.
    # Temporarily discard observations left of median_r_left and right of median_r_right.
    median_r_left = np.median(data_left, axis=0)[0]
    median_r_right = np.median(data_right, axis=0)[0]
    bigger_than_median_left = data_left[data_left[:, 0] > median_r_left]
    smaller_than_median_right = data_right[data_right[:, 0] < median_r_right]
    data_temp = np.concatenate(
        (bigger_than_median_left, smaller_than_median_right), axis=0
    )
    r_temp = data_temp[:, 0] - cutoff
    y_temp = data_temp[:, 1]

    # Estimate third derivative of the regression function.
    r_temp_powers = r_temp[:, None] ** np.arange(4)
    treatment_indicator = np.zeros((data_temp.shape[0], 1))
    treatment_indicator[np.where(data_temp[:, 0] >= cutoff)] = 1
    r_temp_powers = np.concatenate((treatment_indicator, r_temp_powers), axis=1)
    reg_results = np.linalg.lstsq(a=r_temp_powers, b=y_temp, rcond=None)
    m3_hat = 6 * reg_results[0][4]

    # Compute bandwidths used for estimation of the regression functions' curvature.
    h_2_left = (
        3.56
        * (sigma_hat / (f_hat * np.max([m3_hat ** 2, 0.01]))) ** (1 / 7)
        * n_left ** (-1 / 7)
    )
    h_2_right = (
        3.56
        * (sigma_hat / (f_hat * np.max([m3_hat ** 2, 0.01]))) ** (1 / 7)
        * n_right ** (-1 / 7)
    )

    if h_2_left <= 0:
        raise ValueError("The computed bandwidth h_2_left is not positive.")
    if h_2_right <= 0:
        raise ValueError("The computed bandwidth h_2_right is not positive.")
    else:
        pass

    # Select data within the bandwidths at the cutoff.
    data_h_2_left = data_left[data_left[:, 0] >= cutoff - h_2_left]
    data_h_2_right = data_right[data_right[:, 0] <= cutoff + h_2_right]
    n_h_2_left = data_h_2_left.shape[0]
    n_h_2_right = data_h_2_right.shape[0]

    # Estimate the curvature of the regression function left of the cutoff.
    r_h_2_left = data_h_2_left[:, 0] - cutoff
    y_h_2_left = data_h_2_left[:, 1]
    r_powers_h_2_left = r_h_2_left[:, None] ** np.arange(3)
    reg_results_left = np.linalg.lstsq(r_powers_h_2_left, y_h_2_left, rcond=None)
    m2_hat_left = 2 * reg_results_left[0][2]

    # Estimate the curvature of the regression function right of the cutoff.
    r_h_2_right = data_h_2_right[:, 0] - cutoff
    y_h_2_right = data_h_2_right[:, 1]
    r_powers_h_2_right = r_h_2_right[:, None] ** np.arange(3)
    reg_results_right = np.linalg.lstsq(a=r_powers_h_2_right, b=y_h_2_right, rcond=None)
    m2_hat_right = 2 * reg_results_right[0][2]

    # STEP 3: Calculation of regularisation terms and optimal bandwidth.
    regul_term_left = 720 * sigma_hat / (n_h_2_left * h_2_left ** 4)
    regul_term_right = 720 * sigma_hat / (n_h_2_right * h_2_right ** 4)

    h_opt = (
        3.4375
        * (
            2
            * sigma_hat
            / (
                f_hat
                * (
                    (m2_hat_right - m2_hat_left) ** 2
                    + regul_term_right
                    + regul_term_left
                )
            )
        )
        ** (1 / 5)
        * n ** (-1 / 5)
    )

    return h_opt
