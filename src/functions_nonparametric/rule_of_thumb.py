import numpy as np
import pandas as pd


def rule_of_thumb(data, cutoff):
    """ Calculate mean squared error optimal bandwidth to be used in local
        linear regression with a rule-of-thumb procedure developed by
        Fan and Gijbels (1996) and modified for the context of Regression
        Discontinuity Design by Imbens and Kalyanaraman (2009). The procedure
        is tailored for local linear regression using the boundary optimal
        triangle kernel.

    Args:
        data (pd.DataFrame): Dataframe with data on the running variable in a
                            column called "r" and data on the dependent variable
                            in a column called "y".
        cutoff (float): Cutpoint in range of the running variable used to
                        distinguish between treatment and control groups.

    Returns:
        h_opt (float): Mean squared error optimal rule-of-thumb bandwidth.
    """

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

    # Step 1: Estimation of density and conditional variance.
    sigma_r = pd.DataFrame.var(data, axis=0)[0]
    h_pilot = 1.84 * np.sqrt(sigma_r) * n ** (-1 / 5)

    if h_pilot <= 0:
        raise ValueError("The computed pilot bandwidth is not positive.")
    else:
        pass

    data_left_in_h_pilot = data_left[data_left[:, 0] >= cutoff - h_pilot]
    data_right_in_h_pilot = data_right[data_right[:, 0] <= cutoff + h_pilot]
    n_left_in_h_pilot = data_left_in_h_pilot.shape[0]
    n_right_in_h_pilot = data_right_in_h_pilot.shape[0]
    y_mean_left_in_h_pilot = np.sum(data_left_in_h_pilot, axis=0)[1] / n_left_in_h_pilot
    y_mean_right_in_h_pilot = (
        np.sum(data_right_in_h_pilot, axis=0)[1] / n_right_in_h_pilot
    )

    # Compute density estimate.
    f_hat = (n_left_in_h_pilot + n_right_in_h_pilot) / (2 * n * h_pilot)

    if f_hat <= 0:
        raise ValueError("The computed density function estimate is not positive.")
    else:
        pass

    # Compute conditional variance.
    sigma_hat = (
        np.sum((data_left_in_h_pilot[:, 1] - y_mean_left_in_h_pilot) ** 2)
        + np.sum((data_right_in_h_pilot[:, 1] - y_mean_right_in_h_pilot) ** 2)
    ) / (n_left_in_h_pilot + n_right_in_h_pilot)

    if sigma_hat <= 0:
        raise ValueError(
            "The computed conditional variance of the running variable is not positive."
        )
    else:
        pass

    # Step 2: Estimation of second derivatives.
    median_r_left = np.median(data_left, axis=0)[0]
    median_r_right = np.median(data_right, axis=0)[0]
    bigger_than_median_left = data_left[data_left[:, 0] > median_r_left]
    smaller_than_median_right = data_right[data_right[:, 0] < median_r_right]
    data_temp = np.concatenate(
        (bigger_than_median_left, smaller_than_median_right), axis=0
    )
    r_temp = data_temp[:, 0] - cutoff
    y_temp = data_temp[:, 1]

    r_temp_powers = r_temp[:, None] ** np.arange(4)
    treatment_indicator = np.zeros((data_temp.shape[0], 1))
    treatment_indicator[np.where(data_temp[:, 0] >= cutoff)] = 1
    r_temp_powers = np.concatenate((treatment_indicator, r_temp_powers), axis=1)

    reg_results = np.linalg.lstsq(a=r_temp_powers, b=y_temp, rcond=None)
    m3_hat = 6 * reg_results[0][4]

    h_ref_left = (
        3.56
        * (sigma_hat / (f_hat * np.max([m3_hat ** 2, 0.01]))) ** (1 / 7)
        * n_left ** (-1 / 7)
    )
    h_ref_right = (
        3.56
        * (sigma_hat / (f_hat * np.max([m3_hat ** 2, 0.01]))) ** (1 / 7)
        * n_right ** (-1 / 7)
    )

    if h_ref_left <= 0 or h_ref_right <= 0:
        raise ValueError("The computed reference bandwidth is not positive.")
    else:
        pass

    data_left_in_h_ref = data_left[data_left[:, 0] >= cutoff - h_ref_left]
    data_right_in_h_ref = data_right[data_right[:, 0] <= cutoff + h_ref_right]
    n_left_in_h_ref = data_left_in_h_ref.shape[0]
    n_right_in_h_ref = data_right_in_h_ref.shape[0]

    r_left_in_h_ref = data_left_in_h_ref[:, 0] - cutoff
    y_left_in_h_ref = data_left_in_h_ref[:, 1]
    r_powers_left_in_h_ref = r_left_in_h_ref[:, None] ** np.arange(3)
    reg_results_left = np.linalg.lstsq(
        r_powers_left_in_h_ref, y_left_in_h_ref, rcond=None
    )
    m2_hat_left = 2 * reg_results_left[0][2]

    r_right_in_h_ref = data_right_in_h_ref[:, 0] - cutoff
    y_right_in_h_ref = data_right_in_h_ref[:, 1]
    r_powers_right_in_h_ref = r_right_in_h_ref[:, None] ** np.arange(3)
    reg_results_right = np.linalg.lstsq(
        a=r_powers_right_in_h_ref, b=y_right_in_h_ref, rcond=None
    )
    m2_hat_right = 2 * reg_results_right[0][2]

    # Step 3: Calculation of regularisation terms and optimal bandwidth.
    regul_term_left = 720 * sigma_hat / (n_left_in_h_ref * h_ref_left ** 4)
    regul_term_right = 720 * sigma_hat / (n_right_in_h_ref * h_ref_right ** 4)

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
