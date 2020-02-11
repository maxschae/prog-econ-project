import numpy as np
import pandas as pd


def rule_of_thumb(data, cutoff):
    """ Calculate mean squared error optimal bandwidth to be used in local
        polynomial regression with a rule-of-thumb procedure developed by
        Fan and Gijbels (1996) and modified for the context of Regression
        Discontinuity Design by Imbens and Kalyanaraman (2009). The procedure
        is tailored for local polynomial regression using the boundary optimal
        triangle kernel.

    Args:
        data (pd.DataFrame): Dataframe with data on the running variable in the
                            first column and data on the dependent variable in
                            the second column.
        cutoff (float): Cutpoint in the range of the running variable used to
                        distinguish between treatment and control groups.

    Returns:
        h_opt (float): Mean squared error optimal rule-of-thumb bandwidth.
    """

    data.columns = ["x", "y"]
    data_left = np.array(data[data["x"] < cutoff])
    data_right = np.array(data[data["x"] >= cutoff])
    n = data.shape[0]
    n_left = data_left.shape[0]
    n_right = data_right.shape[0]

    if n_left == 0 or n_right == 0:
        raise ValueError(
            "The cutoff must lie within the range of the running variable."
        )
    else:
        pass

    # Step 1.
    sigma_x = pd.DataFrame.var(data, axis=0)[0]
    h_pilot = 1.84 * np.sqrt(sigma_x) * n ** (-1 / 5)

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

    f_hat = (n_left_in_h_pilot + n_right_in_h_pilot) / (n * h_pilot)

    if f_hat <= 0:
        raise ValueError("The computed density function estimate is not positive.")
    else:
        pass

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

    # Step 2.
    median_x_left = np.median(data_left, axis=0)[0]
    median_x_right = np.median(data_right, axis=0)[0]
    bigger_than_median_left = data_left[data_left[:, 0] > median_x_left]
    smaller_than_median_right = data_right[data_right[:, 0] < median_x_right]
    data_temp = np.concatenate(
        (bigger_than_median_left, smaller_than_median_right), axis=0
    )
    x_temp = data_temp[:, 0] - cutoff
    y_temp = data_temp[:, 1]

    x_temp_powers = x_temp[:, None] ** np.arange(4)
    treatment_indicator = np.zeros(data_temp.shape[0])
    treatment_indicator[np.where(data_temp[:, 0] >= cutoff)] = 1
    x_temp_powers[:, 0] += treatment_indicator

    reg_results = np.linalg.lstsq(a=x_temp_powers, b=y_temp, rcond=None)
    m3_hat = 6 * reg_results[0][3]

    h_ref_left = (
        3.56 * (sigma_hat / (f_hat * np.max([m3_hat ** 2, 0.01]))) ** (1 / 7) * n_left
    )
    h_ref_right = (
        3.56 * (sigma_hat / (f_hat * np.max([m3_hat ** 2, 0.01]))) ** (1 / 7) * n_right
    )

    if h_ref_left <= 0 or h_ref_right <= 0:
        raise ValueError("The computed reference bandwidth is not positive.")
    else:
        pass

    data_left_in_h_ref = data_left[data_left[:, 0] >= cutoff - h_ref_left]
    data_right_in_h_ref = data_right[data_right[:, 0] <= cutoff + h_ref_right]
    n_left_in_h_ref = data_left_in_h_ref.shape[0]
    n_right_in_h_ref = data_right_in_h_ref.shape[0]

    x_left_in_h_ref = data_left_in_h_ref[:, 0] - cutoff
    y_left_in_h_ref = data_left_in_h_ref[:, 1]
    x_powers_left_in_h_ref = x_left_in_h_ref[:, None] ** np.arange(3)
    reg_results_left = np.linalg.lstsq(
        x_powers_left_in_h_ref, y_left_in_h_ref, rcond=None
    )
    m2_hat_left = 2 * reg_results_left[0][2]

    x_right_in_h_ref = data_right_in_h_ref[:, 0] - cutoff
    y_right_in_h_ref = data_right_in_h_ref[:, 1]
    x_powers_right_in_h_ref = x_right_in_h_ref[:, None] ** np.arange(3)
    reg_results_right = np.linalg.lstsq(
        a=x_powers_right_in_h_ref, b=y_right_in_h_ref, rcond=None
    )
    m2_hat_right = 2 * reg_results_right[0][2]

    # Step 3.
    r_hat_left = 720 * sigma_hat / (n_left_in_h_ref * h_ref_left ** 4)
    r_hat_right = 720 * sigma_hat / (n_right_in_h_ref * h_ref_right ** 4)

    h_opt = (
        3.4375
        * (2 * sigma_hat)
        / (f_hat * ((m2_hat_right - m2_hat_left) ** 2 + r_hat_right + r_hat_left))
        ** (1 / 5)
        * n ** (-1 / 5)
    )

    return h_opt
