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
    n = data.shape[0]
    data_left = np.array(data[data["x"] <= cutoff])
    data_right = np.array(data[data["x"] > cutoff])

    # Step 1.
    sigma_x = pd.DataFrame.var(data, axis=0)[0]
    h_ref = 1.84 * np.sqrt(sigma_x) * n ** (-1 / 5)

    data_left_in_bw = data_left[data_left[:, 0] >= cutoff - h_ref]
    data_right_in_bw = data_right[data_right[:, 0] <= cutoff + h_ref]
    n_left_in_bw = data_left_in_bw.shape[0]
    n_right_in_bw = data_right_in_bw.shape[0]
    y_ave_left_in_bw = np.sum(data_left_in_bw, axis=0)[1] / n_left_in_bw
    y_ave_right_in_bw = np.sum(data_right_in_bw, axis=0)[1] / n_right_in_bw

    f_hat = (n_left_in_bw + n_right_in_bw) / (n * h_ref)
    sigma_hat = (
        np.sum((data_left_in_bw[:, 1] - y_ave_left_in_bw) ** 2)
        + np.sum((data_right_in_bw[:, 1] - y_ave_right_in_bw) ** 2)
    ) / (n_left_in_bw + n_right_in_bw)

    # Step 2.
    median_x_left = np.median(data_left, axis=0)[0]
    median_x_right = np.median(data_right, axis=0)[0]
    bigger_than_median_left = data_left[data_left[:, 0] > median_x_left]
    smaller_than_median_right = data_right[data_right[:, 0] < median_x_right]
    data_temp = np.concatenate(
        (bigger_than_median_left, smaller_than_median_right), axis=0
    )
    x_temp = data_temp[:, 0]
    x_temp_minus_c = x_temp - cutoff
    y_temp = data_temp[:, 1]
    treatment_indicator = np.zeros(data_temp.shape[0])
    index = np.where(data_temp[:, 0] >= cutoff)
    treatment_indicator[index] = 1
    x_temp_powers = x_temp_minus_c[:, None] ** np.arange(4)
    x_temp_powers[:, 0] += treatment_indicator

    reg_results = np.linalg.lstsq(x_temp_powers, y_temp, rcond=None)
    m3_hat = 6 * reg_results[0][3]

    h_ref_left = (
        3.56
        * (sigma_hat / (f_hat * np.max([m3_hat ** 2, 0.01]))) ** (1 / 7)
        * n_left_in_bw
    )
    h_ref_right = (
        3.56
        * (sigma_hat / (f_hat * np.max([m3_hat ** 2, 0.01]))) ** (1 / 7)
        * n_right_in_bw
    )

    data_temp_left = data_left[data_left[:, 0] >= cutoff - h_ref_left]
    data_temp_right = data_right[data_right[:, 0] <= cutoff + h_ref_right]
    n_temp_left = data_temp_left.shape[0]
    n_temp_right = data_temp_right.shape[0]

    x_temp_left = data_temp_left[:, 0]
    x_temp_left_minus_c = x_temp_left - cutoff
    y_temp_left = data_temp_left[:, 1]
    x_temp_powers_left = x_temp_left_minus_c[:, None] ** np.arange(3)
    reg_results_left = np.linalg.lstsq(x_temp_powers_left, y_temp_left, rcond=None)
    m2_hat_left = 2 * reg_results_left[0][2]

    x_temp_right = data_temp_right[:, 0]
    x_temp_right_minus_c = x_temp_right - cutoff
    y_temp_right = data_temp_right[:, 1]
    x_temp_powers_right = x_temp_right_minus_c[:, None] ** np.arange(3)
    reg_results_right = np.linalg.lstsq(x_temp_powers_right, y_temp_right, rcond=None)
    m2_hat_right = 2 * reg_results_right[0][2]

    # Step 3.
    r_hat_left = 720 * sigma_hat / (n_temp_left * h_ref_left ** 4)
    r_hat_right = 720 * sigma_hat / (n_temp_right * h_ref_right ** 4)

    h_opt = (
        3.4375
        * (2 * sigma_hat)
        / (f_hat * ((m2_hat_right - m2_hat_left) ** 2 + r_hat_right + r_hat_left))
        ** (1 / 5)
        * n ** (-1 / 5)
    )

    return h_opt
