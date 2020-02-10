import numpy as np
from local_polynomial import y_hat_local_polynomial


def cross_validation(data, cutoff, h_grid, degree):
    """ Perform leave-one-out cross-validation to select the mean squared error
        optimal bandwidth used in local polynomial regression out of a given grid.
        The procedure is tailored to the RDD context and follows the ideas of
        Ludwig and Miller (2005) and Imbens and Lemieux (2008).

    Args:
        data (pd.DataFrame): Dataframe with data on the running variable in the
                            first column and data on the dependent variable in
                            the second column.
        cutoff (float): Cutpoint in the range of the running variable used to
                        distinguish between treatment and control groups.
        h_grid (np.array): Grid of bandwidths taken into consideration.
        degree (float): Degree of polynomial used for local polynomial regression.

    Returns:
        h_opt (float): Mean squared error optimal bandwidth out of h_grid.
    """

    if np.any(h_grid <= 0):
        raise ValueError("All bandwidths must be positive.")
    else:
        pass

    data.columns = ["run_var", "y"]
    data.sort_values("run_var", axis=0, ascending=True, inplace=True)
    data_left = np.array(data[data["run_var"] <= cutoff])
    data_right = np.array(data[data["run_var"] > cutoff])

    if data_left.size == 0 or data_right.size == 0:
        raise ValueError(
            "The cutoff must lie within the range of the running variable."
        )
    else:
        pass

    mean_squared_errors = np.zeros_like(h_grid)

    for h_index, h in enumerate(h_grid):
        intermediate_res = 0

        for x_index, x_point in enumerate(data_left[2:, 0]):
            orig_index = x_index + 2
            training_data = data_left[:orig_index, :]
            y_hat = y_hat_local_polynomial(
                x=training_data[:, 0],
                y=training_data[:, 1],
                x0=x_point,
                degree=degree,
                bandwidth=h,
            )
            intermediate_res += (data_left[orig_index, 1] - y_hat) ** 2

        for x_index, x_point in enumerate(data_right[: len(data_right[:, 0]) - 2, 0]):
            training_data = data_right[x_index + 1 :, :]
            y_hat = y_hat_local_polynomial(
                x=training_data[:, 0],
                y=training_data[:, 1],
                x0=x_point,
                degree=degree,
                bandwidth=h,
            )
            intermediate_res += (data_right[x_index, 1] - y_hat) ** 2

        mean_squared_errors[h_index] = intermediate_res / data.shape[0]

    h_opt = h_grid[np.argmin(mean_squared_errors)]

    return h_opt
