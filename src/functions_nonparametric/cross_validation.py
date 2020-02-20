import numpy as np

from src.functions_nonparametric.local_linear import y_hat_local_linear


def cross_validation(data, cutoff, h_grid, min_num_obs):
    """ Perform leave-one-out cross-validation to select the mean squared error
        optimal bandwidth used in local linear regression out of a given grid.
        The procedure is tailored to the RDD context and follows the ideas of
        Ludwig and Miller (2005) and Imbens and Lemieux (2008).

    Args:
        data (pd.DataFrame): Dataframe with data on the running variable in a
                            column called "r" and data on the dependent variable
                            in a column called "y".
        cutoff (float): Cutpoint in the range of the running variable used to
                        distinguish between treatment and control groups.
        h_grid (np.array): Grid of bandwidths taken into consideration.
        min_num_obs (float): Minimum number of observations used for fitting the
                            data at a particular point.

    Returns:
        h_opt (float): Mean squared error optimal bandwidth out of h_grid.
    """

    if np.any(h_grid <= 0):
        raise ValueError("All bandwidths must be positive.")
    else:
        pass

    data = data[["r", "y"]]
    data_left = np.array(data[data["r"] <= cutoff])
    data_right = np.array(data[data["r"] > cutoff])

    if data_left.size == 0 or data_right.size == 0:
        raise ValueError("Cutoff must lie within range of the running variable.")
    else:
        pass

    mean_squared_errors = np.zeros_like(h_grid)

    for h_index, h in enumerate(h_grid):
        intermediate_res = 0

        for r_index, r_point in enumerate(data_left[:, 0]):
            training_data = np.delete(data_left, r_index, axis=0)
            training_data = training_data[training_data[:, 0] <= r_point]
            if training_data.shape[0] >= min_num_obs:
                y_hat = y_hat_local_linear(
                    x=training_data[:, 0],
                    y=training_data[:, 1],
                    x0=r_point,
                    bandwidth=h,
                )
                intermediate_res += (data_left[r_index, 1] - y_hat) ** 2
            else:
                pass

        for r_index, r_point in enumerate(data_right[:, 0]):
            training_data = np.delete(data_right, r_index, axis=0)
            training_data = training_data[training_data[:, 0] >= r_point]
            if training_data.shape[0] >= min_num_obs:
                y_hat = y_hat_local_linear(
                    x=training_data[:, 0],
                    y=training_data[:, 1],
                    x0=r_point,
                    bandwidth=h,
                )
                intermediate_res += (data_right[r_index, 1] - y_hat) ** 2
            else:
                pass

        mean_squared_errors[h_index] = intermediate_res / data.shape[0]

    h_opt = h_grid[np.argmin(mean_squared_errors)]

    return h_opt
