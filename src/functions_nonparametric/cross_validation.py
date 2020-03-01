import numba
import numpy as np


@numba.jit(nopython=True)
def y_hat_local_linear(x, y, x0, bandwidth):
    """
    Perform local linear regression with the triangle kernel and a specified
    bandwidth to predict the value of the dependent variable at some point x0.
    The function is used in cross-validation to predict the value of the outcome
    variable at the hold-out observation.

    Args:
        x (np.array): Array of type np.float64 containing regressor values used
                    for regression.
        y (np.array): Array of type np.float64 containing dependent variable used
                    for regression.
        x0 (float): Regressor value at which value of dependent variable is predicted.
        bandwidth (float): Range of data the kernel uses to assign weights.

    Returns:
        float: Predicted value of the dependent variable at x0.
    """

    if bandwidth <= 0:
        raise ValueError("The specified bandwidth must be positive.")
    else:
        pass

    # Compute weights determined by triangle kernel.
    data_points = np.abs(x - x0) / bandwidth
    weights = np.zeros_like(data_points)
    index_pos_weight = np.where(data_points <= 1)
    weights[index_pos_weight] = 1 - data_points[index_pos_weight]

    # Consider only datapoints with positive weight to reduce computation time.
    x = x[np.where(weights > 0)]
    y = y[np.where(weights > 0)]
    weights = weights[np.where(weights > 0)]
    sqrt_weights = np.sqrt(weights)

    # In case of sparse data in the range of the kernel return nan.
    if x.shape[0] < 2:
        return np.nan
    else:
        pass

    # Compute estimate of outcome variable at x0.
    x_powers = np.column_stack((np.ones(shape=x.shape[0]), x))
    x0_powers = x0 ** np.arange(2)
    x_powers_weighted = np.zeros_like(x_powers)
    for j in range(2):
        x_powers_weighted[:, j] = np.multiply(x_powers[:, j], sqrt_weights)
    y_weighted = y * sqrt_weights

    beta_hat = np.linalg.lstsq(a=x_powers_weighted, b=y_weighted)[0]
    y0_hat = np.dot(x0_powers, beta_hat)

    return y0_hat


def cross_validation(data, cutoff, h_grid, min_num_obs):
    """
    Perform leave-one-out cross-validation to select the mean squared error
    optimal bandwidth used in local linear regression out of a given grid.
    The procedure is tailored for the context of Regression Discontinuity Design
    and follows the ideas of Ludwig and Miller (2005) and Imbens and Lemieux (2008).

    Args:
        data (pd.DataFrame): Dataframe with data on the running variable in a
                            column called "r" and data on the dependent variable
                            in a column called "y", both of type np.float64.
        cutoff (float): Cutpoint in the range of the running variable used to
                        distinguish between treatment and control groups.
        h_grid (np.array): Grid of bandwidths taken into consideration.
        min_num_obs (int): Minimum number of observations used for fitting the
                            data at a particular point.

    Returns:
        float: Mean squared error optimal bandwidth out of h_grid.
    """

    if np.any(h_grid <= 0):
        raise ValueError("All bandwidths must be positive.")
    else:
        pass

    data = data[["r", "y"]]
    data_left = np.array(data[data["r"] < cutoff])
    data_right = np.array(data[data["r"] >= cutoff])

    if data_left.size == 0 or data_right.size == 0:
        raise ValueError("Cutoff must lie within range of the running variable.")
    else:
        pass

    mean_squared_errors = np.zeros_like(h_grid)

    for h_index, h in enumerate(h_grid):
        intermediate_res = 0
        runner_not_nan = 0

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
                if np.isnan(y_hat):
                    pass
                else:
                    intermediate_res += (data_left[r_index, 1] - y_hat) ** 2
                    runner_not_nan += 1
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
                if np.isnan(y_hat):
                    pass
                else:
                    intermediate_res += (data_right[r_index, 1] - y_hat) ** 2
                    runner_not_nan += 1
            else:
                pass

        if intermediate_res == 0:
            raise ValueError("The Kernel does never include any data.")
        else:
            mean_squared_errors[h_index] = intermediate_res / runner_not_nan

    h_opt = h_grid[np.argmin(mean_squared_errors)]

    return h_opt
