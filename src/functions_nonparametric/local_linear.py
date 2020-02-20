import numba
import numpy as np


@numba.jit(nopython=True)
def y_hat_local_linear(x, y, x0, bandwidth):
    """ Perform local linear regression with the triangle kernel to
        predict the value of the dependent variable at some point x0.

    Args:
        x (np.array): Array containing regressor values used for regression.
        y (np.array): Array containing dependent variable used for regression.
        x0 (float): Value at which local linear regression is calculated.
        bandwidth (float): Range of data the kernel uses to assign weights.

    Returns:
        y0_hat (float): Predicted value of the dependent variable at x0.
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

    # Consider only datapoints with positive weight.
    x = x[np.where(weights > 0)]
    y = y[np.where(weights > 0)]
    weights = weights[np.where(weights > 0)]
    sqrt_weights = np.sqrt(weights)

    # In case of sparse data return nan.
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
