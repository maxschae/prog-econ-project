import numpy as np


def y_hat_local_polynomial(x, y, x0, degree=1, bandwidth=1):
    """ Perform local polynomial regression with the triangle kernel to
        predict the value of the dependent variable at some point x0.

    Args:
        x (np.array): Array containing regressor values used for regression.
        y (np.array): Array containing dependent variable used for regression.
        x0 (float): Value at which local polynomial regression is calculated.
        degree (float): Degree of polynomial used in local regression.
        bandwidth (float): Range of data the kernel uses to assign weights.
    Returns:
        y0_hat (float): Predicted value of the dependent variable at x0.
    """

    if bandwidth <= 0:
        raise ValueError("The specified bandwidth must be positive.")
    if degree <= 0:
        raise ValueError(
            "The specified degree for local polynomial regression must be positive."
        )
    else:
        pass

    data_points = np.abs(x - x0) / bandwidth
    weights = np.zeros_like(data_points)
    index = np.where(np.abs(data_points) <= 1)
    weights[index] = 1 - np.abs(data_points[index])

    if np.all(weights == 0):
        raise ValueError("The Kernel does not include any data.")
    else:
        pass

    # Consider only datapoints with positive weight.
    index2 = np.where(np.abs(weights) > 1e-10)[0]
    weights = weights[index2]
    x = x[index2]
    y = y[index2]

    sqrt_weights = np.sqrt(weights)
    x_powers = x[:, None] ** np.arange(degree + 1)
    x0_powers = x0 ** np.arange(degree + 1)

    x_powers_weighted = x_powers * sqrt_weights[:, None]
    y_weighted = y * sqrt_weights

    beta_hat = np.linalg.lstsq(x_powers_weighted, y_weighted, rcond=None)[0]
    y0_hat = x0_powers.dot(beta_hat)

    return y0_hat
