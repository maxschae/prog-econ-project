import numpy as np

from src.functions_nonparametric.cross_validation import cross_validation
from src.functions_nonparametric.rule_of_thumb import rule_of_thumb
from src.functions_nonparametric.treatment_effect_estimation import (
    estimate_treatment_effect_nonparametric,
)
from src.functions_parametric.treatment_effect_estimation import (
    estimate_treatment_effect_parametric,
)
from src.simulation_study.data_generating_process import data_generating_process


def simulate_estimator_performance(params, degree, parametric, bandwidth):
    """
    Collect performance measures on the specified treatment effect estimator applied
    to data simulated with the data_generating_process function. The function works
    for parametric as well as non-parametric treatment effect estimation methods.

    Args:
        params (dict): Dictionary containing simulation parameters.
        degree (int): Degree of polynomial used for global polynomial fitting.
                        A degree of 0 corresponds to a comparison in means.
        parametric (bool): Indication whether the treatment effect is estimated
                           using parametric or non-parametric methods.
        bandwidth (str): Bandwidth used in local linear regression. Options are
                        leave-one-out cross-validation "cv", the rule-of-thumb
                        bandwidth selection procedure "rot" or rescaling of the
                        rule-of-thumb bandwidth by taking 50% or 200% of it,
                        "rot_under" or "rot_over", respectively.

    Returns:
        dict: Dictionary containing measures for descriptive statistics -
            the coverage probability, mean, standard deviation and mean squared
            error of the estimator across all Monte Carlo repetitions as well as
            numeric values of the bandwidths selected by the single procedures.
    """

    tau_hats = []
    tau_in_conf_int = []
    bandwidths_numeric = []

    if parametric is True:
        for _ in range(params["M"]):
            out_reg = estimate_treatment_effect_parametric(
                data=data_generating_process(params=params),
                cutoff=params["cutoff"],
                degree=degree,
            )
            tau_hat = out_reg["coef"]
            ci_lower = out_reg["conf_int_lower"]
            ci_upper = out_reg["conf_int_upper"]

            # Collect estimates for subsequent investigation.
            tau_hats.append(tau_hat)
            if ci_lower <= params["tau"] and params["tau"] <= ci_upper:
                # Count if true value falls into its estimate's confidence region.
                tau_in_conf_int.append(1)
            else:
                tau_in_conf_int.append(0)

    elif parametric is False:
        for _ in range(params["M"]):
            data = data_generating_process(params=params)

            if bandwidth == "cv":
                h_pilot = rule_of_thumb(data, params["cutoff"])
                h = cross_validation(
                    data=data,
                    cutoff=params["cutoff"],
                    h_grid=np.linspace(start=0.5 * h_pilot, stop=2 * h_pilot, num=32),
                    min_num_obs=10,
                )
            elif bandwidth == "rot":
                h = rule_of_thumb(data, params["cutoff"])

            elif bandwidth == "rot_under":
                h = 0.5 * rule_of_thumb(data, params["cutoff"])

            elif bandwidth == "rot_over":
                h = 2 * rule_of_thumb(data, params["cutoff"])

            else:
                raise ValueError("The specified bandwidth procedure is incorrect.")

            out_reg = estimate_treatment_effect_nonparametric(
                data=data, cutoff=params["cutoff"], bandwidth=h,
            )
            tau_hat = out_reg["coef"]
            ci_lower = out_reg["conf_int_lower"]
            ci_upper = out_reg["conf_int_upper"]
            tau_hats.append(tau_hat)
            bandwidths_numeric.append(h)
            if ci_lower <= params["tau"] and params["tau"] <= ci_upper:
                tau_in_conf_int.append(1)
            else:
                tau_in_conf_int.append(0)

    else:
        raise TypeError("Argument 'parametric' must be boolean.")

    performance_measure = {}
    performance_measure["tau_hat"] = np.mean(tau_hats)
    performance_measure["coverage_prob"] = np.mean(tau_in_conf_int)
    performance_measure["stdev_tau_hat"] = np.std(tau_hats)
    performance_measure["mse_tau_hat"] = np.square(
        np.subtract(tau_hats, params["tau"])
    ).mean()
    performance_measure["bandwidths_numeric"] = bandwidths_numeric

    return performance_measure
