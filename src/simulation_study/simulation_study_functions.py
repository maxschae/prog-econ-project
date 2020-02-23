import numpy as np
from data_generating_process import data_generating_process

from src.functions_nonparametric.cross_validation import cross_validation
from src.functions_nonparametric.rule_of_thumb import rule_of_thumb
from src.functions_nonparametric.treatment_effect_estimation import (
    estimate_treatment_effect_nonparametric,
)
from src.functions_parametric.treatment_effect_estimation import (
    estimate_treatment_effect_parametric,
)


def simulate_estimator_performance(params, degree=1, parametric=True, bandwidth="rot"):
    """Collect performance measures on treatment effect estimator.
        Data stems from data generating process.

    Args:
        params (dict): Contains all parameters for simulating
                       artificial data in a regression discontinuity
                       setting.
        degree (int): Specifies degree of polynomial model fitted to
                      data. Degree of '0' corresponds to comparison
                      in means, '1' to the linear model. Default is 1.
        parametric (bool): Specify whether treatment effect is estimated
                           using 'parametric' or 'non-parametric' methods.
                           Defaults to 'parametric' of degree 1.
        bandwidth (str): Bandwidth used in local linear regression.
                         Either 'cv', 'rot', 'rot_under' or 'rot_over'
                         indicating use of cross-validation or rule-of-thumb
                         bandwidth selection procedure. Also, select
                         under- or oversmoothing of rule-of-thumb bandwidth
                         by choosing 50% or 200% thereof.

    Returns:
        performance_measure (dict): Holds coverage probability,
                                    mean, standard deviation and
                                    mean squared error of
                                    estimator across all M repetions.
    """

    tau_hats = []
    conf_ints = []
    tau_in_conf_int = []

    if parametric is True:
        for _ in range(params["M"]):
            out_reg = estimate_treatment_effect_parametric(
                data=data_generating_process(params=params), degree=degree,
            )
            tau_hat = out_reg["coef"]
            ci_lower, ci_upper = out_reg["conf_int"]

            # Collect estimates for subsequent investigation.
            tau_hats.append(tau_hat)
            conf_ints.append(out_reg["conf_int"])

            if ci_lower <= params["tau"] and params["tau"] <= ci_upper:
                # Count if true value falls into its estimate's confidence region.
                tau_in_conf_int.append(1)
            else:
                tau_in_conf_int.append(0)

    elif parametric is False:
        for _ in range(params["M"]):
            data = data_generating_process(params=params)

            if bandwidth not in ["cv", "rot", "rot_under", "rot_over"]:
                raise ValueError("Specified bandwidth is incorrect.")

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

            else:
                h = 2 * rule_of_thumb(data, params["cutoff"])

            out_reg = estimate_treatment_effect_nonparametric(
                data=data, cutoff=params["cutoff"], bandwidth=h,
            )
            tau_hat = out_reg["coef"]
            ci_lower, ci_upper = out_reg["conf_int"]
            tau_hats.append(tau_hat)
            conf_ints.append(out_reg["conf_int"])
            if ci_lower <= params["tau"] and params["tau"] <= ci_upper:
                tau_in_conf_int.append(1)
            else:
                tau_in_conf_int.append(0)

    else:
        raise TypeError("argument 'parametric' must be boolean.")

    performance_measure = {}
    performance_measure["tau_hat"] = np.mean(tau_hats)
    performance_measure["coverage_prob"] = np.mean(tau_in_conf_int)
    performance_measure["stdev_tau_hat"] = np.std(tau_hats)
    performance_measure["mse_tau_hat"] = np.square(
        np.subtract(tau_hats, params["tau"])
    ).mean()
    performance_measure["tau_hats"] = tau_hats
    performance_measure["conf_ints"] = conf_ints

    return performance_measure
