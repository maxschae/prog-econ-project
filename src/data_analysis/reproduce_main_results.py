import numpy as np
import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.functions_nonparametric.cross_validation import cross_validation
from src.functions_nonparametric.rule_of_thumb import rule_of_thumb
from src.functions_nonparametric.treatment_effect_estimation import (
    estimate_treatment_effect_nonparametric,
)
from src.functions_parametric.treatment_effect_estimation import (
    estimate_treatment_effect_parametric,
)

# Read in data.
data = pd.read_stata(ppj("IN_DATA", "Data_public_small.dta"))
data = data.astype(np.float64)
cutoff = 40.0

# Assign treatment status.
data["d"] = 0
data.loc[data["age"] >= cutoff, "d"] = 1


for outcome in ["ned", "wg_c"]:
    results = {}

    # Select data relevant for analysis.
    data_analysis = data.copy()
    data_analysis = data_analysis.loc[data_analysis["ned"] < 2 * 365]

    # Rename columns to align with estimation functions.
    data_analysis = data_analysis.rename(columns={outcome: "y", "age": "r"})

    if outcome == "wg_c":
        data_analysis.dropna(inplace=True)
    else:
        pass

    # Parametric treatment effect estimation.
    for degree in range(5):
        results[f"Degree {degree}"] = estimate_treatment_effect_parametric(
            data=data_analysis, cutoff=cutoff, degree=degree,
        )

    # Non-parametric treatment effect estimation.
    h_rot = rule_of_thumb(data=data_analysis, cutoff=cutoff)

    # Restrict dataset for cross-validation to be computationally feasible.
    np.random.seed(123)
    data_cv_sample = data_analysis.copy()
    data_cv_sample = data_cv_sample.loc[data_cv_sample["r"] < 43]
    data_cv_sample = data_cv_sample.loc[data_cv_sample["r"] > 37]
    data_cv_sample = data_cv_sample.sample(n=2000)
    h_grid = np.linspace(start=0.5 * h_rot, stop=10, num=32)
    h_cv = cross_validation(
        data=data_cv_sample, cutoff=cutoff, h_grid=h_grid, min_num_obs=10,
    )

    results["Rule-of-Thumb"] = estimate_treatment_effect_nonparametric(
        data=data_analysis, cutoff=cutoff, bandwidth=h_rot,
    )

    results["Cross-Validation"] = estimate_treatment_effect_nonparametric(
        data=data_analysis, cutoff=cutoff, bandwidth=h_cv,
    )

    for index, h in enumerate(h_grid):
        results[index + 7] = estimate_treatment_effect_nonparametric(
            data=data_analysis, cutoff=cutoff, bandwidth=h
        )

    # Store results in DataFrame, distinguish between results for table and plot.
    df_result = pd.DataFrame.from_dict(results)
    df_result = df_result.transpose()

    # Collect results for regression table and round the numbers.
    df_table_result = df_result[["coef", "se", "p_value"]]
    df_table_result = df_table_result.iloc[:7, :]
    df_table_result = df_table_result.round(4)

    # Collect results used in the plots considering bandwidth performance.
    df_plot_result = df_result[["coef", "conf_int_lower", "conf_int_upper"]]
    df_plot_result["bandwidth"] = np.nan
    df_plot_result.loc["Rule-of-Thumb", "bandwidth"] = h_rot
    df_plot_result.loc["Cross-Validation", "bandwidth"] = h_cv
    df_plot_result.loc[7:, "bandwidth"] = h_grid
    df_plot_result["rot"] = 0
    df_plot_result.loc["Rule-of-Thumb", "rot"] = 1
    df_plot_result["cv"] = 0
    df_plot_result.loc["Cross-Validation", "cv"] = 1
    df_plot_result["degree"] = np.nan
    df_plot_result["degree"].iloc[:5] = range(5)
    df_plot_result.sort_values(by="bandwidth", axis=0, inplace=True)
    df_plot_result.index = range(39)

    if outcome == "ned":
        df_result_ned = df_table_result.copy()
        df_plot_result_ned = df_plot_result.copy()
    elif outcome == "wg_c":
        df_result_wg_c = df_table_result.copy()
        df_plot_result_wg_c = df_plot_result.copy()
    else:
        pass

df_table_results = pd.merge(
    left=df_result_ned, right=df_result_wg_c, left_index=True, right_index=True
)
df_table_results = df_table_results.rename(
    columns={
        "coef_x": "NE-Duration",
        "coef_y": "Wage Change",
        "se_x": "Std. Err.",
        "se_y": "Std. Err.",
        "p_value_x": "p-value",
        "p_value_y": "p-value",
    }
)


# Construct LaTex table from dataframe holding results.
with open(
    ppj("OUT_TABLES", "data_analysis", "reproduce_main_results_table_2.tex",), "w",
) as j:
    j.write(df_table_results.to_latex(index=True))

# Store results for plotting purposes in separate .dta-files.
df_plot_result_ned.to_stata(ppj("OUT_TABLES", "data_analysis", "plot_results_ned.dta"))
df_plot_result_wg_c.to_stata(
    ppj("OUT_TABLES", "data_analysis", "plot_results_wg_c.dta")
)
