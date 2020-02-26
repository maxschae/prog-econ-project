import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.functions_nonparametric.rule_of_thumb import rule_of_thumb
from src.functions_nonparametric.treatment_effect_estimation import (
    estimate_treatment_effect_nonparametric,
)
from src.functions_parametric.treatment_effect_estimation import (
    estimate_treatment_effect_parametric,
)

# from src.functions_nonparametric.cross_validation import cross_validation


# data = pd.read_stata("Data_public.dta")
data = pd.read_stata(ppj("IN_DATA", "Data_public_small.dta"))

# Limit dataset to relevant variables only to avoid large file limit.
# data_small = data[["monthly_wage_0", "monthly_wage_n0", "age",
#                   "ned", "wg_c"]]
# data_small.to_stata("Data_public_small.dta", write_index=False)

# Set RDD cutoff.
cutoff = 40


# Assign treatment status.
data["d"] = 0
data.loc[data["age"] >= cutoff, "d"] = 1


for outcome in ["ned", "wg_c"]:
    results = {}

    data_analysis = data.copy()
    # Restrict sample.
    data_analysis = data_analysis.loc[data_analysis["ned"] < 2 * 365]

    # Rename columns to align with estimation functions.
    data_analysis = data_analysis.rename(columns={outcome: "y", "age": "r"})

    if outcome == "wg_c":
        data_analysis.dropna(inplace=True)

    for degree in range(5):
        results[str(degree)] = estimate_treatment_effect_parametric(
            data=data_analysis, cutoff=cutoff, degree=degree,
        )

    h_pilot = rule_of_thumb(data=data_analysis, cutoff=cutoff)
    # Restrict dataset for cross-validation to be computationally feasible.
    """
    data_cv_sample = data_analysis.copy()
    data_cv_sample = data_cv_sample.loc[data_cv_sample["r"] < 43]
    data_cv_sample = data_cv_sample.loc[data_cv_sample["r"] > 37]
    data_cv_sample = data_cv_sample.sample(n=1000)
    h_cv = cross_validation(
                data=data_cv_sample,
                cutoff=cutoff,
                h_grid=np.linspace(start=0.5 * h_pilot, stop=2 * h_pilot, num=32),
                min_num_obs=10,
            )
    """
    results["Rule-of-Thumb"] = estimate_treatment_effect_nonparametric(
        data=data_analysis, cutoff=cutoff, bandwidth=h_pilot,
    )
    """
    results["Cross-Validation"] = estimate_treatment_effect_nonparametric(
                        data=data_analysis,
                        cutoff=cutoff,
                        bandwidth=h_cv,
                        )
    """

    df_result = pd.DataFrame.from_dict(results)
    df_result = df_result.transpose()
    df_result = df_result[["coef", "se"]]

    if outcome == "ned":
        df_result_ned = df_result.copy()
    elif outcome == "wg_c":
        df_result_wg_c = df_result.copy()


df_results = pd.merge(
    left=df_result_ned, right=df_result_wg_c, left_index=True, right_index=True
)
df_results = df_results.rename(
    columns={
        "coef_x": "NE-Duration",
        "coef_y": "Wage Change",
        "se_x": "Std. Err.",
        "se_y": "Std. Err.",
    }
)

# Construct table from dataframe holding results.
with open(
    ppj("OUT_TABLES", "data_analysis", "reproduce_main_results_table_2.tex",), "w",
) as j:
    j.write(df_results.to_latex(index=True))
