import matplotlib.pyplot as plt
import pandas as pd

from bld.project_paths import project_paths_join as ppj


outcomes = ["ned", "wg_c"]

for outcome in outcomes:
    data = pd.read_stata(
        ppj("OUT_TABLES", "data_analysis", f"plot_results_{outcome}.dta")
    )

    # Collect values used for plotting.
    bw_data = data[data["bandwidth"] != 0]
    coef_degree1 = data.at[data["degree"].eq(1).idxmax(), "coef"]
    coef_degree2 = data.at[data["degree"].eq(2).idxmax(), "coef"]
    h_rot = data.at[data["rot"].eq(1).idxmax(), "bandwidth"]
    h_cv = data.at[data["cv"].eq(1).idxmax(), "bandwidth"]

    # Plot treatment effect estimate as a function of the bandwidth.
    fig, ax = plt.subplots()
    ax.plot("bandwidth", "coef", color="darkblue", linewidth=2.0, data=bw_data)
    fill = ax.fill_between(
        "bandwidth", "conf_int_lower", "coef", data=bw_data, color="darkblue", alpha=0.2
    )
    ax.fill_between(
        "bandwidth", "coef", "conf_int_upper", data=bw_data, color="darkblue", alpha=0.2
    )
    line1 = ax.axhline(
        y=coef_degree1, linewidth=1.5, linestyle="dashed", color="orangered"
    )
    line2 = ax.axhline(
        y=coef_degree2, linewidth=1.5, linestyle="dashed", color="darkgoldenrod"
    )
    line_rot = ax.axvline(
        x=h_rot,
        linewidth=1.5,
        linestyle="dashdot",
        color="forestgreen",
        label="Rule-of-thumb bandwidth",
    )
    line_cv = ax.axvline(
        x=h_cv,
        linewidth=1.5,
        linestyle="dashdot",
        color="firebrick",
        label="Cross-validation bandwidth",
    )
    ax.set_xlabel("Bandwidth")
    ax.set_ylabel("Estimated treatment effect")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    ax.legend(
        (line1, line2, line_rot, line_cv),
        ("Global linear", "Global quadratic", "Rule-of-Thumb", "Cross-validation"),
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    plt.title("Performance of local linear regression for different bandwidths.")
    plt.savefig(
        ppj("OUT_FIGURES", "data_analysis", f"treatment_effect_estimates_{outcome}.png")
    )
