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
    ax.plot(
        "bandwidth",
        "conf_int_lower",
        data=bw_data,
        color="darkblue",
        linestyle="dashed",
    )
    ax.plot(
        "bandwidth",
        "conf_int_upper",
        data=bw_data,
        color="darkblue",
        linestyle="dashed",
    )
    ax.fill_between(
        "bandwidth",
        "conf_int_lower",
        "coef",
        data=bw_data,
        color="darkblue",
        alpha=0.15,
    )
    ax.fill_between(
        "bandwidth",
        "coef",
        "conf_int_upper",
        data=bw_data,
        color="darkblue",
        alpha=0.15,
    )
    line1 = ax.axhline(y=coef_degree1, linestyle="dashed", color="orangered")
    line2 = ax.axhline(y=coef_degree2, linestyle="dashdot", color="orangered")
    line_rot = ax.axvline(
        x=h_rot, linestyle="dashed", color="green", label="Rule-of-thumb bandwidth",
    )
    line_cv = ax.axvline(
        x=h_cv, linestyle="dashdot", color="green", label="Cross-validation bandwidth",
    )
    ax.set_xlabel("Bandwidth")
    ax.set_ylabel("Treatment effect estimate")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    if outcome == "wg_c":
        ax.legend(
            (line1, line2, line_rot, line_cv),
            ("Global linear", "Global quadratic", "Rule-of-Thumb", "Cross-validation"),
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )
    else:
        pass

    plt.savefig(
        ppj("OUT_FIGURES", "data_analysis", f"treatment_effect_estimates_{outcome}.png")
    )
