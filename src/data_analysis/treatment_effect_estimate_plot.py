import matplotlib.pyplot as plt
import pandas as pd

from bld.project_paths import project_paths_join as ppj


# Create plot for performance of different bandwidths.
fig, ax = plt.subplots(figsize=(12, 4), sharex=True)
plt.subplots_adjust(wspace=0.3)
plot_dict = {"121": "ned", "122": "wg_c"}

for subplot in plot_dict.keys():
    data = pd.read_stata(
        ppj("OUT_TABLES", "data_analysis", f"plot_results_{plot_dict[subplot]}.dta")
    )
    data_graph = data.copy()

    # Collect values used for plotting.
    bw_data = data_graph[data_graph["bandwidth"] != 0]
    coef_degree1 = data_graph.at[data_graph["degree"].eq(1).idxmax(), "coef"]
    coef_degree2 = data_graph.at[data_graph["degree"].eq(2).idxmax(), "coef"]
    h_rot = data_graph.at[data_graph["rot"].eq(1).idxmax(), "bandwidth"]
    h_cv = data_graph.at[data_graph["cv"].eq(1).idxmax(), "bandwidth"]

    # Plot corresponding treatment effect estimate as a function of the bandwidth.
    plt.subplot(subplot)
    plt.plot("bandwidth", "coef", color="darkblue", linewidth=2.0, data=bw_data)
    plt.plot(
        "bandwidth",
        "conf_int_lower",
        data=bw_data,
        color="darkblue",
        linestyle="dashed",
    )
    plt.plot(
        "bandwidth",
        "conf_int_upper",
        data=bw_data,
        color="darkblue",
        linestyle="dashed",
    )
    plt.fill_between(
        "bandwidth",
        "conf_int_lower",
        "coef",
        data=bw_data,
        color="darkblue",
        alpha=0.1,
    )
    plt.fill_between(
        "bandwidth",
        "coef",
        "conf_int_upper",
        data=bw_data,
        color="darkblue",
        alpha=0.1,
    )
    line1 = plt.axhline(y=coef_degree1, linestyle="dashed", color="orangered")
    line2 = plt.axhline(y=coef_degree2, linestyle="dashdot", color="orangered")
    line_rot = plt.axvline(
        x=h_rot, linestyle="dashed", color="green", label="Rule-of-thumb bandwidth",
    )
    line_cv = plt.axvline(
        x=h_cv, linestyle="dashdot", color="green", label="Cross-validation bandwidth",
    )
    plt.xlabel("Bandwidth", size=14)

    # Adjust y-axis and title depending on outcome variable.
    if plot_dict[subplot] == "ned":
        plt.title("Panel A", size=16, loc="left")
        plt.ylabel("Non-employment duration", size=14)

    elif plot_dict[subplot] == "wg_c":
        plt.title("Panel B", size=16, loc="left")
        plt.ylabel("Wage change", size=14)
    else:
        pass

# Create legend and adjust scaling to fit it completely outside the plot.
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
plt.legend(
    (line1, line2, line_rot, line_cv),
    ("Global linear", "Global quadratic", "Rule-of-Thumb", "Cross-validation"),
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    fontsize=10,
)
fig.subplots_adjust(right=0.85)

fig.savefig(ppj("OUT_FIGURES", "data_analysis", f"treatment_effect_estimates.png"))
