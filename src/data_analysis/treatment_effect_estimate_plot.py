import matplotlib.pyplot as plt
import pandas as pd

from bld.project_paths import project_paths_join as ppj


# Create plots.
fig, ax = plt.subplots(figsize=(12, 7.5), sharex=True)
plt.subplots_adjust(wspace=0.3)
plot_dict = {"221": "ned", "222": "wg_c"}

for subplot in plot_dict.keys():
    data = pd.read_stata(
        ppj("OUT_TABLES", "data_analysis", f"plot_results_{plot_dict[subplot]}.dta")
    )

    # Collect values used for plotting.
    bw_data = data[data["bandwidth"] != 0]
    coef_degree1 = data.at[data["degree"].eq(1).idxmax(), "coef"]
    coef_degree2 = data.at[data["degree"].eq(2).idxmax(), "coef"]
    h_rot = data.at[data["rot"].eq(1).idxmax(), "bandwidth"]
    h_cv = data.at[data["cv"].eq(1).idxmax(), "bandwidth"]

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
        alpha=0.15,
    )
    plt.fill_between(
        "bandwidth",
        "coef",
        "conf_int_upper",
        data=bw_data,
        color="darkblue",
        alpha=0.15,
    )
    line1 = plt.axhline(y=coef_degree1, linestyle="dashed", color="orangered")
    line2 = plt.axhline(y=coef_degree2, linestyle="dashdot", color="orangered")
    line_rot = plt.axvline(
        x=h_rot, linestyle="dashed", color="green", label="Rule-of-thumb bandwidth",
    )
    line_cv = plt.axvline(
        x=h_cv, linestyle="dashdot", color="green", label="Cross-validation bandwidth",
    )
    plt.xlabel("Bandwidth", size=12)
    plt.ylabel("Treatment effect estimate", size=12)

    # Customize subplot's title and label for different outcomes.
    if plot_dict[subplot] == "ned":
        plt.title("Non-employment duration", size=16, loc="center")
    elif plot_dict[subplot] == "wg_c":
        plt.title("Wage change", size=16, loc="center")
    else:
        pass

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
plt.legend(
    (line1, line2, line_rot, line_cv),
    ("Global linear", "Global quadratic", "Rule-of-Thumb", "Cross-validation"),
    loc="center left",
    bbox_to_anchor=(1, 0.5),
)

# Adjust the scaling factor to fit your legend text completely outside the plot
# (smaller value results in more space being made for the legend)
fig.subplots_adjust(right=0.85)

fig.savefig(ppj("OUT_FIGURES", "data_analysis", f"treatment_effect_estimates.png"))
