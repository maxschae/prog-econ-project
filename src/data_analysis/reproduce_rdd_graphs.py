import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from bld.project_paths import project_paths_join as ppj


data = pd.read_stata(ppj("IN_DATA", "Data_public_small.dta"))


# Set RDD cutoff.
cutoff = 40

# Assign treatment status.
data["d"] = 0
data.loc[data["age"] >= cutoff, "d"] = 1

data_graph = data.copy()
data_graph = data_graph.loc[data_graph["ned"] < 2 * 365]

# Bin data with age bins covering 4 months.
rmin = min(data_graph["age"])
binsize = 0.333

# Calculate midpoint of lowest bin.
binmp_lowest = np.floor((rmin - cutoff) / binsize) * binsize + binsize / 2 + cutoff

# Assign each running variable observation its bin.
data_graph["binnum"] = round(
    (
        (
            np.floor((data_graph["age"] - cutoff) / binsize) * binsize
            + binsize / 2
            + cutoff
        )
        - binmp_lowest
    )
    / binsize
)

# Calculate mean of outcome and running variable for each discrete value.
data_graph_d = data_graph.groupby(["binnum"], as_index=False).mean()
# Omit first and last bin as they hold too few observations.
data_graph_d = data_graph_d[1:-1]


# Plot data.
sns.set_style("whitegrid")

fig, ax = plt.subplots(figsize=(20, 20), sharex=True)
plt.subplots_adjust(wspace=0.3)

# Specify subplot arrangement and outcome to be plotted against age.
plot_dict = {"221": "ned", "222": "wg_c"}
for subplot in plot_dict.keys():
    # Generate two subplots.
    plt.subplot(subplot)

    for d in [0, 1]:
        # Plot data and quadratic fit separately for each side of cutoff.
        p = sns.regplot(
            "age",
            plot_dict[subplot],
            data=data_graph_d.loc[data_graph_d["d"] == d],
            order=2,
            ci=None,
            color="blue",
            scatter_kws={"s": 50, "alpha": 0.5},
            truncate=True,
        )

    p.tick_params(labelsize=18)
    plt.xlabel("Age at Layoff", size=22)
    plt.axvline(x=cutoff, color="black", alpha=0.8, linestyle="--")

    # Customize subplot's title and label for different outcomes.
    if plot_dict[subplot] == "ned":
        plt.title("Panel A", size=26, loc="left")
        plt.ylabel("Nonemployment Duration", size=22)
    elif plot_dict[subplot] == "wg_c":
        plt.title("Panel B", size=26, loc="left")
        plt.ylabel("Wage Change", size=22)
    else:
        pass


plt.savefig(ppj("OUT_FIGURES", "data_analysis", "rdd_graphs.png"))
