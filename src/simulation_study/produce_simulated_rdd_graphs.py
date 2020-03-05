import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from bld.project_paths import project_paths_join as ppj
from src.simulation_study.data_generating_process import data_generating_process
from src.simulation_study.sim_study import fix_simulation_params


np.random.seed(123)
data_dict = {}

for model in ["linear", "poly", "nonparametric"]:
    # Draw data from each data generating process.
    sim_params = fix_simulation_params(
        n=500, M=100, model=model, discrete=False, cutoff=0, tau=0.75, noise_var=0.25
    )

    data_temp = data_generating_process(params=sim_params)

    # Bin data.
    rmin = min(data_temp["r"])
    binsize = 0.07

    cutoff = sim_params["cutoff"]

    # Calculate midpoint of lowest bin.
    binmp_lowest = np.floor((rmin - cutoff) / binsize) * binsize + binsize / 2 + cutoff

    # Assign each running variable observation its bin.
    data_temp["binnum"] = round(
        (
            (
                np.floor((data_temp["r"] - cutoff) / binsize) * binsize
                + binsize / 2
                + cutoff
            )
            - binmp_lowest
        )
        / binsize
    )

    # Calculate mean of outcome and running variable for each discrete value.
    data_temp = data_temp.groupby(["binnum"], as_index=False).mean()
    # Omit first and last bins as they hold too few observations.
    data_temp = data_temp[3:-3]

    data_dict["data_" + model] = data_temp.rename({"y": "y_" + model})


# Plot binned data.
sns.set_style("whitegrid")

fig, ax = plt.subplots(figsize=(24, 6), sharex=True)
plt.subplots_adjust(wspace=0.3)

# Specify subplot arrangement and outcome by
# different data generating processes.
plot_dict = {"131": "linear", "132": "poly", "133": "nonparametric"}

for subplot in plot_dict.keys():
    plt.subplot(subplot)
    # Prepare data.
    data_graph = data_dict["data_" + plot_dict[subplot]]

    for d in [0, 1]:
        # Plot data and quadratic fit separately for each side of cutoff.
        p = sns.regplot(
            "r",
            "y",
            data=data_graph.loc[data_graph["d"] == d],
            fit_reg=False,
            order=2,
            ci=None,
            color="blue",
            scatter_kws={"s": 50, "alpha": 0.5},
            truncate=True,
        )

    p.tick_params(labelsize=18)
    plt.xlabel("R", size=22)
    plt.ylabel("Y", size=22)
    plt.axvline(x=cutoff, color="black", alpha=0.8, linestyle="--")

    # Customize subplot's title and label for different outcomes.
    if subplot == "131":
        plt.title("Panel A", size=26, loc="left")
    elif subplot == "132":
        plt.title("Panel B", size=26, loc="left")
    elif subplot == "133":
        plt.title("Panel C", size=26, loc="left")
    else:
        pass
plt.savefig("a.png")
plt.savefig(ppj("OUT_FIGURES", "simulation_study", "simulated_rdd_graphs.png"))
