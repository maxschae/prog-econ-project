import random
from time import time

import numpy as np
import pandas as pd
from cross_validation import cross_validation
from rule_of_thumb import rule_of_thumb

# Generate test data for bandwidth selection functions.
random.seed(123)
n = 1000
cutoff = 0
h_grid = np.linspace(start=0.5, stop=3, num=50)
degree = 1
min_num_obs = 5

data = pd.DataFrame()
data["r"] = np.random.normal(loc=0, scale=1, size=n)
data["d"] = 0
data.loc[data["r"] >= cutoff, "d"] = 1
data["y"] = (
    0.5 * data["d"]
    + 1 * data["r"]
    + 3 * data["d"] * data["r"]
    + np.random.normal(loc=0, scale=1, size=n)
)


# Time the bandwidth selection functions.
runtimes_rot = []
runtimes_cv = []
for _i in range(10):
    start_rot = time()
    rule_of_thumb(data=data, cutoff=cutoff)
    stop_rot = time()
    runtimes_rot.append(stop_rot - start_rot)

for _j in range(10):
    start_cv = time()
    cross_validation(
        data=data, cutoff=cutoff, h_grid=h_grid, degree=degree, min_num_obs=min_num_obs
    )
    stop_cv = time()
    runtimes_cv.append(stop_cv - start_cv)

# Calculate mean runtimes.
mean_runtime_rot = np.mean(runtimes_rot[1:])
mean_runtime_cv = np.mean(runtimes_cv[1:])
print(f"Rule-of-thumb bandwidth selection function took {mean_runtime_rot} seconds.")
print(f"Cross-validation bandwidth selection function took {mean_runtime_cv} seconds.")
