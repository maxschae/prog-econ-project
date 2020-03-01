.. _data_analysis:

****************
Data application
****************

The code in **src.data_analysis** contains an application of the studied parametric
and non-parametric treatment effect estimation methods to a large administrative
data set used in *Does Extending Unemployment Benefits Improve Job Quality?* by
Nekoei and Weber (2017). The goal is to estimate the effect of extended unemployment
benefits on non-employment duration and wage changes between the old and the new
job, respectively, by means of Regression Discontinuity Design.

In *reproduce_main_results.py*, we apply the implemented functions in
**src.functions_parametric** and **src.functions_nonparametric** to assess the
effect of the treatment on the outcomes of interest.

The file *reproduce_rdd_graphs.py* reproduces a figure from the original paper
that plots the data for a grouped running variable and corresponding average
outcome and adds the fitted line of a quadratic polynomial to get an intuition
for the discontinuity gap.

In *treatment_effect_estimate_plot.py*, we add a plot of the non-parametric
treatment effect estimate and 95 percent confidence intervals as a function of the
bandwidth to compare the bandwidth selection procedures considered.
