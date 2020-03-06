## Treatment Effect Estimation in Regression Discontinuity Design

**Authors:** Caroline Krayer, Max Schäfer

We set up this project as part of a course called "Effective Programming Practices for Economists" at the University of Bonn in winter semester 2019/20. It is entirely written in Python and uses WAF to run the code. A conda environment installs the packages needed to reproduce our results.

**Content:**
	Treatment effect estimation in regression discontinuity (RD) studies is interesting and challenging since the effect of treatment features a sense of location -- it is quantified by a jump in the regression function at the RD cutoff and misspecifying the functional form is thus more severe. We study the performance of global parametric and local non-parametric methods along the bias and precision trade-off by means of a Monte Carlo simulation study featuring different data-generating processes and data structures. The results reveal that flexible local methods can better handle involved relationships between the running variable and the outcome and are hence well-suited to produce unbiased estimates. Parametric methods can only shine in cases where the underlying data-generating process is linear or based on polynomials -- then, they feature a higher precision compared to non-parametric methods. Using parametric and/or non-parametric methods thus involves a thorough understanding of the underlying data and how they came about. Further, we challenge the set of estimators on a real dataset and revisit results of [Nekoei and Weber (2017)](https://www.aeaweb.org/articles?id=10.1257/aer.20150528) who study the effect of eligible unemployment benefit duration on time unemployed and wage change using a RD approach. We find that their main reported estimate is the largest among all estimates we produce using a variety of bandwidths for local and a variety of polynomial degrees for global methods.




<hr />


[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
