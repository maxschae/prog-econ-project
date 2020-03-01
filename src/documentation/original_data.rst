.. _original_data:

*************
Original data
*************

The dataset used to replicate results from *Does Extending Unemployment Benefits
Improve Job Quality?* by Nekoei and Weber (2017) and to apply different estimation
procedures is a subset of the administrative social security record data from the
Austrian Social Security Database. It covers individual observations on private
sector job separations in Austria for the period of 1980–2011 and people aged 30
to 50 years. It contains 1,738,787 observations.

Due to Github's restriction on the file size, we only include a subset of the
original data provided by the authors in **src.original_data**. The file
*Data_public_small.dta* excludes the variables that we do not need for our analysis
such that it comprises the following variables:

* *monthly_wage_0*: Monthly wage in the old job.
* *monthly_wage_n0*: Monthly wage in the new job.
* *age*: Age of the individual at the time of layoff.
* *ned*: Non-employment duration computed as the number of days between the end of a lost job and the start of a new job.
* *wg_c*: Wage change between pre- and post-unemployment jobs computed as the log difference between daily wage in the year of separation and in the year when the new job started.
