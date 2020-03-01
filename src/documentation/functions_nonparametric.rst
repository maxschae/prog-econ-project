.. _functions_nonparametric:

***************************************
Functions for non-parametric estimation
***************************************

The code in **src.functions_nonparametric** includes an implementation of
non-parametric treatment effect estimation that uses local linear regression to
assess the effect of a binary treatment on an outcome of interest. Further, it
contains an implementation of different algorithms to select the bandwidth used
in local linear regression.

.. _bandwidth_selection_functions:

Bandwidth selection functions
=============================

We implement two data-driven bandwidth selection procedures that are commonly
used in the literature on Regression Discontinuity Designs: Leave-one-out
cross-validation on the one hand and a rule-of-thumb procedure plugging parameter
estimates into a formula for the mean squared error optimal bandwidth on the other.
A detailed description of the algorithms can be found in the research paper in
**src.paper**.

We perform the implementation of the rule-of-thumb bandwidth selection procedure
with the following function located in *rule_of_thumb.py*.

.. automodule:: src.functions_nonparametric.rule_of_thumb
    :members:

The correctness of the function implementation is tested using ``pytest`` in
*test_rule_of_thumb.py*.

For the implementation of leave-one-out cross-validation, we use the following
functions that can be found in *cross_validation.py*.

.. automodule:: src.functions_nonparametric.cross_validation
    :members:

We again perform functional tests using ``pytest`` in *test_cross_validation.py*.

.. _non-parametric_treatment_effect_estimation:

Non-parametric treatment effect estimation
===============================================

The actual non-parametric treatment effect estimation is then performed using
the above procedures in the ensuing function in *treatment_effect_estimation.py*.

.. automodule:: src.functions_nonparametric.treatment_effect_estimation
    :members:

Tests for the function implemented using the ``pytest`` framework are included
in *test_treatment_effect_estimation.py*.
