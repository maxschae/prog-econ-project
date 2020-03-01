.. _functions_parametric:

***********************************
Functions for parametric estimation
***********************************

.. _parametric_treatment_effect_estimation:

Parametric treatment effect estimation
============================================

The code in **src.functions_parametric** includes an implementation of parametric
treatment effect estimation that fits global polynomials on either side of the
cutoff to assess the effect of a binary treatment on an outcome of interest.

We implement the estimation with the following function contained in
*treatment_effect_estimation.py*.

.. automodule:: src.functions_parametric.treatment_effect_estimation
    :members:

Tests for the function implemented using the ``pytest`` framework are included
in *test_treatment_effect_estimation.py*.
