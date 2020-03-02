.. _simulation_study:

****************
Simulation Study
****************

The code in **src.simulation_study** constitutes the main part of analysis for
evaluating the performance of parametric treatment effect estimators towards
non-parametric ones. It includes functions that simulate artificial data used
for the analysis as well as functions assessing the performance of the estimation
methods when applied to the simulated data. The code in *sim_study.py* further
contains the actual simulation study using the above functions.

.. _data_generating_process:

Data Generating Process
============================================

A setup of the simulation environment is performed with the following function in
*sim_study.py*.

.. automodule:: src.simulation_study.sim_study
    :members:

.. raw:: latex

    \clearpage

The actual implementation of the data generating process is then contained in
*data_generating_process.py*.

.. automodule:: src.simulation_study.data_generating_process
    :members:

We add functional tests using ``pytest`` in *test_data_generating_process.py*.

To highlight the data generating process, we construct plots for a visualisation
of the single model specifications in *produce_simulated_rdd_graphs.py*.

.. _evaluation_simulation:

Evaluation of the Simulation
============================================

To assess the performance of the different estimation methods considered, we use
the following function in *simulate_estimator_performance.py*.

.. automodule:: src.simulation_study.simulate_estimator_performance
    :members:

Functional tests using the ``pytest`` framework are included in
*test_simulate_estimator_performance.py*.
