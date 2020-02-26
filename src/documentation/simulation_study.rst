.. _simulation_study:

****************
Simulation Study
****************


The directory *src.simulation_study* contains source files that might differ by model and that are potentially used at various steps of the analysis.

For example, you may have a class that is used both in the :ref:`data_analysis` and the :ref:`introduction` steps. Additionally, maybe you have different utility functions in the baseline version and for your robustness check. You can just inherit from the baseline class and override the utility function then.


Simulation Study ``Example``
============================================

**src.simulation_study**

.. automodule:: src.simulation_study.data_generating_process
    :members:

.. automodule:: src.simulation_study.simulation_study.fix_simulation_params
    :members:
