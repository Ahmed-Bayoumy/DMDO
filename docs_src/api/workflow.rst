Coordination loop and execution
===============================

``MDO``
-------

.. class:: DMDO.MDO

   The multidisciplinary design optimization process: the coordinator, the subproblems, the variable copies and the
   records of a run (``fmin``, ``hmin``, ``tab_inc`` with the maximum inconsistency of every outer iteration,
   ``stop`` with the stopping reason). Built by :func:`~DMDO.build_mdo` or directly with the Python API
   (:doc:`../tutorials/sellar`).

   .. automethod:: DMDO.MDO.run

   .. automethod:: DMDO.MDO.coordinate

Tasks
-----

.. autoclass:: DMDO.SubproblemTask

.. autoclass:: DMDO.SubproblemResult

.. autofunction:: DMDO.solve_task

Executors
---------

.. autoclass:: DMDO.InProcessExecutor
   :members: map, close

.. autoclass:: DMDO.ProcessExecutor
   :members: map, close

.. autofunction:: DMDO.create_executor
