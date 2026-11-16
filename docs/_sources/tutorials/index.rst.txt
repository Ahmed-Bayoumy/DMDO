Tutorials
#########

This part presents complete, runnable examples. The scripts live in the ``examples`` folder of the repository (run
them from that folder) and in the regression suite ``tests/test_basic.py``; each tutorial shows the output they
produce.

:doc:`basic`
   The basic two-discipline problem from a YAML file: running it, reading the console output, the result object and
   the output files.
:doc:`sellar`
   The Sellar problem built with the Python API, and the effect of the inner solver (SLSQP, OMADS poll, OMADS MADS).
:doc:`session`
   Driving a run one outer iteration at a time, observers, early stopping, and solving the subproblems yourself with
   ``ask``/``tell``.
:doc:`parallel`
   Gauss–Seidel and Jacobi sweeps, process pools, custom executors, and why parallel runs give the same answer
   whatever the execution order.
:doc:`extending`
   Analyses as ``Discipline`` objects, named blackboxes, custom subproblem solvers, executors and coordination
   policies, and plugins.
:doc:`service`
   The HTTP service: sessions over HTTP, live events, and remote workers pinned to the machines that hold the
   analysis codes.
:doc:`dashboard`
   The post-processing dashboard: interactive plots of the coordination history and of the subproblem solves.

.. toctree::
   :hidden:

   basic
   sellar
   session
   parallel
   extending
   service
   dashboard

.. note::

   The outputs shown in these tutorials were produced with DMDO |release| on Windows with Python 3.12. Runs are
   reproducible for a given seed and platform; other platforms may differ in the last digits.
