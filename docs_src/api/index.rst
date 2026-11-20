API reference
=============

The public API is importable from the top-level package (``from DMDO import Session``); the service layer from
``DMDO.service``.

:doc:`session`
   :class:`~DMDO.Session`, :class:`~DMDO.RunResult` and the problem-building functions.
:doc:`workflow`
   The coordination loop (:class:`~DMDO.MDO`), task messages and executors.
:doc:`ports`
   The interfaces DMDO talks to: disciplines, subproblem solvers, executors and observers.
:doc:`extension`
   Registries, solver strategies, coordination policies and the built-in observers.
:doc:`architectures`
   Architecture selection (``MDO.architecture``), hierarchical ATC, monolithic MDF and the MDA solvers.
:doc:`problem`
   Variables, analyses, subproblems, the coordinator and the enumerations used to build a problem in Python.
:doc:`service`
   The optional HTTP service, its client and remote workers.

.. toctree::
   :hidden:

   session
   workflow
   ports
   extension
   architectures
   problem
   service
