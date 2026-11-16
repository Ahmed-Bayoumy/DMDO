Interfaces (ports)
==================

.. code-block:: python

   from DMDO import Discipline, SubproblemSolver, SolverRequest, SolverOutcome, Executor, RunObserver, IterationRecord

Disciplines
-----------

.. autoclass:: DMDO.Discipline
   :members: compute

Subproblem solvers
------------------

.. class:: DMDO.SubproblemSolver

   An optimizer for one subproblem (a protocol; subclass :class:`~DMDO.BaseSolver` to implement one).

   .. attribute:: objective_form

      ``"vector"``: ``request.objective(x)`` returns ``[f + penalty, [g1, g2, ...]]``.

   .. method:: solve(request)

      Optimize the subproblem described by ``request`` (a :class:`~DMDO.SolverRequest`) and return a
      :class:`~DMDO.SolverOutcome`.

   .. method:: report_hmin(hmin)

      The constraint violation of a solve in the form recorded by the coordination loop (identity by default).

.. autoclass:: DMDO.SolverRequest

.. autoclass:: DMDO.SolverOutcome

Executors
---------

.. autoclass:: DMDO.Executor
   :members: map, close

Observers
---------

.. class:: DMDO.RunObserver

   Hooks called by the coordination loop. Subclass and override the hooks you need; every hook is optional, and any
   object that defines some of them can be used as an observer. ``run`` is the :class:`~DMDO.MDO` object: read its
   state (``run.Coordinator``, ``run.subProblems``, ``run.fmin``, ...) rather than keeping references to it.
   Observers are called in the order they are given, after DMDO's own, and are not pickled into checkpoints.

   .. method:: on_run_start(run)

      Once, before the first outer iteration.

   .. method:: on_iteration_start(run, iteration)

      Before the subproblems are solved.

   .. method:: on_resume(run, iteration)

      Instead of ``on_iteration_start`` for the first iteration of a run resumed from a checkpoint.

   .. method:: on_sweep_end(run, iteration)

      All subproblems solved and the inconsistencies computed, before the penalty update.

   .. method:: on_iteration_end(run, record)

      The iteration is final for reporting; ``record`` is an :class:`~DMDO.IterationRecord`.

   .. method:: on_multipliers_updated(run, iteration)

      After the multipliers and penalty weights were updated, before the stopping test.

   .. method:: on_run_end(run)

      Once, after the last outer iteration (not called when the run raises).

.. autoclass:: DMDO.IterationRecord
