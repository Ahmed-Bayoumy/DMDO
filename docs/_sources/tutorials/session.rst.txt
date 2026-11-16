The Session API
###############

:class:`~DMDO.Session` is the entry point for driving DMDO from other code. It builds a problem and drives its
coordination loop in three ways, which can be mixed within one run:

``run()``
   to the end, with the session's executor;
``step()``
   one outer iteration at a time, returning an :class:`~DMDO.IterationRecord`;
``ask()`` / ``tell(results)``
   the caller solves the subproblem tasks itself and hands the results back.

All three execute the same coordination code, so they produce identical results.

The example
===========

``examples/session_basics.py`` runs the basic problem (8 outer iterations, for a quick demonstration) in the three
ways:

.. literalinclude:: ../../examples/session_basics.py
   :language: python
   :lines: 5-

Output:

.. code-block:: text

   run():  finished, 8 iterations, objective 5.511311942324389, max|q| 5.000e-03 (Iteration budget exhausted)
     [observer] iteration 0: objective=3.000000 max|q|=9.044e-02
     [observer] iteration 1: objective=4.860200 max|q|=4.142e-02
     [observer] iteration 2: objective=4.903022 max|q|=2.897e-02
     [observer] iteration 3: objective=3.501799 max|q|=9.642e-02
   step(): stopped after 4 iterations (Stopped on request)
   ask/tell (Jacobi): objective 4.337014798880885

Building a session
==================

.. code-block:: python

   Session.from_yaml("problem.yaml", working_dir="runs", blackboxes={...})   # name defaults to the file name
   Session.from_dict(data, working_dir="runs", mdo_name="wing", blackboxes={...})
   Session.from_checkpoint("runs/wing_post/coord_chk.pkl")                    # resume an interrupted run
   Session(mdo)                                                               # an MDO built with the Python API

Problem definitions are plain dictionaries: :func:`~DMDO.load_setup` reads a YAML file into one, which can be
edited before building (``data["coord"]["c1"]["budget"] = 8`` above). Keyword options select the sweep and the
executor (``mode=``, ``sweep=``, ``executor=``, ``max_workers=``) and add observers (``observers=``).

Stepping and stopping
=====================

``step()`` runs one outer iteration and returns its :class:`~DMDO.IterationRecord` (iteration number, objective,
maximum inconsistency, largest penalty weight, change of the design, worst link), or ``None`` once the run is over.
``request_stop()`` ends the run after the current outer iteration; it is safe to call from another thread. In the
example, the inconsistency falls below :math:`3\times10^{-2}` at iteration 2, so the run stops after iteration 3 with
the reason ``"Stopped on request"``.

Observers
=========

An observer receives the hooks of the coordination loop (:doc:`../implementation` lists them in order). Subclass
:class:`~DMDO.RunObserver` and override the hooks you need, or pass any object that defines some of them:

.. code-block:: python

   import json
   from DMDO import RunObserver

   class JsonLines(RunObserver):
       """Append one JSON line per outer iteration, e.g. for a dashboard."""

       def __init__(self, path):
           self.path = path

       def on_iteration_end(self, run, record):
           with open(self.path, "a") as fh:
               fh.write(json.dumps({"iteration": record.iteration, "objective": float(record.objective),
                                    "max_inconsistency": record.max_inconsistency}) + "\n")

   session = Session.from_yaml("problem.yaml", observers=[JsonLines("progress.jsonl")], blackboxes={...})

ask and tell
============

``ask()`` returns the subproblem tasks that can be solved now: one at a time with the Gauss–Seidel sweep, all
subproblems at once with the Jacobi sweep. Each task is a self-contained :class:`~DMDO.SubproblemTask`;
:func:`DMDO.solve_task` solves it, here or anywhere else (another framework's evaluator, a cluster, a remote
service). ``tell(results)`` accepts the results in any order and returns the iteration record when the results
completed an outer iteration.

.. code-block:: python

   from DMDO import solve_task

   while not session.done:
       tasks = session.ask()
       results = [solve_task(task) for task in tasks]     # solve them wherever you like
       record = session.tell(results)
       if record is not None:
           print(record.iteration, record.max_inconsistency)

:doc:`parallel` solves the tasks in a process pool.

Results
=======

``session.result()`` returns a :class:`~DMDO.RunResult`: the status (``created``, ``running``, ``finished``,
``failed``, ``closed``), the stopping reason, the number of outer iterations, the objective of the master
subproblem, the maximum inconsistency, whether the master subproblem is feasible, every variable copy
(``"<name>_<subproblem>"``) and the list of iteration records. ``result.to_dict()`` converts everything to plain
JSON-ready values.

Checkpoints and resume
======================

Every run writes ``coord_chk.pkl`` at the start of each outer iteration after the first. A session resumed from it
continues at the checkpointed iteration and ends with exactly the same result as an uninterrupted run:

.. code-block:: python

   session = Session.from_checkpoint("runs/wing_post/coord_chk.pkl")
   result = session.run()
   print(result.iterations)   # counts the iterations before the checkpoint too
