Implementation
##############

This chapter explains how DMDO is organized: the package layers, the interfaces through which it talks to analyses,
optimizers and execution back ends, the coordination loop, and the layers built on top of it (sessions, observers and
the HTTP service). It is meant for users who extend DMDO or embed it in another tool.

Design principles
=================

DMDO follows a **ports and adapters** design. The coordination logic depends only on a few small interfaces
(*ports*); everything that varies between applications (the analyses, the subproblem optimizers, where the
subproblems run, what is recorded) is an *adapter* that implements one of them. DMDO never imports a third-party
modeling framework: an adapter written in user code only has to satisfy the interface.

* **Strategies** select behavior by name: subproblem solvers, executors and coordination policies are interchangeable
  classes looked up in **registries**, which plugins extend through Python entry points.
* **Messages** decouple the loop from where work is done: a subproblem solve is a picklable task that any executor
  can run, in this process, in a worker process or on another machine.
* **Observers** receive every phase of the loop, so outputs (history file, console, checkpoints) and monitoring are
  pluggable instead of hard-wired.
* **One loop, several drivers**: the coordination loop is a generator; ``MDO.run``, ``Session.step`` and
  ``Session.ask``/``tell`` all drive the same code.

Package layout
==============

Each subpackage may only import from lower layers; a test of the test suite enforces this rule.

.. list-table::
   :header-rows: 1
   :widths: 22 10 68

   * - Subpackage
     - Layer
     - Responsibility
   * - ``DMDO.core``
     - 0
     - Variables (:class:`~DMDO.variableData`), enumerations and base protocols.
   * - ``DMDO.utils``, ``DMDO.ports``, ``DMDO.registry``
     - 1
     - Logging and seeding; the interfaces (ports) and messages; the registries.
   * - ``DMDO.solvers``, ``DMDO.analysis``, ``DMDO.coordination``
     - 3
     - Subproblem solver strategies; disciplinary analyses and MDA groups; the ADMM coordinator and its adaptation
       policies.
   * - ``DMDO.subproblem``
     - 4
     - The subproblem: assembles the penalized objective and calls its solver strategy.
   * - ``DMDO.workflow``
     - 5
     - The coordination loop (:class:`~DMDO.MDO`), executors, task messages and the built-in observers.
   * - ``DMDO.app``
     - 6
     - Problem building from YAML or dictionaries, :func:`~DMDO.main` and the :class:`~DMDO.Session` API.
   * - ``DMDO.service``
     - 7
     - Optional HTTP service, remote workers and client (standard library only, not imported by ``import DMDO``).

Module paths of earlier releases (``DMDO.SP``, ``DMDO.MDO``, ``DMDO.coordinator``, ``DMDO.preprocess``, ...) remain
importable as aliases of the new modules, and checkpoints written by earlier releases can still be loaded.

Ports
=====

``Discipline``
   An analysis: ``compute(inputs) -> outputs``, with inputs and outputs keyed by variable name. Plain callables that
   take and return lists remain supported.
``SubproblemSolver``
   An optimizer for one subproblem: ``solve(request) -> outcome``. The :class:`~DMDO.SolverRequest` carries the
   penalized objective, the constraints, the start point, bounds, variable types, budget and seed; the
   :class:`~DMDO.SolverOutcome` returns the best point, its objective and constraint violation, and solver-specific
   state to warm-start the next outer iteration.
``Executor``
   Runs subproblem tasks: ``map(fn, tasks) -> results`` (results in task order) and ``close()``. Which subproblems
   see each other's updates is decided by the sweep, not by the executor.
``RunObserver``
   Optional hooks called by the loop (see `Observers`_).

Registries and plugins
======================

.. list-table::
   :header-rows: 1
   :widths: 26 26 48

   * - Registry
     - Entry-point group
     - Built-in entries
   * - ``DMDO.solvers``
     - ``dmdo.solvers``
     - ``mads`` (alias ``OMADS``), ``poll``, ``search``, ``scipy``
   * - ``DMDO.executors``
     - ``dmdo.executors``
     - ``inprocess`` (alias ``serial``), ``process`` (alias ``parallel``)
   * - ``DMDO.blackboxes``
     - ``dmdo.blackboxes``
     - (empty) analyses and objectives referenced by name from problem definitions
   * - ``DMDO.policies``
     - ``dmdo.policies``
     - ``none``: the plain ADMM update of :doc:`background`

Names are case-insensitive. Besides registered names, every registry accepts an import path ``"package.module:attr"``,
and installed packages can contribute entries without any code change in DMDO:

.. code-block:: toml

   # pyproject.toml of a plugin package
   [project.entry-points."dmdo.solvers"]
   my_solver = "my_pkg.solvers:MySolver"

Subproblem solvers
==================

A subproblem translates its state into a :class:`~DMDO.SolverRequest` and hands it to its solver strategy:

``mads``, ``poll``, ``search``
   OMADS :cite:p:`OMADS_AB`: the complete Mesh Adaptive Direct Search algorithm :cite:p:`audet2006mesh`, its poll
   step alone, or its sampling search alone. Derivative-free, suitable for noisy, non-smooth or failing analyses.
   Options come from the subproblem's ``CONFIG`` entry; the frame size of the previous outer iteration warm-starts
   the next one.
``scipy``
   :func:`scipy.optimize.minimize` (SLSQP by default) on the scalar penalized objective. With ``is_con: true`` the
   subproblem constraints are passed to the optimizer as inequality constraints; by default they are added to the
   objective as a quadratic penalty :math:`\max(g, 0)^2`, and points violating a constraint by more than
   :math:`10^{-3}` are rejected.

A custom strategy subclasses :class:`~DMDO.BaseSolver`; see :doc:`tutorials/extending`.

The coordination loop
=====================

:meth:`MDO.coordinate() <DMDO.MDO.coordinate>` is the loop of :numref:`A-nhatc` written as a Python generator. It
yields two kinds of items:

* a **batch of subproblem tasks** (:class:`~DMDO.SubproblemTask`), which the driver solves and sends back as
  :class:`~DMDO.SubproblemResult` objects in any order. Gauss–Seidel yields one task at a time, Jacobi one batch with
  every subproblem;
* an **end-of-iteration marker** carrying the :class:`~DMDO.IterationRecord` of the outer iteration, and whether it
  was the last one.

.. code-block:: text

   driver                                   MDO.coordinate()
   ──────                                   ────────────────
   next()                       ───────▶    start; on_run_start
                                ◀───────    [task_1]                (Gauss–Seidel) or [task_1 .. task_M] (Jacobi)
   send([solve_task(task_1)])   ───────▶    merge result
                                  ...
                                ◀───────    IterationEnd(record, finished)
   send(None)                   ───────▶    next outer iteration (or on_run_end and stop)

:meth:`MDO.run() <DMDO.MDO.run>` drives the generator with an executor; :class:`~DMDO.Session` drives it one outer
iteration at a time (``step``) or lets the caller solve the tasks (``ask``/``tell``). All drivers execute the same
code, so they produce identical results.

A task is self-contained: it holds the subproblem (with its analyses and current targets), the multipliers and the
iteration number, and :func:`DMDO.solve_task` solves it. Because tasks and results are picklable, the same task can be
solved in-process, in a worker process or by a remote worker.

Executors
=========

``inprocess``
   Solves tasks in the calling process (default for ``mode="serial"``).
``process``
   A pool of worker processes, created on first use and reused across outer iterations (default for
   ``mode="parallel"``). Analyses and objectives must be importable, module-level callables.
Remote
   :class:`DMDO.service.RemoteExecutor` puts tasks on a queue served by remote workers (see `Service layer`_).
Custom
   Any object with ``map(fn, tasks)`` and ``close()``: a client for a cluster scheduler, a job queue, or another
   framework's parallel evaluator.

The sweep and the executor are independent choices: ``mode="serial"`` means Gauss–Seidel in-process,
``mode="parallel"`` means Jacobi in a process pool, and ``sweep=`` and ``executor=`` override either.

Observers
=========

Observers receive the following hooks, in this order within an outer iteration:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Hook
     - Called
   * - ``on_run_start(run)``
     - once, before the first outer iteration
   * - ``on_iteration_start(run, iteration)``
     - before the subproblems are solved
   * - ``on_resume(run, iteration)``
     - instead of ``on_iteration_start`` for the first iteration of a resumed run
   * - ``on_sweep_end(run, iteration)``
     - all subproblems solved and the inconsistencies computed, before the penalty update
   * - ``on_iteration_end(run, record)``
     - the iteration is final for reporting; ``record`` is an :class:`~DMDO.IterationRecord`
   * - ``on_multipliers_updated(run, iteration)``
     - after the multipliers and penalty weights were updated, before the stopping test
   * - ``on_run_end(run)``
     - once, after the last outer iteration

DMDO's own outputs are observers: :class:`~DMDO.CheckpointWriter` (``coord_chk.pkl``),
:class:`~DMDO.ConsoleReporter` (one progress line per iteration) and :class:`~DMDO.CoordinationHistoryRecorder`
(``Coordination_history.out``). User observers are called after them. Observers are passed to ``run`` or to the
session, never stored on the run, so checkpoints never have to pickle them.

Checkpoints and resume
======================

At the start of every outer iteration after the first, the whole run is pickled to ``coord_chk.pkl``. A run resumed
from that file (``Session.from_checkpoint`` or ``main`` with ``is_resume``) continues at the checkpointed iteration:
its observers receive ``on_resume`` instead of ``on_iteration_start`` for that iteration, and history rows written
after the checkpoint are replaced rather than duplicated. A resumed run ends with exactly the same result as an
uninterrupted one.

Sessions
========

:class:`~DMDO.Session` is the public entry point for other code. It builds the problem (``from_yaml``, ``from_dict``,
``from_checkpoint``, or around an existing :class:`~DMDO.MDO`), drives the loop (``run``, ``step``, ``ask``/``tell``,
``request_stop``) and reports a :class:`~DMDO.RunResult` whose ``to_dict()`` is JSON-ready. ``main({...})`` and the
YAML workflow of earlier releases are thin layers over the same building blocks (:func:`~DMDO.load_setup`,
:func:`~DMDO.build_mdo`).

Coordination policies
=====================

The ADMM coordinator owns the arithmetic of the multiplier update. *How* each update is scaled is a separate strategy,
a coordination policy, chosen per coordinator by the ``adaptation_policy`` option. The built-in ``none`` policy
applies the update of :doc:`background` unchanged. A custom policy subclasses
:class:`~DMDO.CoordinationPolicy` and returns, for each outer iteration, scale factors for the penalty growth
:math:`\beta`, the stall threshold :math:`\gamma` and the inner budget; it is registered under a name in
``DMDO.policies``.

Service layer
=============

``DMDO.service`` exposes sessions over HTTP and distributes subproblem solves to remote workers. It uses only the
Python standard library.

.. code-block:: text

   client (any language)  ── HTTP/JSON ──▶  DMDOService ── sessions (Session + EventLog observer)
                                                 │
                                                 └── TaskQueue ◀── claim / result ── dmdo worker --tags aero
                                                        ▲                     └──────── dmdo worker --tags structures
                                         RemoteExecutor │
                                    (placement: subproblem → tag)

* :class:`~DMDO.service.DMDOService` holds the sessions and a :class:`~DMDO.service.TaskQueue`; each session runs
  in a background thread, and an event-log observer feeds the server-sent event stream of the session.
* :class:`~DMDO.service.RemoteExecutor` turns each subproblem task into a job on the queue. ``placement`` pins
  subproblems to workers with a given tag, so each subproblem runs on the machine that holds its analysis code.
  Workers that disappear can be covered by a lease: an unanswered job is offered again after ``--lease`` seconds.
* Workers are plain loops (:func:`~DMDO.service.run_worker`): claim a job, call ``solve_task``, post the result.

Jobs and results are pickled, so the service, its workers and every client holding the token must trust each other.
The service listens on ``127.0.0.1`` by default and refuses other interfaces without a token; put a TLS-terminating
reverse proxy in front of it when traffic leaves a trusted network.
