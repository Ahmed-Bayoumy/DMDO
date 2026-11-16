Outputs
#######

A run returns its results in memory (:class:`~DMDO.RunResult`) and writes them to its output folder,
``<working_dir>/<mdo_name>_post``.

Result objects
==============

:class:`~DMDO.RunResult` (``session.result()``):

.. list-table::
   :header-rows: 1
   :widths: 22 78
   :class: dmdo-ref

   * - Field
     - Meaning
   * - ``status``
     - ``created``, ``running``, ``finished``, ``failed`` or ``closed``.
   * - ``stop_reason``
     - Why the run ended (``"Iteration budget exhausted"``, ``"Max inconsitency is below stopping threshold"``,
       ``"No progress after several iterations"``, ``"Stopped on request"``).
   * - ``iterations``
     - Number of completed outer iterations.
   * - ``objective``
     - Objective of the master subproblem at the last design.
   * - ``max_inconsistency``
     - Largest scaled inconsistency between linked copies, :math:`\|\mathbf{q}\|_\infty`.
   * - ``feasible``
     - Whether the master subproblem satisfies its constraints (violation :math:`\le 10^{-12}`).
   * - ``variables``
     - Every variable copy, keyed ``"<name>_<subproblem>"``.
   * - ``history``
     - One :class:`~DMDO.IterationRecord` per outer iteration.
   * - ``post_dir``
     - The output folder.

``RunResult.to_dict()`` returns the same content as JSON-ready values.

:class:`~DMDO.IterationRecord` (``session.step()``, observers): ``iteration``, ``objective``,
``max_inconsistency``, ``max_penalty_weight``, ``variables_change`` (Euclidean norm of the change of all copies) and
``worst_link`` (index of the link with the largest inconsistency).

All the ``.out`` files are comma-separated tables; the post-processing dashboard (:doc:`../tutorials/dashboard`)
plots any of their columns interactively.

The output folder
=================

.. code-block:: text

   <mdo_name>_post/
   ├── Coordination_history.out
   ├── DMDO.log
   ├── coord_chk.pkl
   └── coordination_summary_post/
       ├── SP_1/
       │   ├── SP_1_0.out, SP_1_1.out, ...
       │   └── OMADS.log
       └── SP_2/ ...

``Coordination_history.out``
   One row per outer iteration (comma-separated, columns padded with spaces). Load it with
   ``pandas.read_csv(path, skipinitialspace=True)`` and strip the column names.
``DMDO.log``
   The log of the run: problem setup, the progress lines, stopping messages.
``coord_chk.pkl``
   Checkpoint written at the start of every outer iteration after the first; resume with
   ``Session.from_checkpoint`` (:doc:`../tutorials/session`).
``SP_<j>/SP_<j>_<t>.out``
   Every evaluation of subproblem :math:`j` during outer iteration :math:`t`.
``SP_<j>/OMADS.log``
   Log of the OMADS solver of subproblem :math:`j`.

``Coordination_history.out``
============================

.. list-table::
   :header-rows: 1
   :widths: 30 70
   :class: dmdo-ref

   * - Column
     - Meaning
   * - ``Time``
     - Wall-clock time at the end of the outer iteration.
   * - ``Iteration #``
     - Outer iteration.
   * - ``Max. inconsistency``
     - :math:`\|\mathbf{q}\|_\infty` after the inner loop.
   * - columns 4 and 5
     - Reserved flags (0).
   * - ``Objective``
     - Objective of the master subproblem.
   * - ``Status``
     - ``Feasible`` or ``Infeasible`` (constraints of the master subproblem); ``Error`` if no objective value.
   * - ``Variables change``
     - Euclidean norm of the change of all copies during the iteration.
   * - ``max(w)``
     - Largest penalty weight.
   * - ``Coupling_with_qmax``
     - The link with the largest inconsistency, as ``<name>_<subproblem> to <name>_<subproblem>``.
   * - ``hmin_<j>``
     - Constraint violation of subproblem :math:`j`.
   * - ``<name>_<j>``
     - Value of every variable copy.

``SP_<j>_<t>.out``
==================

One row per evaluation of the subproblem: ``Time``, ``Iteration #``, ``Evaluation #``, ``Max. inconsistency``,
``Penalty`` (:math:`\phi`), ``Objective`` (the local objective without the penalty), ``Coupling Status`` (feasibility
of the local constraints), followed by the design variables and the responses of the subproblem.
