MDO architectures
=================

The ``architecture`` key of the ``MDO`` block selects how the problem is solved. Its value is a path of nested
YAML keys and nothing else; the options stay in the blocks that already exist.

.. list-table::
   :header-rows: 1
   :widths: 30 70
   :class: dmdo-ref

   * - ``architecture``
     - Method
   * - ``DIST: {IDF: NHATC}``
     - Distributed non-hierarchical ATC, augmented-Lagrangian coordination (:doc:`../background`). **Default.**
   * - ``DIST: {IDF: ATC}``
     - Distributed hierarchical ATC: the links must form a tree below the ``is_main`` subproblem; solved level by
       level with the same ADMM coordinator.
   * - ``MON: MDF``
     - Monolithic MDF: every subproblem is a standalone problem whose MDA is converged at each evaluation.

In YAML::

  MDO:
    architecture:
      MON: MDF

  MDO:
    architecture:
      DIST:
        IDF: NHATC        # or ATC

Any other value (for example the plain string ``IDF`` of older problem files) logs one warning at the start of the
run and selects the default ``DIST: IDF: NHATC``, which runs exactly as before. Leaving the key out selects the
default without a warning. ``MON: IDF`` and BLISS are recognised but not supported; naming them is an error.

Hierarchical ATC
----------------

``DIST: {IDF: ATC}`` is analytical target cascading :cite:p:`michelena1999system,kim2003target` on a tree of
subproblems. Nothing new is added to the problem
definition; the tree comes from the blocks that already exist:

* The subproblem with ``is_main: True`` is the system-level element, the root of the tree.
* The subproblems linked to the root (through the ``link`` entries of their variables) are its children, the
  subproblems linked to those are the next level, and so on. Every target/response pair therefore connects a parent
  to one of its children.
* The links must form a tree. A cycle, such as three subproblems that are all linked to each other, or a link
  between two siblings, is rejected when the problem is built, with the offending link in the message; use
  ``DIST: {IDF: NHATC}`` for such problems. A subproblem that is not linked to the tree is rejected as well.

The coordination is the ADMM coordinator of the ``coord`` block (``type: ADMM``) that the subproblems name in their
``coordinator`` key: augmented-Lagrangian relaxation ``v q + |w q|^2`` of the parent-child pairs and method-of-
multipliers updates (``beta``, ``gamma``, ``M_update_scheme``, ``budget``, ...). This is the alternating-direction
ATC of Tosserams et al. :cite:p:`tosserams2006augmented`: one pass over the hierarchy per outer iteration, then the multiplier update. The
subproblem blocks are unchanged; their ``budget`` and solver ``tol`` govern each solve.

Each outer iteration solves the tree level by level from the root down. The subproblems of one level share no pairs,
so they are solved as one batch from the same coordinator state (in parallel with ``mode: parallel`` or a parallel
executor) and their results are merged before the next level, which sees the targets its parents just set. The
``sweep`` option of :meth:`DMDO.MDO.run` does not apply. With :class:`DMDO.Session`, every ``ask()`` returns the
tasks of one level. The tree and the solve order are written to the log, for example
``DIST: IDF: ATC hierarchy: [SP2] -> [SP1, SP3]``.

The SBJ formulations (a) (``tests/SBJ/SBJ_a.yaml``, aircraft at the root with three children) and (d)
(``SBJ_d.yaml``, aerodynamics at the root) are trees; (b) and (c) contain cycles and need NHATC.

.. note::

   By default the coordinator updates the multipliers twice per outer iteration (kept for reproducibility of
   earlier runs). For ATC set ``single_multiplier_update: true`` in the ``coord`` block, as the SBJ files do: with the
   double update the penalty weights can grow until the subproblem solves can no longer reduce the inconsistencies.

Monolithic MDF
--------------

Each subproblem of the ``MDO.subproblems`` list is solved on its own; there is no coordinator, so ``coord`` blocks and
the ``coordinator`` keys are not needed (and are ignored if present). A subproblem's variables may not be linked to
another subproblem (``link`` must be ``null``): put every analysis of the problem into the subproblem's MDA.

* The subproblem's ``vars`` are the design variables; its solver settings (``solver``, ``scipy`` or
  ``configurations``, ``budget``) drive the optimisation, as in the distributed case.
* At every evaluation the subproblem's MDA converges its feedback loops (below), then ``opt(x, y)`` returns the
  objective and the constraints ``g <= 0``. The bounds of the MDA ``responses`` (except ``dummy`` and constants) are
  added as constraints. An evaluation whose MDA does not converge is reported as failed (objective ``+inf``).
* :meth:`DMDO.MDO.run` returns ``{"architecture": "MON: MDF", "subproblems": [...]}`` with, per subproblem, the
  best ``x`` and ``y``, ``fmin``, ``hmin``, ``feasible``, the number of evaluations and the MDA iterations and
  analysis evaluations. Every evaluation is written to ``<post_dir>/MDF_history.csv``
  (``MDF_SP<i>_history.csv`` with several subproblems).
* There is no outer loop to step through, so :class:`DMDO.Session` (ask/tell) does not apply.

``examples/sellar_mdf.yaml`` and ``examples/sellar_mdf.py`` solve the Sellar problem this way
(f* = 3.18339 at x = 0, z = (1.9776, 0)).

Feedback detection and MDA solvers
----------------------------------

Inside one MDA block, analysis A feeds analysis B when a variable key in A's ``outputs`` is also in B's ``inputs``.
The match is by variable key (the variable's unique index), never by name. The strongly connected components of
that graph are the feedback loops: two or more analyses that depend on each other, or an analysis that reads its own
output (for example ``inputs: [v3]`` and ``outputs: [v3]``). Only those loops are converged by the MDA solver; the
other analyses are feed-forward and run once, in dependency order, whatever their order in ``analyses``.

The solver is chosen per MDA block with the optional ``solver`` key. It is used only by ``MON: MDF`` (and ignored,
with an info line, otherwise); without the key, or for an MDA without feedback, nothing changes.

.. list-table::
   :header-rows: 1
   :widths: 18 82
   :class: dmdo-ref

   * - ``solver``
     - Method
   * - ``GS`` (default)
     - Nonlinear block Gauss-Seidel: each analysis uses the latest outputs of the analyses before it.
   * - ``Jacobi``
     - Nonlinear block Jacobi: every analysis of a sweep uses the outputs of the previous sweep.
   * - ``Newton``
     - Newton's method on the residual ``R(y) = y - F(y)`` of the loop couplings, forward-difference Jacobian and a
       backtracking step. Fewest iterations; each one costs ``n + 1`` sweeps for ``n`` scalar couplings.

Options are given as a mapping, ``solver: {name: Newton, tol: 1.0e-12, maxiter: 50}``:

.. list-table::
   :header-rows: 1
   :widths: 22 18 60
   :class: dmdo-ref

   * - Option
     - Default
     - Meaning
   * - ``tol``
     - ``1e-10``
     - Convergence on the scaled residual ``max_i |y_i - F_i(y)| / max(1, |y_i|)``.
   * - ``maxiter``
     - ``100``
     - Iteration limit; the MDA is reported as not converged when it is reached.
   * - ``relaxation``
     - ``1.0``
     - GS and Jacobi: ``y <- y + relaxation (F(y) - y)``.
   * - ``fd_step``, ``max_backtracks``
     - ``1e-7``, ``8``
     - Newton: relative finite-difference step and the number of step halvings.

Example (Sellar, both analyses in one MDA)::

  MDA:
    MDA1:
      index: 1
      nAnalyses: 2
      analyses: [1, 2]           # DA1: y1 = f(x, z, y2), DA2: y2 = f(z, y1) -> loop {DA1, DA2}
      variables: [v0, v1, v2]
      responses: [v3, v4]
      solver: Newton

Further solvers can be registered in :data:`DMDO.registry.mda_solvers` (subclasses of
:class:`DMDO.analysis.mda_solvers.MDASolver`) or through the ``dmdo.mda_solvers`` entry point.
