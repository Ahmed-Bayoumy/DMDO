Testing the MDO architectures
=============================

This page records how the architectures added next to the default NHATC coordination were verified:
hierarchical ATC (``DIST: {IDF: ATC}``, :cite:p:`tosserams2006augmented`), monolithic MDF (``MON: MDF``) and the MDA solvers that MDF uses
(:doc:`user_guide/architectures`). The NHATC benchmark results are on :doc:`benchmarking` and are unchanged.

The unit and integration tests are in ``tests/test_mdo_architectures.py``. The SBJ runs below are single runs from
the Table 1 reference design of Tosserams et al. :cite:p:`tosserams2010nonhierarchical`, not a multi-start
campaign.

Default architecture unchanged
------------------------------

The default ``DIST: {IDF: NHATC}`` path, and every older problem file whose ``architecture`` value is not a known
path (for example ``IDF``), runs exactly as before: all seven recorded golden traces (``tests/golden``) are
reproduced bit for bit with the new architectures in place. The traces were compared, not re-recorded.

MDA feedback detection and solvers
----------------------------------

The Sellar analyses :cite:p:`sellar1996response` at :math:`x = 1`, :math:`z = (5, 2)`, both in one MDA, form one
feedback loop (:math:`y_1` from DA1 feeds DA2, :math:`y_2` from DA2 feeds DA1). Each solver converges it to the same
couplings, :math:`y = (25.58830, 12.05849)`, with tolerance :math:`10^{-10}` from :math:`y = (1, 1)`:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25
   :class: dmdo-ref

   * - Solver
     - Iterations
     - Analysis evaluations
     - Final scaled residual
   * - ``GS``
     - 8
     - 16
     - :math:`5.2\times10^{-12}`
   * - ``Jacobi``
     - 14
     - 28
     - :math:`2.1\times10^{-11}`
   * - ``Newton``
     - 4
     - 22
     - :math:`1.6\times10^{-11}`

Further tests check that:

* an analysis that reads its own output is a feedback loop (:math:`y = \cos y` converges to 0.7390851 with each
  solver);
* feed-forward analyses run once, in dependency order, even when listed in the wrong order;
* listing the loop's analyses in either order gives the same couplings;
* Newton converges a vector coupling :math:`y = A y + b`;
* a loop that does not converge within ``maxiter`` is reported as not converged.

Monolithic MDF on Sellar
------------------------

``examples/sellar_mdf.yaml``: one standalone subproblem whose MDA holds both analyses, SLSQP as the optimizer, from
:math:`x = 1`, :math:`z = (5, 2)`. The best known solution is :math:`f^\ast = 3.18339` at :math:`x = 0`,
:math:`z = (1.9776, 0)` :cite:p:`tedford2010benchmarking`.

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20
   :class: dmdo-ref

   * - MDA solver
     - :math:`f`
     - :math:`z_1`
     - Optimizer evaluations
     - MDA iterations (total)
   * - ``GS``
     - 3.183394
     - 1.977639
     - 69
     - 181
   * - ``Jacobi``
     - 3.183394
     - 1.977639
     - 36
     - 208
   * - ``Newton``
     - 3.183394
     - 1.977639
     - 31
     - 69

All three end feasible with :math:`x = 0`, :math:`z_2 = 0`, :math:`y_1 = 3.16` and :math:`y_2 = 3.75528`. With OMADS
(``solver: mads``, 2000 evaluations, Newton MDA) instead of SLSQP the run reaches :math:`f = 3.183398` at
:math:`z_1 = 1.97764`.

The Newton solver ends every MDA with a fixed-point update :math:`y \leftarrow F(y)`, as Gauss–Seidel and Jacobi do.
Without it, the optimizer's finite-difference steps (about :math:`10^{-8}`) changed the couplings by less than the
MDA tolerance, Newton returned the warm-start couplings unchanged, the gradient vanished and SLSQP stopped at the
start point (:math:`f = 23.04`).

The tests also check that subproblems linked to each other are rejected under MDF, that MDF has no ask/tell loop,
and that an ``MDA.solver`` key is ignored by the other architectures.

Hierarchical ATC
----------------

Tree construction
~~~~~~~~~~~~~~~~~

The tree is built from the subproblem links and the ``is_main`` subproblem. The tests cover a star, a chain rooted
at a middle subproblem (the SBJ (d) layout), a four-level chain and list links. They also check that DMDO rejects
a cycle (the geometric-programming problem's links 1–2, 1–3, 2–3), a sibling link, a subproblem outside the tree,
two ``is_main`` subproblems and a link to a subproblem that is not part of the run.

Analytical star
~~~~~~~~~~~~~~~

A root with two children: the root owns copies of :math:`a` and :math:`b` and minimises :math:`(a - b)^2`; child 2
minimises :math:`(a - 3)^2` with its copy of :math:`a`; child 3 minimises :math:`(b + 1)^2` with its copy of
:math:`b`. The optimum is :math:`a^\ast = 5/3`, :math:`b^\ast = 1/3`, :math:`f^\ast = 16/3`. SLSQP in every
subproblem; coordinator :math:`\beta = 1.2`, :math:`\gamma = 0.8`, ``single_multiplier_update: true``.

* ATC solves ``[SP1] -> [SP2, SP3]`` and reaches :math:`a = 1.666667`, :math:`b = 0.333333`,
  :math:`f = 5.3333333` with a maximum inconsistency of :math:`5\times10^{-10}` in 50 outer iterations.
* On this star the level order equals the Gauss–Seidel order and the siblings share no pairs, so ATC and NHATC give
  identical runs.
* Serial and parallel (process pool) ATC runs give identical results; with :class:`~DMDO.Session`, each
  ``ask()`` returns one level (``[1]``, then ``[2, 3]``).

With the coordinator's default double multiplier update per outer iteration, the same problem stalls at
:math:`f \approx 5.37`: the penalty weight of one pair grows past :math:`10^{22}` until SLSQP can no longer reduce
its inconsistency (about :math:`4\times10^{-5}`). Hence the recommendation to set ``single_multiplier_update: true``
for ATC.

Supersonic business jet
~~~~~~~~~~~~~~~~~~~~~~~

The SBJ formulations (a) and (d) of Tosserams et al. :cite:p:`tosserams2010nonhierarchical` (Fig. 6) are trees and
were solved with ``DIST: {IDF: ATC}`` and the unchanged settings of ``tests/SBJ/SBJ_a.yaml`` and ``SBJ_d.yaml``
(SLSQP subproblems, :math:`\beta = 1.03`, :math:`\gamma = 0.75`, stop when the maximum inconsistency is below
:math:`10^{-3}`). The final design was re-analysed with a converged MDA; the paper's MDF optimum is
:math:`W_t = 34.3\times10^3` lb.

.. list-table::
   :header-rows: 1
   :widths: 31 23 23 23
   :class: dmdo-ref

   * - Quantity
     - (a)
     - (d)
     - Paper (Table 1)
   * - Tree (solve order)
     - ``[SP1] -> [SP2, SP3, SP4]``
     - ``[SP2] -> [SP1, SP3]``
     -
   * - Outer iterations
     - 148
     - 149
     -
   * - Wall time
     - 752 s
     - 475 s
     -
   * - :math:`W_t` (MDA of the design)
     - 34,282 lb
     - 34,499 lb
     - 34,300 lb
   * - Range
     - 1977.3 nm
     - 2007.9 nm
     - :math:`\geq` 2000 nm
   * - Largest constraint
     - :math:`1.1\times10^{-2}`
     - :math:`8.7\times10^{-7}`
     - :math:`\leq 0`

Both runs reach the paper's weight to within 0.6 %. Run (d) ends feasible. Run (a) stops on the
:math:`10^{-3}` inconsistency tolerance while its objective is still changing (about 75 lb per iteration over the
last iterations); its remaining inconsistency leaves the re-analysed range 1 % short of 2000 nm. A tighter
``inc_stop`` would be expected to close that gap at the cost of more iterations.
