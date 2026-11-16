Problem objects
===============

These classes build a problem in Python; :doc:`../tutorials/sellar` shows them in use. The YAML builder creates the
same objects.

.. class:: DMDO.variableData(name, sp_index, index, link, coupling_type, baseline, lb, ub, value, scaling, dim, type)

   One variable copy: its name, the subproblem that holds it (``sp_index``), its position in the global list
   (``index``, starting at 1), the subproblem it is linked to (``link``), its coupling type
   (:class:`~DMDO.COUPLING_TYPE`), bounds, baseline and current value, scale, dimension and type (``"REAL"``).

.. class:: DMDO.DA(inputs, outputs, blackbox, links, coupling_type)

   A disciplinary analysis: ``blackbox`` computes ``outputs`` from ``inputs`` (lists of :class:`~DMDO.variableData`).
   ``blackbox`` is a callable working on lists, or a :class:`~DMDO.Discipline`.

.. class:: DMDO.MDA(nAnalyses, analyses, variables, responses)

   The analyses evaluated, in order, for each design of a subproblem.

.. class:: DMDO.ADMM(nsp, beta, budget, index_of_master_SP, display, scaling, mode, M_update_scheme, store_q_io=False, index=None, gamma=0.5)

   The augmented Lagrangian coordinator of :doc:`../background`: ``nsp`` subproblems, penalty growth ``beta``, stall
   threshold ``gamma``, update scheme ``M_update_scheme`` (:class:`~DMDO.w_scheme`), outer budget ``budget`` and the
   master subproblem ``index_of_master_SP``. ``scaling`` is a constant or a list with one scale per variable copy.

.. class:: DMDO.SubProblem(nv, index, vars, resps, is_main, analysis, coordination, opt, fmin_nop, budget, display, psize, pupdate, solver, freal=None, scipy=None, conf=None)

   One subproblem: its design variables ``vars`` and responses ``resps``, its ``analysis`` (:class:`~DMDO.MDA`),
   its ``coordination`` (:class:`~DMDO.ADMM`), the objective callable ``opt``, and the inner ``solver`` with its
   evaluation ``budget`` and settings (``scipy`` for ``solver="scipy"``, ``conf`` for the OMADS solvers).

Enumerations
------------

.. class:: DMDO.COUPLING_TYPE

   ``SHARED``, ``FEEDFORWARD``, ``FEEDBACK``, ``UNCOUPLED``, ``CONSTANT``, ``DUMMY``.

.. class:: DMDO.w_scheme

   Penalty-weight update schemes: ``NORMAL``, ``MEDIAN``, ``MEAN``, ``MAX``, ``NORM_2``, ``RANK``, ``NO_UPDATE``.

.. class:: DMDO.PSIZE_UPDATE

   How the OMADS frame size carries over between outer iterations: ``DEFAULT``, ``SUCCESS``, ``MAX``, ``LAST``.
