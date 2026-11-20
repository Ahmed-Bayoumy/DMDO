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

.. class:: DMDO.SubProblem(nv, index, vars, resps, is_main, analysis, coordination, opt, fmin_nop, budget, display, psize, pupdate, solver, freal=None, scipy=None, conf=None, start_point="warm")

   One subproblem: its design variables ``vars`` and responses ``resps``, its ``analysis`` (:class:`~DMDO.MDA`),
   its ``coordination`` (:class:`~DMDO.ADMM`), the objective callable ``opt``, and the inner ``solver`` with its
   evaluation ``budget`` and settings (``scipy`` for ``solver="scipy"``, ``conf`` for the OMADS solvers).
   ``start_point`` chooses where every inner solve starts (see :meth:`set_start_point`).

   .. method:: set_start_point(start_point="warm")

      ``"warm"`` (default) starts each solve from the last solution, ``"initial"`` from the values the subproblem
      was built with, every time. A mapping ``{variable: value}`` starts from these values; design variables not
      listed keep their initial values. A variable is given by its index (:attr:`variableData.index`, which is
      what a YAML variable key such as ``v27`` is translated to) or by its name; a name shared by several scalar
      variables takes one value for all of them or a list with one value each, and an index entry wins over a
      name entry. Vector variables take a list of ``dim`` values. Unknown variables raise :class:`ValueError`.
      Only the start point is fixed: the frame size still follows ``pupdate``. The initial values are those at the
      time of the call, so call it before the run.

      .. code-block:: python

         sp1.set_start_point("initial")
         sp4.set_start_point({"Sref": 800.0})                          # by name
         first_t = next(v for v in sp4.vars if v.name == "t")
         sp4.set_start_point({first_t.index: 3.0})                     # by index (1-based position in "variables")

Enumerations
------------

.. class:: DMDO.COUPLING_TYPE

   ``SHARED``, ``FEEDFORWARD``, ``FEEDBACK``, ``UNCOUPLED``, ``CONSTANT``, ``DUMMY``.

.. class:: DMDO.w_scheme

   Penalty-weight update schemes: ``NORMAL``, ``MEDIAN``, ``MEAN``, ``MAX``, ``NORM_2``, ``RANK``, ``NO_UPDATE``.

.. class:: DMDO.PSIZE_UPDATE

   How the OMADS frame size carries over between outer iterations: ``DEFAULT``, ``SUCCESS``, ``MAX``, ``LAST``.
