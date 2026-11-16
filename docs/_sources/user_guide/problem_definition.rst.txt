Problem definition
##################

A DMDO problem is described by a dictionary, usually read from a YAML file with :func:`~DMDO.load_setup` and built
by :class:`~DMDO.Session` (or :func:`~DMDO.build_mdo`, :func:`~DMDO.main`). This page lists every section and key.
``examples/basic_problem.yaml`` is a complete example (:doc:`../getting_started`).

.. important::

   Three settings have defaults that are easy to get wrong:

   * declare real variables with the type ``REAL``. OMADS optimizes variables of any other type, including the
     short form ``R``, on an integer lattice;
   * ``gamma`` defaults to ``0``: every link whose inconsistency is not exactly zero then counts as stalled and its
     penalty weight grows at every outer iteration. Values around ``0.2``–``0.5`` are usual;
   * ``scaling`` is used only when it is a float (``0.1``, not ``10``); otherwise the scale factors are computed from
     the variable bounds.

``variables``
=============

One entry per variable **copy**; each entry is a list:

.. code-block:: yaml

   variables:
     #    [name, subproblem, link, coupling, lb,  baseline, ub,  dim, type]
     v0:  ["u",  1,          2,    s,        0.,  1.,       10., 1,   REAL]
     v1:  ["v",  1,          None, u,        0.,  1.,       10., 1,   REAL]

.. list-table::
   :header-rows: 1
   :widths: 16 84
   :class: dmdo-ref

   * - Item
     - Meaning
   * - ``name``
     - Variable name. Copies of the same quantity in different subproblems share the name.
   * - ``subproblem``
     - Index of the subproblem that holds this copy.
   * - ``link``
     - Index of the subproblem holding the copy this one must agree with (a list for several), or ``None`` for a
       local variable.
   * - ``coupling``
     - ``s`` shared design variable, ``ff`` feed-forward coupling, ``fb`` feedback coupling, ``u`` local
       (uncoupled), ``CP`` constant parameter, ``dummy`` placeholder.
   * - ``lb``, ``ub``
     - Bounds. They also set the default scale of the inconsistencies (:doc:`../background`).
   * - ``baseline``
     - Starting value.
   * - ``dim``
     - Dimension (vector-valued variables have one inconsistency per component).
   * - ``type``
     - ``REAL`` (recommended to give explicitly) or ``INTEGER``. Default: ``R``.

``DA``: disciplinary analyses
=============================

.. list-table::
   :header-rows: 1
   :widths: 22 78
   :class: dmdo-ref

   * - Key
     - Meaning
   * - ``index``
     - Analysis index, referenced from ``MDA``.
   * - ``inputs``
     - Variable keys the analysis reads, in the order the callable receives them.
   * - ``outputs``
     - Variable keys the analysis computes, in the order the callable returns them.
   * - ``blackbox``
     - Name of the analysis: a key of the session's ``blackboxes`` mapping, a registered name, or an import path
       ``"package.module:attr"``.
   * - ``type``
     - ``callable``.
   * - ``links``, ``coupling_type``
     - Descriptive: the subproblem the outputs go to and the coupling type (``ff``, ``fb``, ``s``, ``un``).

``MDA``: analyses per subproblem
================================

.. list-table::
   :header-rows: 1
   :widths: 22 78
   :class: dmdo-ref

   * - Key
     - Meaning
   * - ``index``
     - MDA index, referenced from the subproblem.
   * - ``nAnalyses``, ``analyses``
     - Number and indices of the analyses evaluated, in order, for each design of the subproblem.
   * - ``variables``
     - Variable keys the analyses read from the subproblem.
   * - ``responses``
     - Variable keys the analyses compute.

``coord``: the coordinator
==========================

.. list-table::
   :header-rows: 1
   :widths: 24 16 60
   :class: dmdo-ref

   * - Key
     - Default
     - Meaning
   * - ``index``
     - (required)
     - Coordinator index, referenced from the subproblems and ``MDO``.
   * - ``type``
     - (required)
     - ``ADMM``, the only coordinator type.
   * - ``budget``
     - (required)
     - Maximum number of outer iterations.
   * - ``nsp``
     - (required)
     - Number of subproblems.
   * - ``index_of_master_SP``
     - (required)
     - Subproblem that holds the system objective.
   * - ``beta``
     - ``1.8``
     - Penalty growth factor :math:`\beta > 1`.
   * - ``gamma``
     - ``0.0``
     - Stall threshold :math:`\gamma`: a link stalls when :math:`|q^{t}| > \gamma |q^{t-1}|`.
   * - ``M_update_scheme``
     - (required)
     - Which stalled links grow: ``normal``, ``median``, ``max`` or ``rank`` (:doc:`../background`).
   * - ``scaling``
     - bounds
     - A float scales every inconsistency by that constant; any other value uses :math:`0.1/(u-\ell)` per copy.
   * - ``mode``
     - (required)
     - ``serial`` or ``parallel``; the run's ``mode`` argument takes precedence.
   * - ``display``, ``store_q_io``
     - (required)
     - Console output of the coordinator; whether to store the inconsistency history.
   * - ``adaptation_policy``
     - ``none``
     - Name of the coordination policy (:doc:`../tutorials/extending`).

``subproblem``
==============

.. list-table::
   :header-rows: 1
   :widths: 22 16 62
   :class: dmdo-ref

   * - Key
     - Default
     - Meaning
   * - ``index``, ``nv``
     - (required)
     - Subproblem index and number of design variables.
   * - ``vars``
     - (required)
     - Variable keys of the design vector ``x`` passed to the objective.
   * - ``resps``
     - (required)
     - Variable keys of the responses ``y`` passed to the objective.
   * - ``is_main``
     - (required)
     - ``True`` for the master subproblem.
   * - ``MDA``, ``coordinator``
     - (required)
     - Indices of the subproblem's MDA and coordinator.
   * - ``opt``, ``type``
     - (required)
     - Name of the objective callable (resolved like ``blackbox``) and ``callable``.
   * - ``solver``
     - ``mads``
     - Inner solver: ``mads``, ``poll``, ``search``, ``scipy`` or a registered strategy.
   * - ``budget``
     - (required)
     - Evaluation budget of each inner solve.
   * - ``scipy``
     - SLSQP
     - For ``solver: scipy``: ``{method, options, tol, is_con}``. ``is_con: true`` passes the constraints to the
       optimizer; otherwise they are penalized.
   * - ``configurations``
     - none
     - Name of an entry of the ``CONFIG`` section with OMADS settings for this subproblem.
   * - ``psize``, ``pupdate``
     - ``1.0``, ``last``
     - Initial OMADS frame size, and how it carries over between outer iterations (``last``, ``success``,
       ``max``, ``default``).
   * - ``fmin_nop``
     - ``inf``
     - Initial best objective.
   * - ``freal``
     - none
     - A reference value of the objective, used to report the relative objective error of the master subproblem.
   * - ``display``
     - ``False``
     - Console output of the inner solver.

``CONFIG``: OMADS settings
==========================

Named blocks of OMADS options, referenced by ``configurations``. Each has an ``options`` part (``seed``, ``budget``,
``tol``, ``psize_init``, ``opportunistic``, ``rich_direction``, ``check_cache``, ``store_cache``, ...) and a
``search`` part (``type``, ``s_method``, ``ns``); see the OMADS documentation :cite:p:`OMADS_AB` for their meaning.

.. code-block:: yaml

   CONFIG:
     CSP1:
       options: {seed: 12345, budget: 250, tol: 1.0e-9, psize_init: 2.0, opportunistic: False,
                 check_cache: True, store_cache: True, rich_direction: True}
       search: {type: sampling, s_method: ACTIVE, ns: 10}

``MDO``: the run
================

.. list-table::
   :header-rows: 1
   :widths: 22 22 56
   :class: dmdo-ref

   * - Key
     - Default
     - Meaning
   * - ``coordinator``, ``subproblems``
     - (required)
     - Coordinator index and the list of subproblem indices, in sweep order.
   * - ``responses``
     - (required)
     - Variable keys of all analysis responses.
   * - ``inc_stop``
     - ``1e-9``
     - Stop when the maximum inconsistency is below this value and no longer changes (must be a float).
   * - ``noprogress_stop``
     - ``100``
     - Stop when the maximum inconsistency has not improved for this many outer iterations.
   * - ``display``
     - ``True``
     - One progress line per outer iteration.
   * - ``stop``
     - ``"Iteration budget exhausted"``
     - Initial stopping message, reported if no criterion triggers.
   * - ``fmin``, ``hmin``, ``tab_inc``
     - ``inf``, ``inf``, ``[]``
     - Initial values of the run's records.
   * - ``architecture``
     - ``IDF``
     - Informational; DMDO solves the distributed NHATC formulation of :doc:`../background`.

``Sets`` and ``USER``
=====================

``Sets`` defines named value sets for categorical and discrete variables. ``USER`` holds free-form user data,
attached to the built problem.
