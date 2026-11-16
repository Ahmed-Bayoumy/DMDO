Extension points
================

Registries
----------

.. autoclass:: DMDO.Registry
   :members: register, unregister, get, names, canonical_name

.. data:: DMDO.solvers

   Subproblem solver strategies (entry-point group ``dmdo.solvers``): ``mads`` (alias ``OMADS``), ``poll``,
   ``search``, ``scipy``.

.. data:: DMDO.executors

   Executor factories (entry-point group ``dmdo.executors``): ``inprocess`` (alias ``serial``), ``process``
   (alias ``parallel``).

.. data:: DMDO.blackboxes

   Analyses and objectives referenced by name from problem definitions (entry-point group ``dmdo.blackboxes``).

.. data:: DMDO.policies

   Coordination policies (entry-point group ``dmdo.policies``): ``none``.

.. autofunction:: DMDO.register_blackbox

Subproblem solvers
------------------

.. autoclass:: DMDO.BaseSolver
   :members: solve, prepare_settings, select_config

Coordination policies
---------------------

.. class:: DMDO.CoordinationPolicy

   Base class of the coordination policies: decides how each multiplier update of the coordinator is scaled.
   Policies hold no state of their own; they read and write the coordinator they are given.

   .. method:: compute_scales(coord, iteration)

      Return ``(beta_scale, gamma_scale, budget_scale, event)`` for this outer iteration: the factor applied to the
      exponent of the penalty growth :math:`\beta`, the factor applied to the stall threshold :math:`\gamma`, the
      factor applied to the inner evaluation budget of the next outer iteration, and a flag that custom policies
      return as ``False``. All factors are ``1`` in the base class.

   .. method:: multiplier_scales(coord, iteration)

      Called by the coordinator before each update: evaluates :meth:`compute_scales` and records the applied
      factors. Returns ``(beta_scale, gamma_scale, event)``.

.. class:: DMDO.NeutralPolicy

   The ``none`` policy: the update of :doc:`../background`, unchanged.

.. function:: DMDO.policy_for(coord)

   The policy of a coordinator: the one named by its ``adaptation_policy`` option, ``none`` by default.

Built-in observers
------------------

.. autoclass:: DMDO.CheckpointWriter

.. autoclass:: DMDO.ConsoleReporter

.. class:: DMDO.CoordinationHistoryRecorder

   Writes ``Coordination_history.out``: a header at the first outer iteration, then one row per outer iteration
   (:doc:`../user_guide/outputs`).
