Extending DMDO
##############

DMDO talks to analyses, optimizers and execution back ends through small interfaces (:doc:`../implementation`).
This tutorial shows each extension point with code that runs as shown.

Analyses as disciplines
=======================

Any object with a ``compute(inputs) -> outputs`` method, working on dictionaries keyed by variable name, can be used
wherever an analysis callable is accepted. This is the natural form of an adapter around a simulation code or a
model from another framework:

.. code-block:: python

   class Aerodynamics:
       """Adapter around an external aerodynamic model."""

       def __init__(self, model):
           self.model = model

       def compute(self, inputs):                 # {"span": ..., "twist": ..., "weight": ...}
           lift, drag = self.model.evaluate(inputs["span"], inputs["twist"], inputs["weight"])
           return {"lift": lift, "drag": drag}    # must contain every output of the analysis

The outputs are matched to the analysis's ``outputs`` by name, so the order of the dictionary does not matter. For
parallel or remote execution the adapter must be picklable: build heavy models lazily, or rebuild them in
``__setstate__``.

Frameworks such as OpenMDAO :cite:p:`gray2019openmdao` and GEMSEO :cite:p:`gallard2018gems` are examples of tools
whose models can be wrapped this way, or which can drive DMDO through the :doc:`session`; DMDO itself depends on
neither.

Naming analyses and objectives
==============================

The ``blackbox:`` and ``opt:`` names of a problem definition are resolved, in order, from:

#. the ``blackboxes`` mapping given to the session (or to :func:`~DMDO.main`);
#. the global registry, filled with :func:`~DMDO.register_blackbox` or through the ``dmdo.blackboxes`` entry-point
   group of an installed package;
#. an import path ``"package.module:attribute"`` written directly in the YAML file.

.. code-block:: python

   from DMDO import register_blackbox

   register_blackbox("aero", Aerodynamics(model))       # now usable as  blackbox: aero

A custom subproblem solver
==========================

A solver strategy subclasses :class:`~DMDO.BaseSolver` and implements ``solve(request) -> SolverOutcome``. The
request carries the penalized subproblem objective, the start point, the bounds, the evaluation budget and a seed.
``request.objective(x)`` returns ``[f + penalty, [g1, g2, ...]]``, the penalized objective and the local constraints
:math:`g \le 0`.

.. code-block:: python

   import numpy as np
   from DMDO import BaseSolver, SolverOutcome, solvers

   @solvers.register("random_search")
   class RandomSearch(BaseSolver):
       """Uniform random sampling inside the bounds; keeps the best point that satisfies the local constraints."""

       def solve(self, request):
           rng = np.random.default_rng(request.seed + request.iteration)
           lb, ub = np.asarray(request.lb, dtype=float), np.asarray(request.ub, dtype=float)

           def merit(x):
               f, g = request.objective(list(x))
               return f if max(g, default=0.0) <= 0.0 else np.inf

           best_x = np.asarray(request.x0, dtype=float)
           best_f = merit(best_x)
           for _ in range(request.budget):
               x = rng.uniform(lb, ub)
               f = merit(x)
               if f < best_f:
                   best_x, best_f = x, f
           _, g = request.objective(list(best_x))
           h = float(np.sum(np.maximum(np.asarray(g, dtype=float), 0.0) ** 2))   # constraint violation
           return SolverOutcome(x=list(best_x), fmin=best_f, hmin=h)

The subproblems that should use it name it in the problem definition:

.. code-block:: yaml

   subproblem:
     sp1:
       solver: random_search
       budget: 50
       # ...

Strategies are stateless: anything to carry from one outer iteration to the next (a step size, a trust radius)
goes into ``SolverOutcome.warm_start`` and comes back in the next request's ``warm_start``. To make a solver
available to every DMDO installation, publish it in a package with an entry point:

.. code-block:: toml

   [project.entry-points."dmdo.solvers"]
   random_search = "my_pkg.solvers:RandomSearch"

A custom coordination policy
============================

A coordination policy scales each multiplier update: the exponent of the penalty growth :math:`\beta`, the stall
threshold :math:`\gamma` and the inner evaluation budget. The default policy, ``none``, applies the update of
:doc:`../background` unchanged. This one slows the penalty growth after 20 outer iterations:

.. code-block:: python

   from DMDO import CoordinationPolicy, policies

   @policies.register("slow_growth")
   class SlowGrowth(CoordinationPolicy):
       """Halve the exponent of the penalty-weight growth after outer iteration 20."""

       def compute_scales(self, coord, iteration):
           beta_scale = 0.5 if iteration >= 20 else 1.0
           return beta_scale, 1.0, 1.0, False      # beta, gamma and budget scales; last item: always False here

and is selected per coordinator:

.. code-block:: yaml

   coord:
     c1:
       adaptation_policy: slow_growth
       # ...

``coord`` is the coordinator: a policy may read its state (``coord.q``, ``coord.w``, ``coord.objective_history``,
...) to decide.

A custom executor
=================

See :doc:`parallel`.

Observers
=========

See :doc:`session`.
