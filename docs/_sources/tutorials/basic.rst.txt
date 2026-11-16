The basic problem from YAML
###########################

This tutorial runs the basic two-discipline problem of :doc:`../getting_started` from its YAML definition and walks
through what DMDO prints and writes.

The problem
===========

.. math::

   \begin{array}{ll}
   \min_{u,v,w} & u + v + a + b \\[2pt]
   \text{s.t.}  & a = \log_{10} u + \log_{10} v + \log_{10} b, \qquad b = u^{-1} + w^{-1} + a^{-1}, \\
                & w + b - 10 \le 0, \qquad 0 \le u, v, w, a, b \le 10.
   \end{array}

Subproblem 1 owns :math:`v`, computes :math:`a` and holds the objective; subproblem 2 owns :math:`w`, computes
:math:`b` and holds the constraint; both keep copies of :math:`u`, :math:`a` and :math:`b` (see
:doc:`../background` for the partitioned formulation).

The analyses
============

``examples/basic_disciplines.py`` writes the two analyses as :class:`~DMDO.Discipline` objects (inputs and outputs
keyed by variable name) and the subproblem objectives as plain functions:

.. literalinclude:: ../../examples/basic_disciplines.py
   :language: python
   :lines: 7-

The problem definition
======================

.. literalinclude:: ../../examples/basic_problem.yaml
   :language: yaml
   :emphasize-lines: 4-11,52,57,59,74-75,92-93

The highlighted lines are the settings that matter most:

* every variable is declared ``REAL``. OMADS optimizes variables of any other type on an integer lattice, so real
  variables must be declared with this exact keyword;
* the coordinator uses :math:`\beta = 1.3` and :math:`\gamma = 0.5` with the ``normal`` update scheme, and scales
  the inconsistencies by ``0.1`` (one tenth, which is :math:`1/(u - \ell)` for these bounds);
* both subproblems are solved by the OMADS ``poll`` step with 150 evaluations per outer iteration.

Running it
==========

.. code-block:: python

   from basic_disciplines import BLACKBOXES
   from DMDO import Session

   if __name__ == "__main__":
       session = Session.from_yaml("basic_problem.yaml", working_dir="_runs", blackboxes=BLACKBOXES)
       result = session.run()
       print(result.status, "|", result.stop_reason)
       print(result.iterations, result.objective, result.max_inconsistency, result.feasible)

With ``display: True`` (the ``MDO`` section), DMDO prints one line per outer iteration: the maximum inconsistency
``qmax``, the objective of the master subproblem, the change of the design vector ``dx`` and the largest penalty
weight, followed by the link with the largest inconsistency.

.. code-block:: text

      0 || qmax: 0.09043601 ||  Obj: 3.00000000 ||   dx: 7.04896213 || max(w): 1.30000000
   Highest inconsistency : b_2 to b_1
      1 || qmax: 0.04141597 ||  Obj: 4.86020038 ||   dx: 1.13437264 || max(w): 1.69000000
   Highest inconsistency : a_2 to a_1
      2 || qmax: 0.02897034 ||  Obj: 4.90302218 ||   dx: 0.21088849 || max(w): 2.19700000
   Highest inconsistency : a_2 to a_1
   ...
     48 || qmax: 0.00000050 ||  Obj: 5.58977652 ||   dx: 0.00000506 || max(w): 4843785982.75789833
   Highest inconsistency : b_2 to b_1
     49 || qmax: 0.00000031 ||  Obj: 5.58977066 ||   dx: 0.00000507 || max(w): 6296921777.58526802
   Highest inconsistency : a_2 to a_1
   finished | Iteration budget exhausted
   50 5.589770658112853 3.051476282101007e-07 True

After 50 outer iterations the two copies of every shared and coupling variable agree to :math:`3\times10^{-7}`:

.. code-block:: python

   >>> {k: round(float(v), 5) for k, v in result.variables.items()}
   {'u_1': 1.54957, 'v_1': 0.91759, 'a_1': 0.56132, 'b_1': 2.56129,
    'u_2': 1.54957, 'w_2': 7.43805, 'a_2': 0.56132, 'b_2': 2.56129}

These values match, to every digit shown, the recorded regression run of this problem (``basic_poll`` in
:doc:`../benchmarking`), which builds the same problem with the Python API instead of YAML.

The ``max(w)`` column shows the penalty weights growing on the links that stall, from 1.3 to about
:math:`6	imes10^{9}`; this growth is what eventually forces the copies together.

The output files
================

The run writes its outputs to ``_runs/basic_problem_post``:

.. code-block:: text

   _runs/basic_problem_post/
   ├── Coordination_history.out          one row per outer iteration
   ├── DMDO.log                          the log of the run
   ├── coord_chk.pkl                     checkpoint for resuming the run
   └── coordination_summary_post/
       ├── SP_1/
       │   ├── SP_1_0.out ... SP_1_49.out   every evaluation of subproblem 1, per outer iteration
       │   └── OMADS.log
       └── SP_2/ ...

``Coordination_history.out`` is a comma-separated table with, for each outer iteration, the time, the iteration
number, the maximum inconsistency, the objective, the feasibility status, the change of the design, the largest
penalty weight, the link with the largest inconsistency, the constraint violation of each subproblem and the value
of every variable copy. It can be loaded with :func:`pandas.read_csv`:

.. code-block:: python

   import pandas as pd

   history = pd.read_csv("_runs/basic_problem_post/Coordination_history.out", skipinitialspace=True)
   history.columns = history.columns.str.strip()
   history.plot(x="Iteration #", y="Max. inconsistency", logy=True)

:doc:`../user_guide/outputs` describes every file and column.
