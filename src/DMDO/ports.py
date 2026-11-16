"""Ports: the framework-neutral interfaces DMDO's core talks to.

Anything that satisfies one of these protocols can be plugged into DMDO, whether it comes from DMDO itself, from
user code, or from another package (an OpenMDAO model, a GEMSEO discipline, a remote service client, ...). DMDO never
imports those packages; an adapter only has to satisfy the protocol.

* :class:`Discipline` - a disciplinary analysis: named inputs in, named outputs out.
* :class:`SubproblemSolver` - an optimizer that solves one subproblem given a :class:`SolverRequest`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, runtime_checkable

__all__ = ["Discipline", "SubproblemSolver", "SolverRequest", "SolverOutcome"]


@runtime_checkable
class Discipline(Protocol):
  """A disciplinary analysis.

  ``compute`` receives the analysis inputs keyed by variable name (scalars or arrays) and returns a mapping that
  contains every output variable of the analysis, keyed by name. Implementations may also expose
  ``input_names`` / ``output_names`` sequences for validation, but they are not required.

  Example adapter for an external framework (kept in user code, not in DMDO)::

    class OpenMDAODiscipline:
      def __init__(self, prob, inputs, outputs):
        self.prob, self.input_names, self.output_names = prob, inputs, outputs
      def compute(self, inputs):
        for name in self.input_names:
          self.prob.set_val(name, inputs[name])
        self.prob.run_model()
        return {name: self.prob.get_val(name) for name in self.output_names}
  """

  def compute(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    ...


@dataclass
class SolverRequest:
  """Everything a solver needs to optimize one subproblem.

  ``objective(x, *args)`` evaluates the augmented subproblem at ``x``. Its return value has the form the solver
  declared via :attr:`SubproblemSolver.objective_form`: ``"vector"`` -> ``[f + penalty, constraints]``,
  ``"scalar"`` -> a single penalized float. ``constraints(x)`` returns the inequality constraint values (``g <= 0``).
  """
  name: str
  objective: Callable[..., Any]
  constraints: Callable[[Sequence[float]], List[float]]
  x0: List[Any]
  lb: List[Any]
  ub: List[Any]
  var_names: List[str]
  var_types: List[Any]
  var_sets: Optional[Dict[str, Any]]
  constants: List[Any]
  constants_names: List[str]
  budget: int
  budget_scale: float
  seed: int
  iteration: int
  tol: float = 1e-12
  display: bool = False
  post_dir: Optional[str] = None
  #: Solver-specific configuration (YAML ``CONFIG.<name>`` block for OMADS, ``scipy`` block for scipy).
  config: Optional[Dict[str, Any]] = None
  #: Warm-start state carried between successive solves of the same subproblem (e.g. OMADS poll size).
  warm_start: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SolverOutcome:
  """Result of one subproblem solve."""
  x: List[Any]
  fmin: Any
  hmin: Any
  #: The value returned by ``SubProblem.solve`` (kept for backward compatibility: OMADS output dict / scipy x).
  raw: Any = None
  #: Warm-start state to carry into the next solve (merged into the subproblem's state).
  warm_start: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SubproblemSolver(Protocol):
  """Strategy that solves one subproblem.

  Register an implementation with ``DMDO.registry.solvers.register("name", cls)`` (or via the ``dmdo.solvers``
  entry-point group) and select it with ``solver: name`` in the subproblem definition.
  """

  #: Shape of ``SolverRequest.objective``'s return value: ``"vector"`` or ``"scalar"``.
  objective_form: str
  #: Whether trial points outside the bounds should be rejected (returned as infeasible) without evaluation.
  rejects_out_of_bounds: bool

  def solve(self, request: SolverRequest) -> SolverOutcome:
    ...

  def report_hmin(self, hmin: Any) -> Any:
    """The constraint-violation value to report for a solve (logs, detector, rollback)."""
    ...
