"""Ports: the framework-neutral interfaces DMDO's core talks to.

Anything that satisfies one of these protocols can be plugged into DMDO, whether it comes from DMDO itself, from
user code, or from another package (another framework's model, a remote service client, ...). DMDO never
imports those packages; an adapter only has to satisfy the protocol.

* :class:`Discipline` - a disciplinary analysis: named inputs in, named outputs out.
* :class:`SubproblemSolver` - an optimizer that solves one subproblem given a :class:`SolverRequest`.
* :class:`Executor` - runs subproblem tasks: in-process, in a process pool, or on remote workers/services.
* :class:`RunObserver` - hooks into the coordination loop (history files, console, checkpoints, monitoring).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, TypeVar, runtime_checkable

__all__ = ["Discipline", "SubproblemSolver", "SolverRequest", "SolverOutcome", "Executor", "RunObserver",
           "IterationRecord"]

_T = TypeVar("_T")
_R = TypeVar("_R")


@runtime_checkable
class Discipline(Protocol):
  """A disciplinary analysis.

  ``compute`` receives the analysis inputs keyed by variable name (scalars or arrays) and returns a mapping that
  contains every output variable of the analysis, keyed by name. Implementations may also expose
  ``input_names`` / ``output_names`` sequences for validation, but they are not required.

  Example adapter around a model from another framework (kept in user code, not in DMDO)::

    class ExternalModelDiscipline:
      def __init__(self, model, inputs, outputs):
        self.model, self.input_names, self.output_names = model, inputs, outputs
      def compute(self, inputs):
        for name in self.input_names:
          self.model.set_input(name, inputs[name])   # whatever the framework's API is
        self.model.evaluate()
        return {name: self.model.get_output(name) for name in self.output_names}
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


@runtime_checkable
class Executor(Protocol):
  """Runs independent tasks and returns their results **in task order**.

  The coordination loop hands an executor :class:`DMDO.workflow.execution.SubproblemTask` messages together with the
  module-level function that solves one (:func:`DMDO.workflow.execution.solve_task`). Local executors call it in
  this process or in worker processes; a remote executor (an HTTP/gRPC/queue client, a Ray or Dask cluster, ...)
  ships the pickled task to a worker that calls the same function. Whether subproblems see each other's updates
  within an outer iteration is decided by the coordination sweep, not by the executor.

  Register an implementation with ``DMDO.registry.executors.register("name", factory)`` (or via the
  ``dmdo.executors`` entry-point group); the factory is called as ``factory(max_workers=n)``.
  """

  def map(self, fn: Callable[[_T], _R], tasks: Sequence[_T]) -> List[_R]:
    ...

  def close(self) -> None:
    ...


@dataclass(frozen=True)
class IterationRecord:
  """Summary of one finished outer iteration, passed to :meth:`RunObserver.on_iteration_end`."""

  iteration: int
  objective: Any
  max_inconsistency: float
  max_penalty_weight: float
  variables_change: float
  worst_link: int


class RunObserver:
  """Hooks called by the coordination loop (:meth:`DMDO.MDO.run`), in this order within an outer iteration.

  Subclass and override the hooks you need; every hook is optional (an observer may also be any object that defines
  some of these methods). ``run`` is the :class:`DMDO.MDO` instance; read its state (``run.Coordinator``,
  ``run.subProblems``, ``run.fmin``, ...) rather than keeping references to it. Observers are called in the order they
  are given, after DMDO's own (AI regularizer, rollback, checkpoint, console, coordination history). They are not
  pickled into checkpoints: pass them again to ``run(..., resume=True)``.

  ``on_run_start(run)``
      once, before the first outer iteration.
  ``on_iteration_start(run, iteration)``
      before the subproblems are solved.
  ``on_resume(run, iteration)``
      instead of ``on_iteration_start`` for the first iteration of a run resumed from a checkpoint: the checkpoint is
      written after that iteration's ``on_iteration_start`` hooks, so their effects are already in the restored state.
  ``on_sweep_end(run, iteration)``
      all subproblems solved and the inconsistencies computed, before the penalty update.
  ``on_iteration_end(run, record)``
      the iteration's results are final for reporting (an :class:`IterationRecord`).
  ``on_multipliers_updated(run, iteration)``
      after the Lagrange multipliers and penalty weights were updated, before the stopping test.
  ``on_run_end(run)``
      once, after the last outer iteration (not called when the run raises).
  """

  def on_run_start(self, run: Any) -> None:
    pass

  def on_iteration_start(self, run: Any, iteration: int) -> None:
    pass

  def on_resume(self, run: Any, iteration: int) -> None:
    pass

  def on_sweep_end(self, run: Any, iteration: int) -> None:
    pass

  def on_iteration_end(self, run: Any, record: IterationRecord) -> None:
    pass

  def on_multipliers_updated(self, run: Any, iteration: int) -> None:
    pass

  def on_run_end(self, run: Any) -> None:
    pass
