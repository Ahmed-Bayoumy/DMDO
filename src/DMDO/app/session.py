"""Session: the public, framework-neutral API for building and driving a DMDO run from other code.

A session wraps one :class:`~DMDO.MDO` problem and its coordination loop (:meth:`DMDO.MDO.coordinate`). It can be
driven three ways, which can be mixed:

- ``run()``: to the end, solving subproblems with the session's executor;
- ``step()``: one outer iteration at a time;
- ``ask()`` / ``tell(results)``: the caller solves the subproblem tasks itself (in another framework, on its own
  workers, ...) with :func:`DMDO.solve_task` or an equivalent, and hands the results back.

::

    from DMDO import Session, solve_task

    with Session.from_yaml("problem.yaml", working_dir="runs", blackboxes={"A1": a1}) as session:
        while not session.done:
            tasks = session.ask()
            session.tell([solve_task(t) for t in tasks])
        print(session.result().objective)
"""
import copy
import os
import pickle
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from ..core.enums import FEASIBILITY_TOL
from ..ports import IterationRecord
from ..workflow.execution import SubproblemResult, SubproblemTask, create_executor, solve_task
from ..workflow.mdo import MDO, IterationEnd
from .main import build_mdo, load_setup

__all__ = ["Session", "RunResult", "SessionError"]


class SessionError(RuntimeError):
  """A session method was called in a state that does not allow it."""


def jsonable(value: Any) -> Any:
  """Convert numpy scalars/arrays (also nested in lists/dicts) to JSON-friendly Python values."""
  if isinstance(value, np.ndarray):
    return [jsonable(v) for v in value.tolist()]
  if isinstance(value, np.generic):
    return jsonable(value.item())
  if isinstance(value, (list, tuple)):
    return [jsonable(v) for v in value]
  if isinstance(value, dict):
    return {str(k): jsonable(v) for k, v in value.items()}
  if isinstance(value, float) and not np.isfinite(value):
    return str(value)
  return value


def record_to_dict(record: IterationRecord) -> Dict[str, Any]:
  return jsonable(asdict(record))


@dataclass
class RunResult:
  """Outcome of a session: final design, objective, consistency and the per-iteration history."""

  status: str
  stop_reason: str
  iterations: int
  objective: Any
  max_inconsistency: float
  feasible: bool
  variables: Dict[str, Any]
  history: List[IterationRecord] = field(default_factory=list)
  post_dir: Optional[str] = None

  def to_dict(self) -> Dict[str, Any]:
    out = {k: jsonable(v) for k, v in asdict(self).items() if k != "history"}
    out["history"] = [record_to_dict(r) for r in self.history]
    return out


class Session:
  """Build and drive one DMDO coordination run.

  :param mdo: the problem (built with :meth:`from_yaml`, :meth:`from_dict`, :meth:`from_checkpoint` or by hand).
  :param mode: ``"serial"`` (Gauss-Seidel, in-process) or ``"parallel"`` (Jacobi, process pool) defaults; default: the
    coordinator's ``mode`` from the problem definition.
  :param sweep: ``"gauss_seidel"`` or ``"jacobi"``, overriding ``mode``.
  :param executor: executor used by :meth:`run` and :meth:`step` (an :class:`~DMDO.ports.Executor` or a registered
    name); not used by ask/tell.
  :param max_workers: worker count for executors created from a name.
  :param observers: extra :class:`~DMDO.ports.RunObserver` objects.
  :param resume: continue a checkpointed run (see :meth:`from_checkpoint`).
  """

  def __init__(self, mdo: MDO, *, mode: Optional[str] = None, sweep: Optional[str] = None, executor: Any = None,
               max_workers: Optional[int] = None, observers: Iterable[Any] = (), resume: bool = False,
               file: Optional[str] = None) -> None:
    self.mdo = mdo
    self.mode = mode if mode is not None else mdo.default_mode()
    self.sweep = sweep
    self.max_workers = max_workers
    self.observers: List[Any] = list(observers)
    self.history: List[IterationRecord] = []
    self.status = "created"
    self.error: Optional[BaseException] = None
    self._executor_spec = executor
    self._executor = None
    self._resume = resume
    # Iterations completed before the checkpoint a resumed session starts from.
    self._completed_before = int(getattr(mdo, "iter", 0)) if resume else 0
    self._file = file if file is not None else os.path.join(mdo.post_dir, "coordination_summary.out")
    self._loop = None
    self._item: Any = None  # what the loop yielded last and is waiting on
    self._stop_requested = threading.Event()
    self._lock = threading.RLock()

  # ------------------------------------------------------------------------------------------------ construction
  @classmethod
  def from_dict(cls, data: Mapping[str, Any], *, working_dir: str = ".", mdo_name: str = "MDO",
                blackboxes: Optional[Mapping[str, Any]] = None, **options) -> "Session":
    """Build from a problem definition dictionary (the content of a DMDO YAML file).

    ``blackboxes`` maps the ``blackbox:``/``opt:`` names of the definition to callables or
    :class:`~DMDO.ports.Discipline` objects (names can also come from :func:`DMDO.register_blackbox` or be
    ``"module:attr"`` import paths). Outputs go to ``<working_dir>/<mdo_name>_post``.
    """
    os.makedirs(working_dir, exist_ok=True)
    mdo = build_mdo(dict(data), working_dir=working_dir, mdo_name=mdo_name,
                    blackboxes=dict(blackboxes) if blackboxes else None)
    return cls(mdo, **options)

  @classmethod
  def from_yaml(cls, setup_file: str, *, working_dir: str = ".", mdo_name: Optional[str] = None,
                blackboxes: Optional[Mapping[str, Any]] = None, **options) -> "Session":
    """Build from a DMDO YAML (or JSON) problem file; ``mdo_name`` defaults to the file name."""
    if mdo_name is None:
      mdo_name = os.path.splitext(os.path.basename(setup_file))[0]
    return cls.from_dict(load_setup(setup_file), working_dir=working_dir, mdo_name=mdo_name,
                         blackboxes=blackboxes, **options)

  @classmethod
  def from_checkpoint(cls, checkpoint_file: str, **options) -> "Session":
    """Resume from a ``coord_chk.pkl`` checkpoint written by a previous run."""
    with open(checkpoint_file, "rb") as fh:
      mdo = pickle.load(fh)
    options.setdefault("mode", getattr(mdo, "mode", "serial"))
    return cls(mdo, resume=True, file=getattr(mdo, "file", None), **options)

  # ------------------------------------------------------------------------------------------------ state
  @property
  def done(self) -> bool:
    return self.status in ("finished", "failed", "closed")

  @property
  def iteration(self) -> int:
    """Number of completed outer iterations (including those before a resumed checkpoint)."""
    return self._completed_before + len(self.history)

  @property
  def pending_tasks(self) -> List[SubproblemTask]:
    return list(self._item) if isinstance(self._item, list) else []

  def add_observer(self, observer: Any) -> None:
    if self._loop is not None:
      raise SessionError("Observers must be added before the session starts")
    self.observers.append(observer)

  def request_stop(self) -> None:
    """End the run after the current outer iteration (thread-safe)."""
    self._stop_requested.set()

  # ------------------------------------------------------------------------------------------------ driving
  def _start(self) -> None:
    if self._loop is None:
      self._loop = self.mdo.coordinate(file=self._file, resume=self._resume, mode=self.mode, sweep=self.sweep,
                                       observers=self.observers, should_stop=self._stop_requested.is_set)
      self.status = "running"

  def _advance(self, reply=None) -> Any:
    """Resume the loop with ``reply``; return the next item, or None when the loop has ended."""
    try:
      item = self._loop.send(reply)
    except StopIteration:
      self._item = None
      self.status = "finished"
      self._release_executor()
      return None
    except BaseException as exc:
      self._item = None
      self.status = "failed"
      self.error = exc
      self._release_executor()
      raise
    if isinstance(item, IterationEnd):
      self.history.append(item.record)
      if item.finished:
        # Let the loop run its end-of-run hooks now, so the session is complete as soon as the last record is out.
        self._advance(None)
        return item
    self._item = item
    return item

  def _require_open(self) -> None:
    if self.done:
      raise SessionError(f"The session is {self.status}")

  def ask(self) -> List[SubproblemTask]:
    """The subproblem tasks to solve next ([] once the run is over).

    Tasks of one batch are independent (Jacobi: every subproblem; Gauss-Seidel: one subproblem at a time). Calling
    ``ask`` again before ``tell`` returns the same batch.
    """
    with self._lock:
      if self.done:
        return []
      self._start()
      while not isinstance(self._item, list):
        if self._advance(None) is None:
          return []
        if self.done:
          return []
      return list(self._item)

  def tell(self, results: Sequence[SubproblemResult]) -> Optional[IterationRecord]:
    """Hand back the results of the last :meth:`ask` batch (any order).

    Returns the :class:`~DMDO.ports.IterationRecord` when these results completed an outer iteration, else None.
    """
    with self._lock:
      self._require_open()
      if not isinstance(self._item, list):
        raise SessionError("tell() called without pending tasks; call ask() first")
      item = self._advance(list(results))
      return item.record if isinstance(item, IterationEnd) else None

  def step(self) -> Optional[IterationRecord]:
    """Run one outer iteration with the session's executor; None when the run is already over."""
    with self._lock:
      if self.done:
        return None
      self._start()
      executor = self._get_executor()
      item = self._item
      if not isinstance(item, list):
        item = self._advance(None)
      while isinstance(item, list):
        item = self._advance(executor.map(solve_task, item))
      return item.record if isinstance(item, IterationEnd) else None

  def run(self) -> RunResult:
    """Run (or continue) to the end of the coordination and return the result."""
    while self.step() is not None:
      pass
    return self.result()

  def result(self) -> RunResult:
    """The current outcome (final once :attr:`done`)."""
    mdo = self.mdo
    coord = mdo.Coordinator
    hmin = mdo.hmin
    hmax = max(np.atleast_1d(np.asarray(hmin, dtype=float))) if hmin is not None and np.size(hmin) else 0.0
    variables = {f"{v.name}_{v.sp_index}": copy.deepcopy(v.value) for v in getattr(coord, "master_vars", [])}
    q = np.asarray(getattr(coord, "q", []), dtype=float)
    return RunResult(
      status=self.status,
      stop_reason=str(getattr(mdo, "stop", "")),
      iterations=self.iteration,
      objective=mdo.fmin,
      max_inconsistency=float(np.max(np.abs(q))) if q.size else float("nan"),
      feasible=bool(hmax <= FEASIBILITY_TOL),
      variables=variables,
      history=list(self.history),
      post_dir=getattr(mdo, "post_dir", None),
    )

  # ------------------------------------------------------------------------------------------------ resources
  def _get_executor(self):
    if self._executor is None:
      spec = self._executor_spec
      if spec is None:
        spec = "inprocess" if MDO._is_serial(self.mode) else "process"
      self._owns_executor = spec is None or isinstance(spec, (str, type))
      self._executor = create_executor(spec, max_workers=self.max_workers or len(self.mdo.subProblems))
    return self._executor

  def _release_executor(self) -> None:
    if self._executor is not None and getattr(self, "_owns_executor", False):
      self._executor.close()
    self._executor = None

  def close(self) -> None:
    """Stop the run (if still going) and release the executor."""
    with self._lock:
      if self._loop is not None and not self.done:
        self._loop.close()
      if not self.done:
        self.status = "closed"
      self._release_executor()

  def __enter__(self) -> "Session":
    return self

  def __exit__(self, *exc) -> bool:
    self.close()
    return False

  def __repr__(self) -> str:
    return f"Session({getattr(self.mdo, 'mdo_name', '?')!r}, status={self.status!r}, iterations={self.iteration})"

