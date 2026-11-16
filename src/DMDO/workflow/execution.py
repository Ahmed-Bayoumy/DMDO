"""Subproblem task messages and local executors.

A coordination sweep turns each subproblem solve into a :class:`SubproblemTask`, hands the tasks to an
:class:`~DMDO.ports.Executor` together with :func:`solve_task`, and merges the returned :class:`SubproblemResult`
messages. Because a task carries everything the solve needs (the subproblem state, including its copy of the
coordinator, plus the multipliers), the same task can run in this process, in a worker process, or on a remote worker
that unpickles it and calls :func:`solve_task`.
"""
from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

from ..ports import Executor
from ..registry import executors

__all__ = ["SubproblemTask", "SubproblemResult", "solve_task", "InProcessExecutor", "ProcessExecutor",
           "create_executor"]


@dataclass
class SubproblemTask:
  """One subproblem solve: ``subproblem.solve(v, w, file=file, iter=iteration)``."""
  index: int
  subproblem: Any
  v: Any
  w: Any
  iteration: int
  file: Optional[str] = None


@dataclass
class SubproblemResult:
  """The solved subproblem (its state and its coordinator copy after the solve) and the solver's raw output."""
  index: int
  subproblem: Any
  output: Any


def solve_task(task: SubproblemTask) -> SubproblemResult:
  """Solve one subproblem task. Module-level so executors can ship it to other processes or machines."""
  sp = task.subproblem
  output = sp.solve(task.v, task.w, file=task.file, iter=task.iteration)
  return SubproblemResult(index=task.index, subproblem=sp, output=output)


class InProcessExecutor:
  """Runs tasks one after another in the calling process (no copies: tasks are solved in place)."""

  def __init__(self, max_workers: Optional[int] = None) -> None:
    self.max_workers = 1

  def map(self, fn: Callable, tasks: Sequence) -> List:
    return [fn(task) for task in tasks]

  def close(self) -> None:
    pass

  def __enter__(self):
    return self

  def __exit__(self, *exc):
    self.close()
    return False


def _init_worker() -> None:
  """Worker-process setup: UTF-8 stdout/stderr on Windows, where a worker's redirected streams default to the ANSI
  code page and the OMADS progress bar (block characters) would raise UnicodeEncodeError."""
  from ..utils.fix_windows_encoding import fix_encoding
  fix_encoding()


class ProcessExecutor:
  """Runs tasks in a pool of worker processes; the pool is created lazily and reused across outer iterations.

  Tasks and results are pickled, so blackboxes must be importable (module-level) callables. On Windows/macOS the
  calling script needs the usual ``if __name__ == "__main__":`` guard.
  """

  def __init__(self, max_workers: Optional[int] = None) -> None:
    self.max_workers = max_workers
    self._pool: Optional[concurrent.futures.ProcessPoolExecutor] = None

  def map(self, fn: Callable, tasks: Sequence) -> List:
    if self._pool is None:
      self._pool = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers, initializer=_init_worker)
    return list(self._pool.map(fn, tasks))

  def close(self) -> None:
    if self._pool is not None:
      self._pool.shutdown(wait=True)
      self._pool = None

  def __enter__(self):
    return self

  def __exit__(self, *exc):
    self.close()
    return False

  def __getstate__(self):
    state = self.__dict__.copy()
    state["_pool"] = None
    return state


executors.register("inprocess", InProcessExecutor, aliases=("serial", "in_process", "local"), override=True)
executors.register("process", ProcessExecutor, aliases=("parallel", "processpool", "process_pool"), override=True)


def create_executor(spec: Any, max_workers: Optional[int] = None) -> Executor:
  """An executor from an instance (returned as is), a registered name, or a factory."""
  if not isinstance(spec, type) and isinstance(spec, Executor):
    return spec
  factory = executors.get(spec) if isinstance(spec, str) else spec
  return factory(max_workers=max_workers)
