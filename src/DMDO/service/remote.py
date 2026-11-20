# ------------------------------------------------------------------------------------#
#  Distributed Multidisciplinary Design Optimization - DMDO                           #
#                                                                                     #
#  Author: Ahmed H. Bayoumy                                                           #
#  email: ahmed.bayoumy@mail.mcgill.ca                                                #
#                                                                                     #
#  This program is free software: you can redistribute it and/or modify it under the  #
#  terms of the GNU General Public License as published by the Free Software          #
#  Foundation, either version 3 of the License, or (at your option) any later         #
#  version.                                                                           #
#                                                                                     #
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY    #
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A    #
#  PARTICULAR PURPOSE.  See the GNU General Public License for more details.          #
#                                                                                     #
#  You should have received a copy of the GNU General Public License along            #
#  with this program. If not, see <http://www.gnu.org/licenses/>.                     #
#                                                                                     #
#  You can find information on simple_mads at                                         #
#  https://github.com/Ahmed-Bayoumy/DMDO                                              #
# ------------------------------------------------------------------------------------#
"""Remote execution: a task queue shared with worker processes, and the executor that feeds it.

:class:`RemoteExecutor` implements :class:`DMDO.ports.Executor`: ``map(fn, tasks)`` puts one pickled ``(fn, task)``
job per task on a :class:`TaskQueue` and blocks until workers have posted every result. Workers (see
:mod:`DMDO.service.worker`) claim jobs over HTTP, optionally only jobs carrying one of their tags, so a subproblem can
be pinned to the machine that has its analysis code (``placement={subproblem_index: tag}``).

Jobs and results are pickled, so the service and its workers must trust each other (see :mod:`DMDO.service`).
"""
import itertools
import pickle
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = ["TaskQueue", "RemoteExecutor", "RemoteTaskError"]


class RemoteTaskError(RuntimeError):
  """A worker reported an exception while running a job."""


@dataclass
class _Job:
  id: str
  payload: bytes
  tag: Optional[str]
  claimed_by: Optional[str] = None
  claimed_at: float = 0.0
  done: bool = False
  ok: bool = False
  result: bytes = b""
  attempts: int = 0
  events: List[str] = field(default_factory=list)


class TaskQueue:
  """Thread-safe job queue between executors (producers) and HTTP workers (consumers).

  :param lease_seconds: a claimed job whose result has not arrived after this many seconds is offered again (a worker
    that died mid-job); None disables re-delivery.
  """

  def __init__(self, lease_seconds: Optional[float] = None) -> None:
    self.lease_seconds = lease_seconds
    self._jobs: Dict[str, _Job] = {}
    self._pending: List[str] = []
    self._cond = threading.Condition()
    self._ids = itertools.count(1)

  # ------------------------------------------------------------------------------------------------ producer side
  def put(self, payload: bytes, tag: Optional[str] = None) -> str:
    with self._cond:
      job_id = f"job-{next(self._ids)}"
      self._jobs[job_id] = _Job(id=job_id, payload=payload, tag=tag)
      self._pending.append(job_id)
      self._cond.notify_all()
      return job_id

  def wait(self, job_ids: Sequence[str], timeout: Optional[float] = None) -> List[Tuple[bool, bytes]]:
    """Block until every job in ``job_ids`` has a result; return ``(ok, payload)`` per job, in order."""
    deadline = None if timeout is None else time.monotonic() + timeout
    with self._cond:
      while not all(self._jobs[j].done for j in job_ids):
        self._requeue_expired()
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
          raise TimeoutError(f"Remote jobs not finished after {timeout} s: "
                             f"{[j for j in job_ids if not self._jobs[j].done]}")
        self._cond.wait(timeout=1.0 if remaining is None else min(1.0, remaining))
      return [(self._jobs[j].ok, self._jobs.pop(j).result) for j in job_ids]

  def cancel(self, job_ids: Iterable[str]) -> None:
    with self._cond:
      for j in job_ids:
        self._jobs.pop(j, None)
        if j in self._pending:
          self._pending.remove(j)

  # ------------------------------------------------------------------------------------------------ worker side
  def claim(self, worker: str, tags: Iterable[str] = (), timeout: float = 0.0) -> Optional[Tuple[str, bytes]]:
    """Take the oldest job this worker may run: untagged jobs, or jobs tagged with one of ``tags``."""
    tags = set(tags)
    deadline = time.monotonic() + max(timeout, 0.0)
    with self._cond:
      while True:
        self._requeue_expired()
        for job_id in self._pending:
          job = self._jobs[job_id]
          if job.tag is None or job.tag in tags:
            self._pending.remove(job_id)
            job.claimed_by, job.claimed_at = worker, time.monotonic()
            job.attempts += 1
            return job.id, job.payload
        remaining = deadline - time.monotonic()
        if remaining <= 0:
          return None
        self._cond.wait(timeout=min(remaining, 1.0))

  def complete(self, job_id: str, ok: bool, payload: bytes) -> bool:
    """Store a job's result; False if the job is unknown (cancelled, or already completed by another worker)."""
    with self._cond:
      job = self._jobs.get(job_id)
      if job is None or job.done:
        return False
      if job_id in self._pending:
        self._pending.remove(job_id)
      job.done, job.ok, job.result = True, ok, payload
      self._cond.notify_all()
      return True

  def _requeue_expired(self) -> None:
    if self.lease_seconds is None:
      return
    now = time.monotonic()
    for job in self._jobs.values():
      if (not job.done and job.claimed_by is not None and job.id not in self._pending
              and now - job.claimed_at >= self.lease_seconds):
        job.claimed_by = None
        self._pending.append(job.id)
        self._cond.notify_all()

  def stats(self) -> Dict[str, int]:
    with self._cond:
      return {"pending": len(self._pending),
              "running": sum(1 for j in self._jobs.values() if not j.done and j.id not in self._pending),
              "finished": sum(1 for j in self._jobs.values() if j.done)}


class RemoteExecutor:
  """Executor whose tasks run on remote workers connected to a :class:`TaskQueue`.

  :param queue: the queue workers claim jobs from (the service's queue).
  :param placement: ``{subproblem index: worker tag}``; tasks of other subproblems go to any worker.
  :param timeout: seconds to wait for one ``map`` call (None: no limit).
  """

  def __init__(self, queue: TaskQueue, placement: Optional[Mapping[Any, str]] = None,
               timeout: Optional[float] = None, max_workers: Optional[int] = None) -> None:
    self.queue = queue
    self.placement = {str(k): str(v) for k, v in (placement or {}).items()}
    self.timeout = timeout

  def _tag(self, task: Any) -> Optional[str]:
    sp = getattr(task, "subproblem", None)
    key = getattr(sp, "index", getattr(task, "index", None))
    return self.placement.get(str(key))

  def map(self, fn: Callable, tasks: Sequence[Any]) -> List[Any]:
    job_ids = [self.queue.put(pickle.dumps((fn, task)), tag=self._tag(task)) for task in tasks]
    try:
      outcomes = self.queue.wait(job_ids, timeout=self.timeout)
    except BaseException:
      self.queue.cancel(job_ids)
      raise
    results = []
    for job_id, (ok, payload) in zip(job_ids, outcomes):
      if not ok:
        raise RemoteTaskError(f"{job_id} failed on a worker:\n{payload.decode('utf-8', 'replace')}")
      results.append(pickle.loads(payload))
    return results

  def close(self) -> None:
    pass
