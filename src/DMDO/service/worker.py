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
"""Remote worker: claims subproblem jobs from a DMDO service, runs them, posts the results back.

A worker needs DMDO and the problem's analysis code importable (the jobs reference blackboxes by module path when
they are pickled). Start one per machine/discipline, optionally restricted to tagged jobs::

    dmdo worker --connect http://dmdo-host:8400 --tags aero --token $DMDO_TOKEN
"""
import os
import pickle
import socket
import threading
import traceback
from typing import Iterable, Optional

from .client import Client, ServiceError

__all__ = ["run_worker"]


def run_worker(url: str, tags: Iterable[str] = (), token: Optional[str] = None, name: Optional[str] = None,
               max_jobs: Optional[int] = None, poll_wait: float = 20.0,
               stop_event: Optional[threading.Event] = None) -> int:
  """Serve jobs until ``stop_event`` is set or ``max_jobs`` jobs were run; returns the number of jobs run."""
  client = Client(url, token=token, timeout=poll_wait + 30.0)
  name = name or f"{socket.gethostname()}-{os.getpid()}-{threading.get_ident()}"
  tags = [t for t in tags if t]
  done = 0
  while (stop_event is None or not stop_event.is_set()) and (max_jobs is None or done < max_jobs):
    try:
      job = client.claim_job(name, tags=tags, wait=poll_wait)
    except (ServiceError, OSError):
      if stop_event is not None and stop_event.wait(1.0):
        break
      continue
    if job is None:
      continue
    job_id, job_payload = job
    try:
      fn, task = pickle.loads(job_payload)
      ok, payload = True, pickle.dumps(fn(task))
    except Exception:
      ok, payload = False, traceback.format_exc().encode("utf-8")
    try:
      client.complete_job(job_id, ok, payload)
    except ServiceError:
      pass  # job was cancelled or already completed by another worker
    done += 1
  return done
