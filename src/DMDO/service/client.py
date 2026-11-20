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
"""HTTP client for a DMDO service (standard library only): sessions for drivers, job transport for workers."""
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple

__all__ = ["Client", "RemoteSession", "ServiceError"]


class ServiceError(RuntimeError):
  """The service answered with an error status."""

  def __init__(self, status: int, message: str) -> None:
    super().__init__(f"HTTP {status}: {message}")
    self.status = status
    self.message = message


class _Http:
  def __init__(self, url: str, token: Optional[str] = None, timeout: float = 60.0) -> None:
    self.url = url.rstrip("/")
    self.token = token
    self.timeout = timeout

  def request(self, method: str, path: str, body: Optional[bytes] = None, content_type: str = "application/json",
              timeout: Optional[float] = None):
    req = urllib.request.Request(self.url + path, data=body, method=method)
    if body is not None:
      req.add_header("Content-Type", content_type)
    if self.token:
      req.add_header("Authorization", f"Bearer {self.token}")
    try:
      return urllib.request.urlopen(req, timeout=self.timeout if timeout is None else timeout)
    except urllib.error.HTTPError as exc:
      detail = exc.read().decode("utf-8", "replace")
      try:
        detail = json.loads(detail).get("error", detail)
      except ValueError:
        pass
      raise ServiceError(exc.code, detail) from None

  def json(self, method: str, path: str, payload: Any = None, timeout: Optional[float] = None) -> Any:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    with self.request(method, path, body, timeout=timeout) as resp:
      raw = resp.read()
    return json.loads(raw) if raw else None


class RemoteSession:
  """A session living in a DMDO service; mirrors the run/step/result part of :class:`DMDO.Session`."""

  def __init__(self, http: _Http, session_id: str) -> None:
    self._http = http
    self.id = session_id

  def _path(self, suffix: str = "") -> str:
    return f"/sessions/{self.id}{suffix}"

  def status(self) -> Dict[str, Any]:
    return self._http.json("GET", self._path())

  def run(self, wait: bool = False, poll: float = 0.5) -> Dict[str, Any]:
    """Start running to the end in the service; with ``wait`` block until done and return the result."""
    self._http.json("POST", self._path("/run"))
    if not wait:
      return self.status()
    for event in self.events():
      if event["event"] in ("run_end", "run_failed", "closed"):
        break
    return self.result()

  def step(self) -> Optional[Dict[str, Any]]:
    """Run one outer iteration; the iteration record, or None when the run is over."""
    return self._http.json("POST", self._path("/step"), timeout=None)["record"]

  def stop(self) -> None:
    self._http.json("POST", self._path("/stop"))

  def result(self) -> Dict[str, Any]:
    return self._http.json("GET", self._path("/result"))

  def history(self):
    return self._http.json("GET", self._path("/history"))

  def events(self, since: int = 0) -> Iterator[Dict[str, Any]]:
    """Server-sent events of the run (``iteration_end`` records, then ``run_end``/``run_failed``)."""
    with self._http.request("GET", self._path(f"/events?since={since}"), timeout=None) as resp:
      name, data = None, []
      for raw in resp:
        line = raw.decode("utf-8").rstrip("\r\n")
        if line.startswith("event:"):
          name = line[6:].strip()
        elif line.startswith("data:"):
          data.append(line[5:].strip())
        elif not line and name:
          payload = json.loads("\n".join(data)) if data else None
          yield {"event": name, "data": payload}
          if name in ("run_end", "run_failed", "closed"):
            return
          name, data = None, []

  def delete(self) -> None:
    self._http.json("DELETE", self._path())

  def __repr__(self) -> str:
    return f"RemoteSession({self.id!r}, {self._http.url!r})"


class Client:
  """Connect to a DMDO service started with ``dmdo serve`` (or :func:`DMDO.service.serve`).

  ::

      client = Client("http://localhost:8400", token="...")
      session = client.create_session(setup_file="problem.yaml", mdo_name="wing",
                                      blackboxes={"A1": "my_pkg.aero:A1"}, sweep="jacobi")
      for event in session.run().events(): ...
  """

  def __init__(self, url: str, token: Optional[str] = None, timeout: float = 60.0) -> None:
    self._http = _Http(url, token, timeout)

  def health(self) -> Dict[str, Any]:
    return self._http.json("GET", "/health")

  def create_session(self, setup: Optional[Mapping[str, Any]] = None, setup_file: Optional[str] = None,
                     mdo_name: str = "MDO", blackboxes: Optional[Mapping[str, str]] = None, **options) -> RemoteSession:
    """Create a session from a problem definition dict (``setup``) or a YAML path readable by the service.

    ``blackboxes`` maps names to ``"module:attr"`` import paths resolved by the service (and its workers).
    ``options``: ``mode``, ``sweep``, ``executor`` (``"inprocess"``, ``"process"`` or ``"remote"``), ``placement``
    (``{subproblem index: worker tag}``, implies ``executor="remote"``) and ``max_workers``.
    """
    payload = {"setup": setup, "setup_file": setup_file, "mdo_name": mdo_name, "blackboxes": dict(blackboxes or {}),
               "options": options}
    reply = self._http.json("POST", "/sessions", payload, timeout=None)
    return RemoteSession(self._http, reply["session_id"])

  def session(self, session_id: str) -> RemoteSession:
    return RemoteSession(self._http, session_id)

  def sessions(self):
    return self._http.json("GET", "/sessions")

  # ------------------------------------------------------------------------------------------------ worker transport
  def claim_job(self, worker: str, tags=(), wait: float = 20.0) -> Optional[Tuple[str, bytes]]:
    """Long-poll for a job; ``(job_id, pickled (fn, task))`` or None when nothing arrived within ``wait`` seconds."""
    query = urllib.parse.urlencode({"worker": worker, "tags": ",".join(tags), "wait": wait})
    with self._http.request("GET", f"/work?{query}", timeout=wait + 30.0) as resp:
      if resp.status == 204:
        return None
      return resp.headers["X-DMDO-Job"], resp.read()

  def complete_job(self, job_id: str, ok: bool, payload: bytes) -> None:
    path = f"/work/{urllib.parse.quote(job_id)}?ok={int(bool(ok))}"
    with self._http.request("POST", path, payload, content_type="application/octet-stream"):
      pass
