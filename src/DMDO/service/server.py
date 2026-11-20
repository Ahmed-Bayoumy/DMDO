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
"""DMDO as an HTTP service (standard library only): sessions over JSON, live events, and a job queue for workers.

Endpoints (JSON unless noted; send ``Authorization: Bearer <token>`` when the service has a token):

=======  ================================  =======================================================================
GET      /health                           service status, session count, job-queue counters
GET      /sessions                         list sessions
POST     /sessions                         create: ``{"setup": {...} | "setup_file": "path", "mdo_name": "...",
                                           "blackboxes": {"A1": "pkg.module:attr"}, "options": {"mode", "sweep",
                                           "executor": "inprocess"|"process"|"remote", "placement": {"1": "tag"},
                                           "max_workers", "timeout"}}``
GET      /sessions/{id}                    status
POST     /sessions/{id}/run                run to the end in the background (202)
POST     /sessions/{id}/step               run one outer iteration: ``{"record": {...} | null}``
POST     /sessions/{id}/stop               stop after the current iteration (or close an idle session)
GET      /sessions/{id}/result             :class:`DMDO.RunResult` as JSON
GET      /sessions/{id}/history            iteration records
GET      /sessions/{id}/events?since=n     ``text/event-stream``: ``iteration_end`` ... then ``run_end``/``run_failed``
DELETE   /sessions/{id}                    remove an idle or finished session
GET      /work?worker=&tags=a,b&wait=s     workers: claim a job (pickled, ``X-DMDO-Job`` header) or 204
POST     /work/{job}?ok=1                  workers: post a job's pickled result (``ok=0``: a traceback)
=======  ================================  =======================================================================
"""
import hmac
import ipaddress
import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from ..app.session import Session, SessionError, record_to_dict, jsonable
from ..ports import RunObserver
from ..registry import RegistryError, blackboxes as blackbox_registry
from .remote import RemoteExecutor, TaskQueue

__all__ = ["DMDOService", "ServiceHandle", "serve", "start_service"]

TERMINAL_EVENTS = ("run_end", "run_failed", "closed")
SESSION_OPTIONS = ("mode", "sweep", "executor", "placement", "max_workers", "timeout")


class EventLog(RunObserver):
  """Collects a session's events for the ``/events`` stream."""

  def __init__(self) -> None:
    self.events: List[Tuple[str, Any]] = []
    self._cond = threading.Condition()

  def emit(self, name: str, data: Any) -> None:
    with self._cond:
      self.events.append((name, jsonable(data)))
      self._cond.notify_all()

  def wait_from(self, index: int, timeout: float) -> List[Tuple[str, Any]]:
    with self._cond:
      if len(self.events) <= index:
        self._cond.wait(timeout=timeout)
      return self.events[index:]

  def on_iteration_end(self, run, record) -> None:
    self.emit("iteration_end", record_to_dict(record))

  def on_run_end(self, run) -> None:
    self.emit("run_end", {"stop_reason": str(getattr(run, "stop", "")), "iterations": int(run.iter) + 1})


@dataclass
class ManagedSession:
  id: str
  session: Session
  events: EventLog
  created: float = field(default_factory=time.time)
  thread: Optional[threading.Thread] = None

  @property
  def running(self) -> bool:
    return self.thread is not None and self.thread.is_alive()

  def describe(self) -> Dict[str, Any]:
    s = self.session
    return {"session_id": self.id, "mdo_name": getattr(s.mdo, "mdo_name", None), "status": s.status,
            "iteration": s.iteration, "done": s.done, "running": self.running,
            "error": None if s.error is None else repr(s.error), "created": self.created}


class HTTPError(Exception):
  def __init__(self, status: int, message: str) -> None:
    super().__init__(message)
    self.status = status


class DMDOService:
  """Holds the sessions and the job queue; transport-independent (the HTTP handler below calls it).

  :param root_dir: each session works in ``<root_dir>/<session id>``.
  :param token: shared secret clients and workers must send; None disables authentication.
  :param lease_seconds: re-offer a claimed job to another worker after this long without a result (None: never).
  """

  def __init__(self, root_dir: str = "dmdo_service", token: Optional[str] = None,
               lease_seconds: Optional[float] = None) -> None:
    self.root_dir = root_dir
    self.token = token
    self.queue = TaskQueue(lease_seconds=lease_seconds)
    self.sessions: Dict[str, ManagedSession] = {}
    self._lock = threading.Lock()

  # ------------------------------------------------------------------------------------------------ sessions
  def get(self, session_id: str) -> ManagedSession:
    managed = self.sessions.get(session_id)
    if managed is None:
      raise HTTPError(404, f"Unknown session '{session_id}'")
    return managed

  def _resolve_blackboxes(self, mapping: Dict[str, Any]) -> Dict[str, Any]:
    resolved = {}
    for name, path in (mapping or {}).items():
      if not isinstance(path, str):
        raise HTTPError(400, f"Blackbox '{name}' must be given as a 'module:attr' import path")
      try:
        resolved[name] = blackbox_registry.get(path)
      except RegistryError as exc:
        raise HTTPError(400, str(exc))
    return resolved

  def _executor(self, options: Dict[str, Any]):
    spec = options.pop("executor", None)
    placement = options.pop("placement", None)
    timeout = options.pop("timeout", None)
    if placement and spec in (None, "remote"):
      spec = "remote"
    if spec == "remote":
      return RemoteExecutor(self.queue, placement=placement, timeout=timeout)
    if spec is not None and not isinstance(spec, str):
      raise HTTPError(400, "executor must be 'inprocess', 'process', 'remote' or a registered executor name")
    return spec

  def create(self, payload: Dict[str, Any]) -> ManagedSession:
    options = dict(payload.get("options") or {})
    unknown = set(options) - set(SESSION_OPTIONS)
    if unknown:
      raise HTTPError(400, f"Unknown session options {sorted(unknown)}; allowed: {list(SESSION_OPTIONS)}")
    mdo_name = str(payload.get("mdo_name") or "MDO")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", mdo_name):
      raise HTTPError(400, "mdo_name may only contain letters, digits, '_', '-' and '.'")
    session_id = uuid.uuid4().hex[:12]
    events = EventLog()
    common = dict(working_dir=f"{self.root_dir}/{session_id}", mdo_name=mdo_name,
                  blackboxes=self._resolve_blackboxes(payload.get("blackboxes")), executor=self._executor(options),
                  observers=[events], **options)
    try:
      if payload.get("setup") is not None:
        session = Session.from_dict(payload["setup"], **common)
      elif payload.get("setup_file"):
        session = Session.from_yaml(str(payload["setup_file"]), **common)
      else:
        raise HTTPError(400, "Give the problem definition as 'setup' (a dict) or 'setup_file' (a path)")
    except HTTPError:
      raise
    except Exception as exc:
      raise HTTPError(400, f"Could not build the problem: {exc!r}")
    managed = ManagedSession(id=session_id, session=session, events=events)
    with self._lock:
      self.sessions[session_id] = managed
    return managed

  def run(self, managed: ManagedSession) -> None:
    with self._lock:
      if managed.running:
        raise HTTPError(409, "The session is already running")
      if managed.session.done:
        raise HTTPError(409, f"The session is {managed.session.status}")

      def target():
        try:
          managed.session.run()
        except Exception as exc:
          managed.events.emit("run_failed", {"error": repr(exc)})

      managed.thread = threading.Thread(target=target, name=f"dmdo-session-{managed.id}", daemon=True)
      managed.thread.start()

  def step(self, managed: ManagedSession) -> Optional[Dict[str, Any]]:
    if managed.running:
      raise HTTPError(409, "The session is running; stop it before stepping")
    try:
      record = managed.session.step()
    except SessionError as exc:
      raise HTTPError(409, str(exc))
    except Exception as exc:
      managed.events.emit("run_failed", {"error": repr(exc)})
      raise HTTPError(500, f"The iteration failed: {exc!r}")
    return None if record is None else record_to_dict(record)

  def stop(self, managed: ManagedSession) -> None:
    if managed.running:
      managed.session.request_stop()
    elif not managed.session.done:
      managed.session.close()
      managed.events.emit("closed", {"iterations": managed.session.iteration})

  def delete(self, managed: ManagedSession) -> None:
    if managed.running:
      raise HTTPError(409, "The session is running; stop it first")
    self.stop(managed)
    with self._lock:
      self.sessions.pop(managed.id, None)

  def health(self) -> Dict[str, Any]:
    try:
      from importlib.metadata import version
      dmdo_version = version("DMDO")
    except Exception:
      dmdo_version = "unknown"
    return {"status": "ok", "version": dmdo_version, "sessions": len(self.sessions), "jobs": self.queue.stats()}


# ---------------------------------------------------------------------------------------------------- HTTP
_SESSION = re.compile(r"^/sessions/([0-9a-f]{12})(/[a-z]+)?$")
_JOB = re.compile(r"^/work/([A-Za-z0-9_.-]+)$")


class _Handler(BaseHTTPRequestHandler):
  """HTTP front end of a :class:`DMDOService` (bound per server by :func:`_make_handler`)."""

  server_version = "DMDO"
  service: DMDOService = None

  def log_message(self, fmt, *args):  # keep the console quiet; the run has its own log
    pass

  # -------------------------------------------------------------------------------------------- helpers
  def _authorized(self) -> bool:
    if not self.service.token:
      return True
    header = self.headers.get("Authorization", "")
    return header.startswith("Bearer ") and hmac.compare_digest(header[7:].encode(), self.service.token.encode())

  def _send(self, status: int, body: bytes, content_type: str, headers: Optional[Dict[str, str]] = None) -> None:
    self.send_response(status)
    self.send_header("Content-Type", content_type)
    self.send_header("Content-Length", str(len(body)))
    for key, value in (headers or {}).items():
      self.send_header(key, value)
    self.end_headers()
    self.wfile.write(body)

  def _json(self, status: int, payload: Any) -> None:
    self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

  def _body(self) -> bytes:
    length = int(self.headers.get("Content-Length") or 0)
    return self.rfile.read(length) if length else b""

  def _json_body(self) -> Dict[str, Any]:
    raw = self._body()
    if not raw:
      return {}
    try:
      data = json.loads(raw)
    except ValueError:
      raise HTTPError(400, "The request body is not valid JSON")
    if not isinstance(data, dict):
      raise HTTPError(400, "The request body must be a JSON object")
    return data

  def _dispatch(self, method: str) -> None:
    url = urlparse(self.path)
    query = {k: v[-1] for k, v in parse_qs(url.query).items()}
    try:
      if not self._authorized():
        raise HTTPError(401, "Missing or invalid token")
      self._route(method, url.path.rstrip("/") or "/", query)
    except HTTPError as exc:
      self._json(exc.status, {"error": str(exc)})
    except (BrokenPipeError, ConnectionResetError):
      pass
    except Exception as exc:  # pragma: no cover - unexpected server error
      self._json(500, {"error": repr(exc)})

  # -------------------------------------------------------------------------------------------- routes
  def _route(self, method: str, path: str, query: Dict[str, str]) -> None:  # noqa: C901
    if path == "/health" and method == "GET":
      return self._json(200, self.service.health())
    if path == "/sessions" and method == "GET":
      return self._json(200, [m.describe() for m in list(self.service.sessions.values())])
    if path == "/sessions" and method == "POST":
      managed = self.service.create(self._json_body())
      return self._json(201, managed.describe())
    if path == "/work" and method == "GET":
      tags = [t for t in query.get("tags", "").split(",") if t]
      wait = min(max(float(query.get("wait", 0) or 0), 0.0), 60.0)
      job = self.service.queue.claim(query.get("worker", "anonymous"), tags=tags, timeout=wait)
      if job is None:
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()
        return None
      return self._send(200, job[1], "application/octet-stream", {"X-DMDO-Job": job[0]})
    match = _JOB.match(path)
    if match and method == "POST":
      accepted = self.service.queue.complete(match.group(1), query.get("ok", "1") == "1", self._body())
      return self._json(200, {"accepted": accepted})
    match = _SESSION.match(path)
    if not match:
      raise HTTPError(404, f"No route for {method} {path}")
    managed = self.service.get(match.group(1))
    action = (match.group(2) or "").lstrip("/")
    if method == "GET" and action == "":
      return self._json(200, managed.describe())
    if method == "DELETE" and action == "":
      self.service.delete(managed)
      return self._json(200, {"deleted": managed.id})
    if method == "POST" and action == "run":
      self.service.run(managed)
      return self._json(202, managed.describe())
    if method == "POST" and action == "step":
      return self._json(200, {"record": self.service.step(managed), "status": managed.describe()})
    if method == "POST" and action == "stop":
      self.service.stop(managed)
      return self._json(202, managed.describe())
    if method == "GET" and action == "result":
      return self._json(200, managed.session.result().to_dict())
    if method == "GET" and action == "history":
      return self._json(200, [record_to_dict(r) for r in list(managed.session.history)])
    if method == "GET" and action == "events":
      return self._stream(managed, int(query.get("since", 0) or 0))
    raise HTTPError(404, f"No route for {method} {path}")

  def _stream(self, managed: ManagedSession, index: int) -> None:
    self.send_response(200)
    self.send_header("Content-Type", "text/event-stream")
    self.send_header("Cache-Control", "no-cache")
    self.end_headers()
    while True:
      events = managed.events.wait_from(index, timeout=15.0)
      if not events:
        if managed.session.done and not managed.running:
          return
        self.wfile.write(b": keep-alive\n\n")
        self.wfile.flush()
        continue
      for name, data in events:
        self.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode("utf-8"))
        index += 1
        if name in TERMINAL_EVENTS:
          self.wfile.flush()
          return
      self.wfile.flush()

  def do_GET(self):
    self._dispatch("GET")

  def do_POST(self):
    self._dispatch("POST")

  def do_DELETE(self):
    self._dispatch("DELETE")



def _make_handler(service: DMDOService):
  return type("Handler", (_Handler,), {"service": service})


def _is_loopback(host: str) -> bool:
  if host in ("localhost", ""):
    return host == "localhost"
  try:
    return ipaddress.ip_address(host).is_loopback
  except ValueError:
    return False


@dataclass
class ServiceHandle:
  """A service running in a background thread (see :func:`start_service`)."""
  service: DMDOService
  server: ThreadingHTTPServer
  thread: threading.Thread

  @property
  def url(self) -> str:
    host, port = self.server.server_address[:2]
    return f"http://{host}:{port}"

  def shutdown(self) -> None:
    self.server.shutdown()
    self.server.server_close()
    self.thread.join(timeout=10)


def _make_server(host: str, port: int, service: DMDOService, allow_insecure: bool) -> ThreadingHTTPServer:
  if not service.token and not _is_loopback(host) and not allow_insecure:
    raise ValueError(f"Refusing to listen on {host} without a token: jobs are pickled, so anyone who can reach the "
                     "service could run code on it and its workers. Set a token (or allow_insecure=True).")
  server = ThreadingHTTPServer((host, port), _make_handler(service))
  server.daemon_threads = True
  return server


def start_service(host: str = "127.0.0.1", port: int = 0, root_dir: str = "dmdo_service", token: Optional[str] = None,
                  lease_seconds: Optional[float] = None, allow_insecure: bool = False) -> ServiceHandle:
  """Start the service in a background thread (``port=0``: any free port); returns a handle with ``url``."""
  service = DMDOService(root_dir=root_dir, token=token, lease_seconds=lease_seconds)
  server = _make_server(host, port, service, allow_insecure)
  thread = threading.Thread(target=server.serve_forever, name="dmdo-service", daemon=True)
  thread.start()
  return ServiceHandle(service=service, server=server, thread=thread)


def serve(host: str = "127.0.0.1", port: int = 8400, root_dir: str = "dmdo_service", token: Optional[str] = None,
          lease_seconds: Optional[float] = None, allow_insecure: bool = False) -> None:
  """Run the service in the foreground until interrupted."""
  service = DMDOService(root_dir=root_dir, token=token, lease_seconds=lease_seconds)
  server = _make_server(host, port, service, allow_insecure)
  print(f"DMDO service listening on http://{host}:{server.server_address[1]} (sessions in {root_dir})")
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    pass
  finally:
    server.server_close()
