"""Optional service layer: DMDO sessions over HTTP, and remote workers for subproblem solves.

Standard library only (``http.server``/``urllib``); nothing here is imported by ``import DMDO``.

- :func:`serve` / :func:`start_service`: the HTTP service (``dmdo serve`` or ``python -m DMDO.service serve``);
- :class:`Client`: create and drive sessions from any Python process (any other language can use the HTTP API);
- :func:`run_worker`: a worker that claims subproblem jobs (``dmdo worker --connect URL --tags aero``);
- :class:`RemoteExecutor`: the :class:`DMDO.ports.Executor` that sends a session's solves to those workers.

Security: jobs and results are pickled, so a service and its workers must trust each other and every client that
holds the token. Run the service inside a trusted network, set a token (``--token`` / ``DMDO_TOKEN``) whenever it
listens on anything other than the loopback interface (it refuses to otherwise), and put TLS in front of it (a reverse
proxy) when traffic leaves a trusted network. Session problem files given as ``setup_file`` are read from the
service's file system.
"""
from .client import Client, RemoteSession, ServiceError
from .remote import RemoteExecutor, RemoteTaskError, TaskQueue
from .server import DMDOService, ServiceHandle, serve, start_service
from .worker import run_worker

__all__ = ["Client", "RemoteSession", "ServiceError", "RemoteExecutor", "RemoteTaskError", "TaskQueue",
           "DMDOService", "ServiceHandle", "serve", "start_service", "run_worker"]
