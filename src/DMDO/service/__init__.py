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
