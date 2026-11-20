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
"""``dmdo`` command line: ``dmdo serve`` starts the HTTP service, ``dmdo worker`` a remote worker."""
import argparse
import os
import sys
from typing import List, Optional


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(prog="dmdo", description="DMDO service and remote workers")
  sub = parser.add_subparsers(dest="command", required=True)

  srv = sub.add_parser("serve", help="run the DMDO HTTP service")
  srv.add_argument("--host", default="127.0.0.1", help="interface to listen on (default: 127.0.0.1)")
  srv.add_argument("--port", type=int, default=8400)
  srv.add_argument("--root", default="dmdo_service", help="directory for session working directories")
  srv.add_argument("--token", default=os.environ.get("DMDO_TOKEN"), help="shared secret (default: $DMDO_TOKEN)")
  srv.add_argument("--lease", type=float, default=None,
                   help="seconds before a claimed but unfinished job is offered to another worker")
  srv.add_argument("--allow-insecure", action="store_true",
                   help="allow listening on a non-loopback interface without a token")

  wrk = sub.add_parser("worker", help="run a remote worker that solves subproblem jobs")
  wrk.add_argument("--connect", required=True, help="service URL, e.g. http://dmdo-host:8400")
  wrk.add_argument("--tags", default="", help="comma-separated tags of the jobs this worker accepts")
  wrk.add_argument("--token", default=os.environ.get("DMDO_TOKEN"), help="shared secret (default: $DMDO_TOKEN)")
  wrk.add_argument("--name", default=None, help="worker name shown to the service")
  wrk.add_argument("--max-jobs", type=int, default=None, help="exit after this many jobs")
  wrk.add_argument("--path", action="append", default=[],
                   help="directory to add to sys.path (where the problem's analysis modules live)")
  return parser


def main(argv: Optional[List[str]] = None) -> int:
  args = _parser().parse_args(argv)
  from ..utils._encoding import fix_encoding  # this process is ours: UTF-8 console for the OMADS progress bars
  fix_encoding()
  if args.command == "serve":
    from .server import serve
    serve(host=args.host, port=args.port, root_dir=args.root, token=args.token, lease_seconds=args.lease,
          allow_insecure=args.allow_insecure)
    return 0
  for path in args.path:
    sys.path.insert(0, os.path.abspath(path))
  from .worker import run_worker
  tags = [t.strip() for t in args.tags.split(",") if t.strip()]
  print(f"DMDO worker connected to {args.connect} (tags: {', '.join(tags) or 'any'})")
  try:
    run_worker(args.connect, tags=tags, token=args.token, name=args.name, max_jobs=args.max_jobs)
  except KeyboardInterrupt:
    pass
  return 0


if __name__ == "__main__":
  sys.exit(main())
