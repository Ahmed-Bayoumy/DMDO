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
"""Service layer: HTTP sessions, live events, remote workers (in threads), token auth and the job queue."""
import contextlib
import json
import os
import sys
import threading
import time
import urllib.request

import pytest

from DMDO import Session, load_setup
from DMDO.service import Client, RemoteExecutor, ServiceError, TaskQueue, run_worker, start_service
from DMDO.service.server import _is_loopback

TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TESTS)
sys.path.insert(0, os.path.join(TESTS, "golden"))
import test_basic  # noqa: E402
from _harness import _close_file_handlers_under  # noqa: E402

SETUP = os.path.join(TESTS, "test_files", "Basic_MDO.yaml")
BLACKBOX_PATHS = {"A1": "test_basic:A1", "A2": "test_basic:A2", "opt1": "test_basic:opt1", "opt2": "test_basic:opt2"}
BUDGET = 4
TOKEN = "unit-test-token"


@pytest.fixture
def setup():
  data = load_setup(SETUP)
  data["coord"]["c1"]["budget"] = BUDGET
  data["MDO"]["display"] = False
  return data


@pytest.fixture(autouse=True)
def quiet():
  with open(os.devnull, "w", encoding="utf-8") as devnull, contextlib.redirect_stdout(devnull):
    yield


@pytest.fixture
def service(tmp_path):
  handle = start_service(root_dir=str(tmp_path / "service"), token=TOKEN)
  yield handle
  handle.shutdown()
  _close_file_handlers_under(str(tmp_path))


@pytest.fixture
def client(service):
  return Client(service.url, token=TOKEN)


@contextlib.contextmanager
def workers(url, *tag_sets):
  stop = threading.Event()
  threads = [threading.Thread(target=run_worker, args=(url,), daemon=True,
                              kwargs=dict(tags=tags, token=TOKEN, poll_wait=0.5, stop_event=stop, name=f"w{i}"))
             for i, tags in enumerate(tag_sets)]
  for t in threads:
    t.start()
  try:
    yield
  finally:
    stop.set()
    for t in threads:
      t.join(timeout=10)


def _local_reference(setup, wd, sweep="gauss_seidel"):
  bb = {"A1": test_basic.A1, "A2": test_basic.A2, "opt1": test_basic.opt1, "opt2": test_basic.opt2}
  session = Session.from_dict(setup, working_dir=wd, mdo_name="local", blackboxes=bb, sweep=sweep)
  session.run()
  return session


# --------------------------------------------------------------------------------------------- HTTP sessions
def test_health_and_token(service, client):
  assert client.health()["status"] == "ok"
  with pytest.raises(ServiceError) as err:
    Client(service.url, token="wrong").health()
  assert err.value.status == 401
  with pytest.raises(ServiceError):
    Client(service.url).sessions()


def test_step_over_http_matches_a_local_session(setup, client, service, tmp_path):
  remote = client.create_session(setup=setup, mdo_name="http_step", blackboxes=BLACKBOX_PATHS)
  records = []
  while (record := remote.step()) is not None:
    records.append(record)
  assert [r["iteration"] for r in records] == list(range(BUDGET))
  result = remote.result()
  assert result["status"] == "finished" and result["iterations"] == BUDGET
  reference = _local_reference(setup, str(tmp_path / "ref"))
  local = reference.result().to_dict()
  assert result["variables"] == local["variables"]
  assert result["objective"] == local["objective"]
  assert remote.history() == local["history"]


def test_run_in_background_streams_events(setup, client):
  remote = client.create_session(setup=setup, mdo_name="http_run", blackboxes=BLACKBOX_PATHS)
  remote.run()
  events = list(remote.events())
  assert [e["event"] for e in events] == ["iteration_end"] * BUDGET + ["run_end"]
  assert events[-1]["data"]["iterations"] == BUDGET
  assert remote.status()["status"] == "finished"
  with pytest.raises(ServiceError) as err:
    remote.run()
  assert err.value.status == 409


def test_bad_requests_are_rejected(setup, client):
  for kwargs, fragment in [
    (dict(setup=setup, blackboxes={"A1": "no_such_module:fn"}), "Could not import"),
    (dict(setup=setup, frobnicate=True), "Unknown session options"),
    (dict(), "setup"),
    (dict(setup=setup, mdo_name="../escape"), "mdo_name"),
  ]:
    with pytest.raises(ServiceError) as err:
      client.create_session(**kwargs)
    assert err.value.status == 400 and fragment in err.value.message
  with pytest.raises(ServiceError) as err:
    client.session("0123456789ab").status()
  assert err.value.status == 404


def test_stop_and_delete(setup, client):
  remote = client.create_session(setup=setup, mdo_name="http_stop", blackboxes=BLACKBOX_PATHS)
  remote.step()
  remote.stop()  # idle session: closed right away
  assert remote.status()["status"] == "closed"
  assert [e["event"] for e in remote.events()][-1] == "closed"
  remote.delete()
  assert all(s["session_id"] != remote.id for s in client.sessions())


# --------------------------------------------------------------------------------------------- remote workers
def test_remote_workers_with_placement_match_local_jacobi(setup, client, service, tmp_path):
  with workers(service.url, ["aero"], ["structures"]):
    remote = client.create_session(setup=setup, mdo_name="remote", blackboxes=BLACKBOX_PATHS, sweep="jacobi",
                                   placement={1: "aero", 2: "structures"}, timeout=120)
    result = remote.run(wait=True)
  assert result["status"] == "finished"
  reference = _local_reference(setup, str(tmp_path / "ref"), sweep="jacobi")
  assert result["variables"] == reference.result().to_dict()["variables"]
  assert service.service.queue.stats()["pending"] == 0


def test_remote_executor_reports_worker_failures(service):
  queue = service.service.queue
  with workers(service.url, []):
    with pytest.raises(Exception, match="ZeroDivisionError"):
      RemoteExecutor(queue, timeout=30).map(_divide_by_zero, [1])
    assert RemoteExecutor(queue, timeout=30).map(abs, [-3, 4, -5]) == [3, 4, 5]


def _divide_by_zero(x):
  return x / 0


# --------------------------------------------------------------------------------------------- queue and security
def test_task_queue_tags():
  q = TaskQueue()
  aero = q.put(b"a", tag="aero")
  anywhere = q.put(b"b")
  assert q.claim("w1", tags=["structures"]) == (anywhere, b"b")
  assert q.claim("w1", tags=["structures"]) is None  # the aero job is not offered to other tags
  assert q.claim("w2", tags=["aero"]) == (aero, b"a")
  assert q.complete(aero, True, b"done") and not q.complete(aero, True, b"again")


def test_task_queue_lease_re_offers_abandoned_jobs():
  q = TaskQueue(lease_seconds=0.05)
  job = q.put(b"a")
  assert q.claim("w1")[0] == job
  assert q.claim("w2") is None          # still leased to w1
  time.sleep(0.1)
  assert q.claim("w2")[0] == job        # w1 never answered: offered again
  assert q.complete(job, True, b"done")


def test_service_refuses_public_interfaces_without_token(tmp_path):
  with pytest.raises(ValueError, match="token"):
    start_service(host="0.0.0.0", root_dir=str(tmp_path))
  assert _is_loopback("127.0.0.1") and _is_loopback("localhost") and not _is_loopback("0.0.0.0")


def test_events_endpoint_is_plain_server_sent_events(setup, service, client):
  remote = client.create_session(setup=setup, mdo_name="sse", blackboxes=BLACKBOX_PATHS)
  remote.step()
  remote.stop()
  req = urllib.request.Request(f"{service.url}/sessions/{remote.id}/events",
                               headers={"Authorization": f"Bearer {TOKEN}"})
  with urllib.request.urlopen(req, timeout=30) as resp:
    assert resp.headers["Content-Type"] == "text/event-stream"
    text = resp.read().decode()
  blocks = [b for b in text.split("\n\n") if b.strip()]
  assert blocks[0].startswith("event: iteration_end\ndata: ")
  assert json.loads(blocks[0].split("data: ", 1)[1])["iteration"] == 0
  assert blocks[-1].startswith("event: closed")
