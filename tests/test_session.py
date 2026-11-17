"""Session API: run / step / ask-tell on the basic two-discipline problem."""
import contextlib
import csv
import json
import os
import sys

import pytest

from DMDO import Session, SessionError, build_mdo, load_setup, main, solve_task

TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TESTS)
sys.path.insert(0, os.path.join(TESTS, "golden"))
import test_basic  # noqa: E402
from _harness import _close_file_handlers_under, snapshot  # noqa: E402

SETUP = os.path.join(TESTS, "test_files", "Basic_MDO.yaml")
BLACKBOXES = {"A1": test_basic.A1, "A2": test_basic.A2, "opt1": test_basic.opt1, "opt2": test_basic.opt2}
BUDGET = 5


@pytest.fixture
def data():
  d = load_setup(SETUP)
  d["coord"]["c1"]["budget"] = BUDGET
  return d


@pytest.fixture
def wd(tmp_path):
  yield str(tmp_path)
  _close_file_handlers_under(str(tmp_path))


@pytest.fixture(autouse=True)
def quiet():
  with open(os.devnull, "w", encoding="utf-8") as devnull, contextlib.redirect_stdout(devnull):
    yield


def _session(data, wd, name, **options):
  return Session.from_dict(data, working_dir=wd, mdo_name=name, blackboxes=BLACKBOXES, **options)


def _ask_tell(session, reorder=lambda results: results):
  batches = []
  while not session.done:
    tasks = session.ask()
    if not tasks:
      break
    batches.append(len(tasks))
    session.tell(reorder([solve_task(t) for t in tasks]))
  return batches


def test_session_run_matches_mdo_run(data, wd):
  mdo = build_mdo(data, working_dir=wd, mdo_name="direct", blackboxes=BLACKBOXES)
  mdo.run(file=os.path.join(mdo.post_dir, "coordination_summary.out"))
  session = _session(data, wd, "session")
  result = session.run()
  assert snapshot(session.mdo) == snapshot(mdo)
  assert result.status == "finished" and session.done
  assert result.iterations == BUDGET == len(result.history)
  assert result.stop_reason == "Iteration budget exhausted"


def test_ask_tell_matches_run(data, wd):
  reference = _session(data, wd, "ref")
  reference.run()
  session = _session(data, wd, "asktell")
  batches = _ask_tell(session)
  assert batches == [1] * (2 * BUDGET)  # Gauss-Seidel: one subproblem at a time
  assert snapshot(session.mdo) == snapshot(reference.mdo)
  assert session.ask() == []


def test_jacobi_ask_tell_accepts_results_in_any_order(data, wd):
  reference = _session(data, wd, "ref", sweep="jacobi")
  reference.run()
  session = _session(data, wd, "asktell", sweep="jacobi")
  batches = _ask_tell(session, reorder=lambda results: list(reversed(results)))
  assert batches == [2] * BUDGET  # Jacobi: every subproblem of the iteration in one batch
  assert snapshot(session.mdo) == snapshot(reference.mdo)


def test_mode_defaults_to_the_coordinator_mode_of_the_problem_definition(data, wd):
  from DMDO.workflow.execution import InProcessExecutor, ProcessExecutor

  assert _session(data, wd, "yaml_serial").mode == "serial"
  data["coord"]["c1"]["mode"] = "parallel"
  session = _session(data, wd, "yaml_parallel")
  assert session.mode == "parallel" and session.mdo.default_mode() == "parallel"
  executor = session._get_executor()
  try:
    assert isinstance(executor, ProcessExecutor)
  finally:
    session._release_executor()
  # An explicit mode still takes precedence over the problem definition.
  explicit = _session(data, wd, "explicit_serial", mode="serial")
  assert explicit.mode == "serial" and isinstance(explicit._get_executor(), InProcessExecutor)


def test_parallel_mode_from_the_problem_definition_sweeps_jacobi(data, wd):
  data["coord"]["c1"]["mode"] = "parallel"
  session = _session(data, wd, "yaml_jacobi")
  batches = _ask_tell(session)
  assert batches == [2] * BUDGET  # Jacobi: every subproblem of the iteration in one batch
  assert session.mdo.mode == "parallel"


def test_step_yields_one_record_per_outer_iteration(data, wd):
  session = _session(data, wd, "step")
  records = []
  while (record := session.step()) is not None:
    records.append(record)
  assert [r.iteration for r in records] == list(range(BUDGET))
  assert session.done and session.step() is None
  assert session.result().history == records


def test_ask_is_idempotent_and_tell_needs_pending_tasks(data, wd):
  session = _session(data, wd, "states")
  with pytest.raises(SessionError):
    session.tell([])
  first = session.ask()
  assert [t.index for t in session.ask()] == [t.index for t in first]
  with pytest.raises(ValueError, match="Expected results"):
    session.tell([])
  assert session.status == "failed" and session.done


def test_request_stop_ends_after_the_current_iteration(data, wd):
  session = _session(data, wd, "stop")
  session.step()
  session.request_stop()
  session.step()
  result = session.result()
  assert session.done and result.iterations == 2
  assert result.stop_reason == "Stopped on request"


def test_mixing_ask_tell_and_step(data, wd):
  reference = _session(data, wd, "ref")
  reference.run()
  session = _session(data, wd, "mixed")
  tasks = session.ask()  # first subproblem of iteration 0
  session.tell([solve_task(t) for t in tasks])
  session.step()  # finishes iteration 0 with the executor
  session.run()
  assert snapshot(session.mdo) == snapshot(reference.mdo)


def test_result_is_json_serialisable(data, wd):
  result = _session(data, wd, "json").run()
  payload = json.loads(json.dumps(result.to_dict()))
  assert payload["iterations"] == BUDGET
  assert set(payload["variables"]) == {"u_1", "v_1", "a_1", "b_1", "u_2", "w_2", "a_2", "b_2"}
  assert len(payload["history"]) == BUDGET


def test_from_yaml_names_the_run_after_the_file(wd):
  session = Session.from_yaml(SETUP, working_dir=wd, blackboxes=BLACKBOXES)
  assert session.mdo.mdo_name == "Basic_MDO"
  assert os.path.isdir(os.path.join(wd, "Basic_MDO_post"))
  session.close()
  assert session.status == "closed" and session.ask() == []


def _history_iterations(post_dir):
  with open(os.path.join(post_dir, "Coordination_history.out"), newline="") as fh:
    rows = [row for row in csv.reader(fh) if row]
  return [int(float(row[1])) for row in rows[1:]]


def _interrupted_checkpoint(data, wd, name, steps):
  """Run ``steps`` outer iterations, then drop the session (the checkpoint is from the start of the last one)."""
  session = _session(data, wd, name)
  for _ in range(steps):
    session.step()
  session.close()
  return os.path.join(session.mdo.post_dir, "coord_chk.pkl")


def test_resumed_session_matches_an_uninterrupted_run(data, wd):
  reference = _session(data, wd, "ref")
  reference.run()
  checkpoint = _interrupted_checkpoint(data, wd, "resume", steps=4)  # checkpoint at the start of iteration 3
  resumed = Session.from_checkpoint(checkpoint)
  assert resumed.mdo.iter == 3
  result = resumed.run()
  assert snapshot(resumed.mdo) == snapshot(reference.mdo)
  assert result.iterations == BUDGET
  assert [r.iteration for r in result.history] == [3, 4]
  # Rows written after the checkpoint (iteration 3 of the interrupted run) are replaced, not duplicated.
  assert _history_iterations(resumed.mdo.post_dir) == list(range(BUDGET))


def test_main_resume_keeps_the_checkpoint_and_continues(data, wd):
  reference = _session(data, wd, "ref")
  reference.run()
  checkpoint = _interrupted_checkpoint(data, wd, "main_resume", steps=3)
  mdo = main({"is_resume": True, "restart_file": checkpoint, "working_dir": wd, "mdo_name": "main_resume"})
  assert snapshot(mdo) == snapshot(reference.mdo)
  assert _history_iterations(mdo.post_dir) == list(range(BUDGET))


def test_from_checkpoint_loads_a_resumable_session(data, wd):
  session = _session(data, wd, "chk")
  session.step()
  session.step()
  checkpoint = os.path.join(session.mdo.post_dir, "coord_chk.pkl")
  session.close()
  resumed = Session.from_checkpoint(checkpoint)
  assert resumed.status == "created" and resumed._resume
  assert resumed.mdo.mdo_name == "chk"

def test_relative_working_dir_and_feasible_result(data, wd, monkeypatch):
  monkeypatch.chdir(wd)
  session = _session(data, "runs", "relative")
  session.step()
  post = os.path.join("runs", "relative_post")
  # Subproblem outputs sit next to the coordination history, not in a copy of the relative path.
  assert os.path.isdir(os.path.join(post, "coordination_summary_post", "SP_1"))
  assert not os.path.exists(os.path.join(post, "runs"))
  # OMADS reports a feasible solve as h = 1e-15; it must not be reported as infeasible.
  assert session.result().feasible
  with open(os.path.join(post, "Coordination_history.out")) as fh:
    assert "Infeasible" not in fh.read()
