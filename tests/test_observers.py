"""Run observers (coordination-loop hooks) and coordination adaptation policies."""
import contextlib
import os
import pickle
import sys
import tempfile
from types import SimpleNamespace

import pytest

from DMDO import (ADMM, MDO, CoordinationPolicy, IterationRecord, NeutralPolicy, RunObserver, SpikeAdaptationPolicy,
                  policies, policy_for)
from DMDO.workflow.observers import HISTORY_FILE, default_observers

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden"))
from _harness import _close_file_handlers_under, _in_dir, snapshot  # noqa: E402

HOOKS_PER_ITERATION = ["on_iteration_start", "on_sweep_end", "on_iteration_end", "on_multipliers_updated"]


class Recorder(RunObserver):
  def __init__(self):
    self.calls = []
    self.records = []

  def on_run_start(self, run):
    self.calls.append(("on_run_start", None))

  def on_iteration_start(self, run, iteration):
    self.calls.append(("on_iteration_start", iteration))

  def on_sweep_end(self, run, iteration):
    self.calls.append(("on_sweep_end", iteration))

  def on_iteration_end(self, run, record):
    self.calls.append(("on_iteration_end", record.iteration))
    self.records.append(record)

  def on_multipliers_updated(self, run, iteration):
    self.calls.append(("on_multipliers_updated", iteration))

  def on_run_end(self, run):
    self.calls.append(("on_run_end", None))


def _run_basic(budget=4, setup=None, inspect=None, **run_kwargs):
  """Run the basic two-discipline problem; ``setup(mdo)`` runs before ``MDO.run``, ``inspect(mdo, wd)`` after."""
  import test_basic

  out = {}
  original = MDO.run

  def wrapped(self, *args, **kwargs):
    self.Coordinator.budget = budget
    if setup is not None:
      setup(self)
    kwargs.update(run_kwargs)
    result = original(self, *args, **kwargs)
    out["snapshot"] = snapshot(self)
    if inspect is not None:
      out["inspect"] = inspect(self)
    return result

  with tempfile.TemporaryDirectory(prefix="dmdo_obs_", ignore_cleanup_errors=True) as wd:
    os.makedirs(os.path.join(wd, "tests", "test_files"))
    MDO.run = wrapped
    try:
      with _in_dir(wd), open(os.devnull, "w", encoding="utf-8") as devnull, contextlib.redirect_stdout(devnull):
        test_basic.test_basic_MDO()
    finally:
      MDO.run = original
      _close_file_handlers_under(wd)
  return out


# --------------------------------------------------------------------------------------------- observers
def test_observer_hooks_follow_the_iteration_order():
  rec = Recorder()
  out = _run_basic(observers=[rec])
  n = out["snapshot"]["iterations"]
  expected = [("on_run_start", None)]
  for it in range(n):
    expected += [(hook, it) for hook in HOOKS_PER_ITERATION]
  expected.append(("on_run_end", None))
  assert rec.calls == expected


def test_iteration_records_match_the_run():
  rec = Recorder()
  snap = _run_basic(observers=[rec])["snapshot"]
  assert all(isinstance(r, IterationRecord) for r in rec.records)
  assert [r.iteration for r in rec.records] == list(range(snap["iterations"]))
  assert [float(r.objective) for r in rec.records] == snap["objective_history"]


def test_duck_typed_observer_with_a_single_hook():
  seen = []
  observer = SimpleNamespace(on_iteration_end=lambda run, record: seen.append(record.iteration))
  snap = _run_basic(observers=[observer])["snapshot"]
  assert seen == list(range(snap["iterations"]))


def test_observers_do_not_change_the_results():
  plain = _run_basic()["snapshot"]
  observed = _run_basic(observers=[Recorder()])["snapshot"]
  assert observed == plain


def test_history_file_and_checkpoint_are_written_by_the_default_observers():
  def inspect(mdo):
    with open(mdo.Ol_file) as fh:
      rows = [line for line in fh.read().splitlines() if line.strip()]
    with open(os.path.join(mdo.post_dir, "coord_chk.pkl"), "rb") as fh:
      checkpoint = pickle.load(fh)
    return os.path.basename(mdo.Ol_file), rows, checkpoint

  rec = Recorder()
  out = _run_basic(observers=[rec], inspect=inspect)
  name, rows, checkpoint = out["inspect"]
  assert name == HISTORY_FILE
  assert len(rows) == 1 + out["snapshot"]["iterations"]  # header + one row per outer iteration
  assert rows[0].split(",")[:3] == ["Time", "Iteration #".rjust(30), "Max. inconsistency".rjust(30)]
  # Observers are passed to run(), never stored on the run, so they are not pickled into checkpoints.
  assert isinstance(checkpoint, MDO)
  assert not any(isinstance(v, RunObserver) for v in vars(checkpoint).values())


def test_worst_link_label_names_the_two_linked_subproblems():
  from DMDO.workflow.observers import worst_link_label

  def inspect(mdo):
    coord = mdo.Coordinator
    labels = [worst_link_label(coord, link) for link in range(coord._linker.shape[1])]
    with open(mdo.Ol_file) as fh:
      rows = [[c.strip() for c in line.split(",")] for line in fh.read().splitlines() if line.strip()]
    return coord, labels, rows

  rec = Recorder()
  coord, labels, rows = _run_basic(observers=[rec], inspect=inspect)["inspect"]
  for link, label in enumerate(labels):
    a = coord.master_vars[coord._linker[0, link] - 1]
    b = coord.master_vars[coord._linker[1, link] - 1]
    assert a.sp_index != b.sp_index
    assert label == f"{a.name}_{a.sp_index} to {b.name}_{b.sp_index}"
  column = rows[0].index("Coupling_with_qmax")
  assert [row[column] for row in rows[1:]] == [labels[r.worst_link] for r in rec.records]


def test_default_observers_depend_on_the_ai_regularizer():
  def names(is_ai):
    run = SimpleNamespace(Coordinator=SimpleNamespace(is_AI_regularizer=is_ai))
    return [type(o).__name__ for o in default_observers(run)]

  assert names(False) == ["CheckpointWriter", "ConsoleReporter", "CoordinationHistoryRecorder"]
  assert names(True) == ["AIRegularizerHooks", "RollbackGuard", "CheckpointWriter", "ConsoleReporter",
                         "CoordinationHistoryRecorder"]


def test_prepare_post_accepts_a_file_argument(tmp_path):
  # app.main calls prepare_post(path) for run modes other than "run"/"build".
  mdo = object.__new__(MDO)
  mdo.post_dir = str(tmp_path / "post")
  mdo.Coordinator = SimpleNamespace()  # no master_vars yet: no header is written
  mdo.prepare_post(str(tmp_path / "ignored.out"))
  assert os.path.isdir(mdo.post_dir)
  assert mdo.Ol_file == os.path.join(mdo.post_dir, HISTORY_FILE)


# --------------------------------------------------------------------------------------------- policies
def _coord(**attrs):
  coord = object.__new__(ADMM)
  coord.__dict__.update(attrs)
  return coord


def test_policy_defaults_follow_the_ai_regularizer_flag():
  assert isinstance(policy_for(_coord(is_AI_regularizer=False)), NeutralPolicy)
  assert isinstance(policy_for(_coord(is_AI_regularizer=True)), SpikeAdaptationPolicy)
  assert isinstance(policy_for(_coord(is_AI_regularizer=True, adaptation_policy="none")), NeutralPolicy)
  assert isinstance(policy_for(_coord(is_AI_regularizer=False, adaptation_policy="AI_Regularizer")),
                    SpikeAdaptationPolicy)


def test_neutral_policy_commits_neutral_scales():
  coord = _coord(is_AI_regularizer=False)
  assert coord.assess_iteration(3) == (False, 0, -1)
  assert policy_for(coord).multiplier_scales(coord, 3) == (1.0, 1.0, False)
  assert coord._effective_budget_scale == 1.0
  assert coord._applied_scales == (1.0, 1.0, 1.0, "")


def test_legacy_coordinator_entry_points_delegate_to_the_policy():
  coord = _coord(is_AI_regularizer=False)
  assert coord._get_expected_spike_event(1) == (False, 0, -1)
  assert coord._compute_ai_scales(1) == (1.0, 1.0, 1.0, False)
  assert coord._get_ai_spike_multiplier_scales(1) == (1.0, 1.0, False)


class FreezePenaltyPolicy(CoordinationPolicy):
  """Never grows the penalty weights."""

  def compute_scales(self, coord, iteration):
    return 0.0, 1.0, 1.0, False


@pytest.fixture
def freeze_policy():
  policies.register("unit_test_freeze", FreezePenaltyPolicy, override=True)
  yield "unit_test_freeze"
  policies.unregister("unit_test_freeze")


def test_custom_policy_plugs_into_the_penalty_update(freeze_policy):
  def setup(mdo):
    mdo.Coordinator.adaptation_policy = freeze_policy

  plain = _run_basic()["snapshot"]
  frozen = _run_basic(setup=setup)["snapshot"]
  assert max(plain["final_w"]) > 1.0
  assert frozen["final_w"] == [1.0] * len(frozen["final_w"])
