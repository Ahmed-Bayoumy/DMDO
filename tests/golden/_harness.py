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
"""Golden-trace harness: records the outer-iteration history of reference MDO runs.

The traces pin the numerical behaviour of DMDO so that refactoring steps can be
checked for unintended changes. Record with ``python tests/golden/record_traces.py``;
compare with ``pytest tests/golden``.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
from typing import Any, Callable, Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
TRACE_DIR = os.path.join(HERE, "traces")

for _p in (os.path.join(REPO, "tests"), os.path.join(REPO, "tests", "GP")):
  if _p not in sys.path:
    sys.path.insert(0, _p)


def _to_float_list(values: Any) -> List[float]:
  arr = np.asarray(values, dtype=float).ravel()
  return [float(x) for x in arr]


def snapshot(mdo) -> Dict[str, Any]:
  """Extract a JSON-serialisable fingerprint of a finished MDO run."""
  coord = mdo.Coordinator
  snap: Dict[str, Any] = {
    "iterations": int(mdo.iter) + 1,
    "objective_history": _to_float_list(getattr(coord, "objective_history", [])),
    "qmax_history": _to_float_list(mdo.tab_inc),
    "final_q": _to_float_list(coord.q),
    "final_w": _to_float_list(coord.w),
    "final_v": _to_float_list(coord.v),
    "master_vars": [
      {"name": v.name, "sp_index": int(v.sp_index), "value": _to_float_list(v.value)}
      for v in coord.master_vars
    ],
  }
  if getattr(coord, "is_AI_regularizer", False):
    snap["spike_prob"] = float(getattr(coord, "spike_prob", np.nan))
    snap["alarm_detected"] = bool(getattr(coord, "alarm_detected", False))
  return snap


@contextlib.contextmanager
def capture_runs():
  """Patch ``MDO.run`` so every completed run's snapshot is appended to the yielded list."""
  from DMDO import MDO

  captured: List[Dict[str, Any]] = []
  original = MDO.run

  def wrapped(self, *args, **kwargs):
    out = original(self, *args, **kwargs)
    captured.append(snapshot(self))
    return out

  MDO.run = wrapped
  try:
    yield captured
  finally:
    MDO.run = original


@contextlib.contextmanager
def _in_dir(path: str):
  old = os.getcwd()
  os.chdir(path)
  try:
    yield
  finally:
    os.chdir(old)


def _case_basic_poll():
  import test_basic
  test_basic.test_basic_MDO()


def _case_sellar_scipy():
  import test_basic
  test_basic.Sellar_scipy()


def _case_sellar_poll():
  import test_basic
  test_basic.Sellar_OMADS_POLL()


def _case_sellar_mads():
  import test_basic
  test_basic.Sellar_OMADS_MADS()


def _case_speed_reducer_scipy():
  import test_basic
  test_basic.speedReducerScipy()


def _case_gp_mads():
  import test_basic
  test_basic.geometric_programming()


def _case_gp_ai_regularizer():
  import GP
  GP.GP_auto_build_and_run(os.path.join(REPO, "tests", "GP", "GP.yaml"), os.getcwd())


CASES: Dict[str, Callable[[], None]] = {
  "basic_poll": _case_basic_poll,
  "sellar_scipy": _case_sellar_scipy,
  "sellar_poll": _case_sellar_poll,
  "sellar_mads": _case_sellar_mads,
  "speed_reducer_scipy": _case_speed_reducer_scipy,
  "gp_mads": _case_gp_mads,
  "gp_ai_regularizer": _case_gp_ai_regularizer,
}


def _close_file_handlers_under(directory: str) -> None:
  """Close logging file handlers writing below ``directory`` (Windows cannot delete open files)."""
  import logging
  root = os.path.normcase(os.path.abspath(directory))
  loggers = [logging.getLogger()] + [lg for lg in logging.Logger.manager.loggerDict.values()
                                     if isinstance(lg, logging.Logger)]
  for lg in loggers:
    for handler in list(lg.handlers):
      path = getattr(handler, "baseFilename", None)
      if path and os.path.normcase(os.path.abspath(path)).startswith(root):
        handler.close()
        lg.removeHandler(handler)


def run_case(name: str) -> Dict[str, Any]:
  """Run one case in a private scratch directory (the reference scripts write to ./tests/test_files)."""
  with tempfile.TemporaryDirectory(prefix=f"dmdo_golden_{name}_", ignore_cleanup_errors=True) as wd:
    os.makedirs(os.path.join(wd, "tests", "test_files"))
    with _in_dir(wd), capture_runs() as runs:
      try:
        CASES[name]()
      finally:
        _close_file_handlers_under(wd)
  if len(runs) != 1:
    raise RuntimeError(f"Case {name} produced {len(runs)} MDO runs, expected 1")
  return runs[0]


def trace_path(name: str) -> str:
  return os.path.join(TRACE_DIR, f"{name}.json")


def load_trace(name: str) -> Dict[str, Any]:
  with open(trace_path(name), "r") as fh:
    return json.load(fh)


def save_trace(name: str, snap: Dict[str, Any]) -> None:
  os.makedirs(TRACE_DIR, exist_ok=True)
  with open(trace_path(name), "w") as fh:
    json.dump(snap, fh, indent=1)


def compare(expected: Dict[str, Any], actual: Dict[str, Any], rtol: float = 1e-9, atol: float = 1e-12) -> List[str]:
  """Return a list of human-readable differences (empty when traces match)."""
  diffs: List[str] = []
  for key, exp in expected.items():
    act = actual.get(key)
    if key == "master_vars":
      if len(exp) != len(act):
        diffs.append(f"master_vars: length {len(exp)} != {len(act)}")
        continue
      for e, a in zip(exp, act):
        if (e["name"], e["sp_index"]) != (a["name"], a["sp_index"]) or not np.allclose(
            e["value"], a["value"], rtol=rtol, atol=atol, equal_nan=True):
          diffs.append(f"master_vars: {e} != {a}")
    elif isinstance(exp, list):
      if act is None or len(exp) != len(act):
        diffs.append(f"{key}: length {len(exp)} != {None if act is None else len(act)}")
      elif not np.allclose(exp, act, rtol=rtol, atol=atol, equal_nan=True):
        idx = int(np.argmax(~np.isclose(exp, act, rtol=rtol, atol=atol, equal_nan=True)))
        diffs.append(f"{key}: first mismatch at [{idx}] {exp[idx]!r} != {act[idx]!r}")
    elif isinstance(exp, float):
      if act is None or not np.isclose(exp, act, rtol=rtol, atol=atol, equal_nan=True):
        diffs.append(f"{key}: {exp!r} != {act!r}")
    elif exp != act:
      diffs.append(f"{key}: {exp!r} != {act!r}")
  return diffs
