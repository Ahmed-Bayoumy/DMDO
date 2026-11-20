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
"""Executors and coordination sweeps: task messages, executor resolution, Jacobi order independence."""
import contextlib
import os
import sys
import tempfile

import pytest

from DMDO import (MDO, Executor, InProcessExecutor, ProcessExecutor, SubproblemResult, SubproblemTask,
                  create_executor, executors, solve_task)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden"))
from _harness import _close_file_handlers_under, _in_dir, compare, snapshot  # noqa: E402


def _square(x):
  return x * x


# --------------------------------------------------------------------------------------------- executors
def test_inprocess_executor_preserves_order():
  assert InProcessExecutor().map(_square, [3, 1, 2]) == [9, 1, 4]


def test_process_executor_preserves_order_and_reuses_pool():
  with ProcessExecutor(max_workers=2) as ex:
    assert ex.map(_square, list(range(6))) == [0, 1, 4, 9, 16, 25]
    pool = ex._pool
    assert ex.map(_square, [7]) == [49]
    assert ex._pool is pool
  assert ex._pool is None


def test_create_executor_from_names_classes_and_instances():
  assert isinstance(create_executor("serial"), InProcessExecutor)
  assert isinstance(create_executor("Parallel", max_workers=3), ProcessExecutor)
  assert create_executor("process", max_workers=3).max_workers == 3
  assert isinstance(create_executor(InProcessExecutor), InProcessExecutor)
  inst = InProcessExecutor()
  assert create_executor(inst) is inst
  assert isinstance(inst, Executor)


def test_custom_executor_plugs_in_by_name():
  class Recording(InProcessExecutor):
    calls = []

    def map(self, fn, tasks):
      type(self).calls.append(len(tasks))
      return super().map(fn, tasks)

  executors.register("unit_test_executor", Recording, override=True)
  try:
    ex = create_executor("unit_test_executor")
    ex.map(_square, [1, 2])
    assert Recording.calls == [2]
  finally:
    executors.unregister("unit_test_executor")


def test_solve_task_wraps_subproblem_solve():
  class FakeSP:
    def solve(self, v, w, file=None, iter=None):
      return {"v": v, "w": w, "file": file, "iter": iter}

  result = solve_task(SubproblemTask(index=4, subproblem=FakeSP(), v=[1.], w=[2.], iteration=7, file="f"))
  assert isinstance(result, SubproblemResult) and result.index == 4
  assert result.output == {"v": [1.], "w": [2.], "file": "f", "iter": 7}


def test_unknown_sweep_is_rejected():
  with pytest.raises(ValueError, match="sweep"):
    MDO._resolve_sweep(object.__new__(MDO), "serial", "zigzag")


# --------------------------------------------------------------------------------------------- Jacobi sweep
class ReversedExecutor(InProcessExecutor):
  """Solves tasks in reverse order (as if the last subproblem finished first); returns them in task order."""

  def map(self, fn, tasks):
    return list(reversed([fn(t) for t in reversed(tasks)]))


class ShuffledResultsExecutor(InProcessExecutor):
  """Returns results out of order; the sweep must sort them by task index."""

  def map(self, fn, tasks):
    return list(reversed([fn(t) for t in tasks]))


def _run_basic(budget=6, **run_kwargs):
  import test_basic

  captured = []
  original = MDO.run

  def wrapped(self, *args, **kwargs):
    self.Coordinator.budget = budget
    kwargs.update(run_kwargs)
    out = original(self, *args, **kwargs)
    captured.append(snapshot(self))
    return out

  with tempfile.TemporaryDirectory(prefix="dmdo_exec_", ignore_cleanup_errors=True) as wd:
    os.makedirs(os.path.join(wd, "tests", "test_files"))
    MDO.run = wrapped
    try:
      with _in_dir(wd), open(os.devnull, "w", encoding="utf-8") as devnull, contextlib.redirect_stdout(devnull):
        test_basic.test_basic_MDO()
    finally:
      MDO.run = original
      _close_file_handlers_under(wd)
  assert len(captured) == 1
  return captured[0]


@pytest.fixture(scope="module")
def jacobi_reference():
  return _run_basic(sweep="jacobi", executor=InProcessExecutor())


@pytest.mark.parametrize("executor", [ReversedExecutor(), ShuffledResultsExecutor()], ids=["reversed", "shuffled"])
def test_jacobi_sweep_is_independent_of_execution_order(jacobi_reference, executor):
  assert compare(jacobi_reference, _run_basic(sweep="jacobi", executor=executor), rtol=0, atol=0) == []


def test_jacobi_sweep_in_worker_processes_matches_in_process(jacobi_reference):
  assert compare(jacobi_reference, _run_basic(mode="parallel"), rtol=0, atol=0) == []


def test_gauss_seidel_differs_from_jacobi(jacobi_reference):
  # Sanity check that the sweep option is honoured (the two schemes follow different paths).
  assert compare(jacobi_reference, _run_basic(sweep="gauss_seidel")) != []
