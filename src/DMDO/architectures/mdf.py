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
"""Monolithic MDF (``architecture: {MON: MDF}``).

Every subproblem is a standalone MDO problem: its variables may not be linked to other subproblems. Its solver
(``solver``, ``scipy`` / ``configurations``, ``budget`` keys, unchanged) optimises the subproblem's design variables
directly. At every evaluation the subproblem's MDA converges its feedback loops (``MDA.<id>.solver``, default GS; see
:mod:`DMDO.analysis.mda_solvers`), then ``opt(x, y)`` gives the objective and the constraints ``g <= 0``. The bounds
of the MDA responses are added as constraints, as the distributed path does for its couplings. An evaluation whose
MDA does not converge is reported as failed (objective +inf).

There is no coordinator: subproblems are solved one after the other, independently.
"""
import copy
import csv
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..analysis.da import analyze_and_validate_types
from ..core.enums import COUPLING_TYPE
from ..ports import SolverRequest
from ..utils.log import MSG_TYPE

HISTORY_FILE = "MDF_history.csv"


class StandaloneCoordination:
  """Stand-in for the coordinator of a subproblem that is solved without coordination (MON: MDF)."""
  is_AI_regularizer = False
  save_q_in_out = False
  index = None

  def __repr__(self) -> str:
    return "StandaloneCoordination()"


@dataclass
class MDFResult:
  """Outcome of the MDF solve of one subproblem."""
  subproblem: int
  x: Dict[str, Any]
  y: Dict[str, Any]
  fmin: float
  hmin: float
  feasible: bool
  evaluations: int
  mda_converged: bool
  mda_iterations: int
  mda_evaluations: int
  history: List[Dict[str, Any]] = field(default_factory=list)

  def to_dict(self) -> Dict[str, Any]:
    out = dict(self.__dict__)
    out.pop("history")
    return out


def _flat(value: Any) -> List[float]:
  return list(np.ravel(np.asarray(value, dtype=float)))


class MDFSubproblem:
  """Monolithic MDF evaluation of one standalone subproblem."""

  def __init__(self, sp, log=None, post_dir: Optional[str] = None):
    self.sp = sp
    self.post_dir = post_dir
    self.mda = sp.MDA_process
    self.log = log
    self.design = [v for v in sp.get_design_vars() if v.coupling_type != COUPLING_TYPE.CONSTANT]
    # MDA responses are the state of the MDA; the bounds of every response but dummies become constraints.
    self.responses = list(self.mda.responses or [])
    self.bounded = [r for r in self.responses if r.coupling_type not in (COUPLING_TYPE.DUMMY, COUPLING_TYPE.CONSTANT)]
    self.state: Dict[int, Any] = {int(r.index): copy.deepcopy(r.value) for r in self.responses}
    self.evaluations = 0
    self.mda_iterations = 0
    self.mda_evaluations = 0
    self.last_mda = None
    self.history: List[Dict[str, Any]] = []
    self._con_memo: Dict[tuple, List[float]] = {}
    types = analyze_and_validate_types(sp.opt)
    keys = list(types)
    self._x_as_dict = len(keys) > 0 and types[keys[0]] == "dict"
    self._y_as_dict = len(keys) > 1 and types[keys[1]] == "dict"

  # ------------------------------------------------------------------------------------------- evaluation
  def _analyse(self, x: List[Any], clist: Optional[List[Any]] = None):
    """Set the design variables, converge the MDA and evaluate ``opt``. Returns ``(f, g)``."""
    sp = self.sp
    sp.set_variables_value(list(x), clist)
    state = dict(self.state)                      # warm start from the last converged couplings
    for v in sp.vars:
      state[int(v.index)] = copy.deepcopy(v.value)
    res = self.mda.run_coupled(state)
    self.last_mda = res
    self.mda_iterations += res.iterations
    self.mda_evaluations += res.evaluations
    self.evaluations += 1
    for r in self.responses:
      r.value = copy.deepcopy(state[int(r.index)])
    if not res.converged:
      return np.inf, [np.inf]
    self.state = {int(r.index): copy.deepcopy(r.value) for r in self.responses}
    inp1 = sp.get_dict_vars(sp.get_design_vars()) if self._x_as_dict else list(x)
    inp2 = sp.get_dict_outputs(self.responses) if self._y_as_dict else [r.value for r in self.responses]
    fun = sp.opt(inp1, inp2, clist)
    f = float(fun[0])
    g = [float(c) for c in np.ravel(np.asarray(fun[1], dtype=float))] if len(fun) > 1 and fun[1] is not None else []
    for r in self.bounded:
      vals, lbs, ubs = _flat(r.value), _flat(r.lb), _flat(r.ub)
      g += [vi - ui for vi, ui in zip(vals, ubs)] + [li - vi for vi, li in zip(vals, lbs)]
    return f, g

  def _record(self, x, f, g) -> None:
    h = float(np.sum(np.maximum(np.asarray(g, dtype=float), 0.0) ** 2)) if g else 0.0
    row = {"evaluation": self.evaluations, "objective": f, "h": h,
           "mda_converged": bool(self.last_mda.converged), "mda_iterations": self.last_mda.iterations,
           "mda_residual": self.last_mda.residual}
    for v, xi in zip(self.sp.get_list_vars_names(self.design), x):
      row[f"x:{v}"] = xi
    for r in self.responses:
      row[f"y:{r.name}"] = copy.deepcopy(r.value)
    self.history.append(row)

  def evaluate(self, x, clist=None, *argv):
    f, g = self._analyse(x, clist)
    self._record(x, f, g)
    self._con_memo[tuple(np.ravel(np.asarray(x, dtype=float)))] = list(g)
    impl = self.sp._solver_impl()
    if impl.objective_form != "scalar":
      return [f, g if g else [0.0]]
    settings = self.sp.scipy or {}
    if settings.get("is_con", False) or not g:
      return f
    # Unconstrained scalar solver: same exterior penalty as SubProblem.evaluate.
    return f + float(np.sum(np.maximum(np.asarray(g), 0.0) ** 2)) * 1e6

  def constraints(self, x) -> List[float]:
    key = tuple(np.ravel(np.asarray(x, dtype=float)))
    if key not in self._con_memo:
      self.evaluate(x)
    return list(self._con_memo[key]) or [0.0]

  # ------------------------------------------------------------------------------------------- solve
  def solve(self) -> MDFResult:
    sp = self.sp
    impl = sp._solver_impl()
    x0 = sp._solve_start_point()
    lb, ub = sp.get_list_vars_lb(self.design), sp.get_list_vars_ub(self.design)
    x0 = [min(max(x, lo), hi) if all(isinstance(t, (int, float)) for t in (x, lo, hi)) else x
          for x, lo, hi in zip(x0, lb, ub)]
    request = SolverRequest(
      name=f"SP_{sp.index}", objective=self.evaluate, constraints=self.constraints, x0=x0, lb=lb, ub=ub,
      var_names=sp.get_list_vars_names(self.design), var_types=sp.get_vars_types(self.design), var_sets=sp.sets,
      constants=sp.get_list_constant_updates(self.design), constants_names=sp.get_list_const_names(self.design),
      budget=int(sp.budget), budget_scale=1.0, seed=int(sp.RANDOM_SEED), iteration=0, tol=sp.tol,
      display=sp.display, post_dir=getattr(sp, "postDir", None) or self.post_dir, config=impl.select_config(sp.conf, sp.scipy),
      warm_start={"psize": sp.psize, "psize_policy": sp.psize_init})
    if self.log is not None:
      self.log.log_msg(msg=f"MON: MDF: solving subproblem {sp.index} with {sp.solver}", msg_type=MSG_TYPE.INFO.value)
    outcome = impl.solve(request)
    x_best = list(outcome.x) if len(outcome.x) > 0 else x0
    f, g = self._analyse(x_best, sp.get_list_constant_updates(self.design))
    self._record(x_best, f, g)
    h = float(np.sum(np.maximum(np.asarray(g, dtype=float), 0.0) ** 2)) if g else 0.0
    sp.fmin, sp.hmin, sp.fmin_nop = f, h, f
    return MDFResult(
      subproblem=int(sp.index),
      x={n: xi for n, xi in zip(sp.get_list_vars_names(self.design), x_best)},
      y={r.name: copy.deepcopy(r.value) for r in self.responses},
      fmin=f, hmin=h, feasible=bool(np.isfinite(f) and (not g or max(g) <= 1e-6)),
      evaluations=self.evaluations, mda_converged=bool(self.last_mda.converged),
      mda_iterations=self.mda_iterations, mda_evaluations=self.mda_evaluations, history=self.history)


def check_standalone(subproblems) -> None:
  """MDF subproblems are standalone: none of their variables may link to another subproblem of the run."""
  indices = {int(sp.index) for sp in subproblems}
  for sp in subproblems:
    for v in list(sp.vars) + list(sp.MDA_process.responses or []):
      links = v.link if isinstance(v.link, (list, tuple)) else [v.link]
      others = sorted(int(k) for k in links if k is not None and str(k) != "None" and int(k) in indices
                      and int(k) != int(sp.index))
      if others:
        raise ValueError(f"MON: MDF: variable '{v.name}' (index {v.index}) of subproblem {sp.index} is linked to "
                         f"subproblem(s) {others}. MDF subproblems are standalone problems; put every analysis of "
                         f"the problem in one subproblem's MDA (set link to null).")


def write_history(path: str, history: List[Dict[str, Any]]) -> None:
  if not history:
    return
  keys: List[str] = []
  for row in history:
    keys += [k for k in row if k not in keys]
  with open(path, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=keys)
    writer.writeheader()
    writer.writerows(history)


def run_mdf(mdo) -> Dict[str, Any]:
  """Solve every subproblem of ``mdo`` with monolithic MDF. Returns ``{"architecture", "subproblems"}``."""
  log = getattr(mdo, "log", None)
  check_standalone(mdo.subProblems)
  results = []
  for sp in mdo.subProblems:
    if log is not None and not hasattr(sp, "log"):
      sp.log = log
    post = getattr(mdo, "post_dir", None)
    if post:
      os.makedirs(post, exist_ok=True)
    driver = MDFSubproblem(sp, log=log, post_dir=post or os.getcwd())
    res = driver.solve()
    results.append(res)
    if log is not None:
      log.log_msg(msg=f"MON: MDF: subproblem {sp.index}: f = {res.fmin}, h = {res.hmin}, x = {res.x}, y = {res.y}, "
                      f"{res.evaluations} evaluations, {res.mda_iterations} MDA iterations",
                  msg_type=MSG_TYPE.INFO.value)
    if post:
      name = HISTORY_FILE if len(mdo.subProblems) == 1 else f"MDF_SP{sp.index}_history.csv"
      write_history(os.path.join(post, name), res.history)
  # The run's variables hold the solution.
  by_index = {int(v.index): v for sp in mdo.subProblems for v in list(sp.vars) + list(sp.MDA_process.responses or [])}
  for v in mdo.variables or []:
    src = by_index.get(int(v.index))
    if src is not None and src is not v:
      v.value = copy.deepcopy(src.value)
  mdo.mdf_results = results
  if mdo.display:
    for r in results:
      print(f"MON: MDF subproblem {r.subproblem}: f = {r.fmin:.8g}, h = {r.hmin:.3g}, x = {r.x}, y = {r.y} "
            f"({r.evaluations} evaluations, {r.mda_iterations} MDA iterations)")
  return {"architecture": mdo.architecture.label, "subproblems": [r.to_dict() for r in results]}
