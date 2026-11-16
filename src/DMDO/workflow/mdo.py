"""
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
# """


import copy
import csv
from dataclasses import dataclass
import os
import pickle
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..subproblem.subproblem import SubProblem
from ..utils.log import MSG_TYPE, logger
from ..ai.detector import SolveQualityExtractor, latest_detector_metrics
from ..core.protocols import Process_data
from ..coordination.admm import ADMM
from ..core.variables import variableData
from .execution import SubproblemTask, create_executor, solve_task

@dataclass
class MDO_data(Process_data):
  Architecture: int
  Coordinator: ADMM
  subProblems: List[SubProblem]
  fmin: float
  hmin: float
  display: bool
  inc_stop: float
  stop: str
  tab_inc: List
  noprogress_stop: int
  eps_qio: List[float] = None
  eps_fio: List[float] = None
  log: logger = None
  iter: int = 0
  file: str = None
  initialized: bool = False
  working_dir: str = "."
  post_dir: str = "./_post"
  mdo_name: str = "undefined"


@dataclass
class MDO(MDO_data):
  """_summary_

  :param MDO_data: _description_
  :type MDO_data: _type_
  """
  def __getstate__(self):
    # Remove or replace non-picklable attributes
    state = self.__dict__.copy()
    # Example: remove or replace RLock
    if '_lock' in state:
        del state['_lock']
    # Or replace with a placeholder
    # state['_lock'] = None
    return state
  
  def is_pickleable(self, obj):
    try:
        pickle.dumps(obj)
        return True
    except Exception as e:
        print(f"Object is not pickleable: {e}")
        return False

  def __setstate__(self, state):
      self.__dict__.update(state)
      # Reconstruct non-picklable objects if needed
      # self._lock = threading.RLock()
  def get_list_of_var_values(self, x: List[variableData]):
    x_temp = []
    for i in range(len(x)):
      x_temp.append(x[i].value)
    return x_temp

  def get_master_vars_difference(self):
    dx = []
    for i in range(len(self.Coordinator.master_vars)):
      mv_clone = copy.deepcopy(self.Coordinator.master_vars[i])
      mvold_clone = copy.deepcopy(self.Coordinator.master_vars_old[i])
      mv_type = mv_clone.type[0][0].lower() if isinstance(mv_clone.type, list) else mv_clone.type[0].lower()

      if mv_clone.dim > 1:
        for k in range(mv_clone.dim):
          if mv_type == 'c' or mv_type == 'd':
            mv_clone.value[k] = self.sets[mv_clone.set].index(self.Coordinator.master_vars[i].value[k])
        for k in range(mvold_clone.dim):
          if mv_type == 'c' or mv_type == 'd':
            mvold_clone.value[k] = self.sets[mvold_clone.set].index(self.Coordinator.master_vars_old[i].value[k])
      else:
        if mv_type == 'c' or mv_type == 'd':
            mv_clone.value = self.sets[mv_clone.set].index(self.Coordinator.master_vars[i].value)
            mvold_clone.value = self.sets[mvold_clone.set].index(self.Coordinator.master_vars_old[i].value)
      # if mv_type =='c':
      #   if mv_cloneuntil.value == mvold_clone.value:
      #     dx.append(0)
      #   else:
      #     dx.append(1)
      # else:
      dx.append(mv_clone - mvold_clone)
      # x.append(self.Coordinator.master_vars[i].value)
      # xold.append(self.Coordinator.master_vars_old[i].value)
    DX: list = []
    for i in range(len(dx)):
      if isinstance(dx[i], list) or isinstance(dx[i], np.ndarray):
        for j in range(len(dx[i])):
          DX.append(dx[i][j])
      else:
        DX.append(dx[i])
    return np.linalg.norm(DX, 2)

  def check_termination_critt(self, iter):
    if (
      iter > 1
      and np.abs(self.tab_inc[iter]) < self.inc_stop
      and np.max(np.abs(np.subtract(np.abs(self.Coordinator.q), np.abs(self.Coordinator.qold)))) < self.inc_stop
    ):
      msg = f'Stop: qmax = {np.max(np.abs(self.Coordinator.q))} < {self.inc_stop}'
      self.log.log_msg(msg=msg, msg_type=MSG_TYPE.INFO.value)
      print(msg)
      msg = "Max inconsitency is below stopping threshold"
      self.log.log_msg(msg=msg, msg_type=MSG_TYPE.INFO.value)
      self.stop = msg
      return True

    i = self.noprogress_stop

    if iter > i + 2 and np.log(np.min(self.Coordinator.w) > 6.) \
      and np.less_equal(self.tab_inc[iter-i], np.min(self.tab_inc[iter-i+1:iter])):
      msg = f'Stop: no progress after {i} iterations.'
      self.log.log_msg(msg=msg, msg_type=MSG_TYPE.INFO.value)
      print(msg)
      msg = "No progress after several iterations"
      self.log.log_msg(msg=msg, msg_type=MSG_TYPE.INFO.value)
      print(msg)
      self.stop = msg
      return True

    return False

  def get_weights_size(self, sp_i: int) -> int:
    link1 = self.subProblems[sp_i].coord._linker[0]
    out = 0
    for i in range(len(link1)):
      out += self.subProblems[sp_i].coord.master_vars[link1[i]-1].dim
    
    return out

  def _get_sp_indices(self) -> List[int]:
    return [sp.index for sp in self.subProblems]

  def _get_hmin_columns(self) -> List[str]:
    return [f'{f"hmin_{sp_index}".rjust(30)}' for sp_index in self._get_sp_indices()]

  def _get_ol_keys(self) -> List[str]:
    hmin_cols = self._get_hmin_columns()
    if self.Coordinator.is_AI_regularizer:
      adaptive_cols = (
        [f'{"Eff_spike_w_scale".rjust(30)}',
         f'{"Eff_spike_gamma_scale".rjust(30)}',
         f'{"Eff_budget_scale".rjust(30)}',
         f'{"Adapt_event".rjust(30)}']
        if getattr(self.Coordinator, "spike_adaptive_scales", True)
        else []
      )
      return [f'{"Time"}', f'{"Iteration #".rjust(30)}', \
              f'{"Max. inconsistency".rjust(30)}', \
              # f'{"Max_inc_predicted".rjust(30)}', \
              f'{"spike_location".rjust(30)}', \
              f'{"alarm_detected".rjust(30)}', \
              f'{"spike_detected".rjust(30)}', \
              f'{"Objective".rjust(30)}', \
              # f'{"Obj_predicted".rjust(30)}', \
              f'{"Spike_prob".rjust(30)}', \
              f'{"Forecast_prob".rjust(30)}', \
              f'{"Non_improvement_prob".rjust(30)}', \
              f'{"Obj_non_improvement_prob".rjust(30)}', \
              f'{"Constraint_non_improvement_prob".rjust(30)}', \
              f'{"Detector_uncertainty".rjust(30)}', \
              f'{"Constraint_pressure".rjust(30)}', \
              f'{"Status".rjust(30)}', \
              f'{"Variables change".rjust(30)}', \
              f'{"max(w)".rjust(30)}', \
              f'{"Coupling_with_qmax".rjust(30)}'] + \
                adaptive_cols + \
                hmin_cols + \
                [f'{f"{x.name}_{x.sp_index}".rjust(30)}' for x in self.Coordinator.master_vars]

    return [f'{"Time"}', f'{"Iteration #".rjust(30)}', \
            f'{"Max. inconsistency".rjust(30)}', \
          f'{"alarm_detected".rjust(30)}', \
          f'{"spike_detected".rjust(30)}', \
            f'{"Objective".rjust(30)}', \
            f'{"Status".rjust(30)}', \
            f'{"Variables change".rjust(30)}', \
            f'{"max(w)".rjust(30)}', \
            f'{"Coupling_with_qmax".rjust(30)}'] + \
              hmin_cols + \
              [f'{f"{x.name}_{x.sp_index}".rjust(30)}' for x in self.Coordinator.master_vars]

  def _reset_ai_regularizer_outputs(self):
    # A premature-freeze event (causal response) also triggers the next iteration's budget reallocation.
    last_detected = bool(
      getattr(self.Coordinator, "spike_detected", False) or getattr(self.Coordinator, "freeze_detected", False)
    )
    self.Coordinator.last_spike_detected = last_detected
    self.Coordinator.freeze_detected = False
    # A stale location without a detection would trigger budget expansion every iteration.
    self.Coordinator.last_spike_location = int(getattr(self.Coordinator, "spike_location", 0)) if last_detected else 0
    self.Coordinator.qmax_predicted = 0.
    self.Coordinator.spike_location = 0
    self.Coordinator.alarm_detected = False
    self.Coordinator.spike_detected = False
    self.Coordinator.ai_event_strength = -1.
    self.Coordinator.fmin_predicted = 0.
    self.Coordinator.spike_prob = 0.
    self.Coordinator.forecast_prob = 0.
    self.Coordinator.non_improvement_prob = 0.
    self.Coordinator.objective_non_improvement_prob = 0.
    self.Coordinator.constraint_non_improvement_prob = 0.
    self.Coordinator.detector_uncertainty = 1.
    self.Coordinator.constraint_pressure = 0.

  def _get_detector(self):
    detector = getattr(self.Coordinator, "ai_regularizer", None)
    if detector is None:
      for sp in self.subProblems:
        detector = getattr(sp.coord, "ai_regularizer", None) or getattr(sp, "_AI_regularizer", None)
        if detector is not None:
          break
      if detector is not None:
        self.Coordinator.ai_regularizer = detector
    return detector

  @staticmethod
  def _scalarize_hmin(value) -> float:
    array = np.atleast_1d(np.asarray(value, dtype=float))
    finite = array[np.isfinite(array)]
    if finite.size > 0:
      return float(max(np.max(finite), 0.0))
    return float(np.inf) if np.any(array > 0.0) else 0.0

  def _attribute_spike_subproblem(self) -> int:
    """Subproblem blamed for the largest inconsistency.

    Of the two ends of the worst link, the one with the more erratic inner solve.
    """
    coord = self.Coordinator
    master = int(coord.index_of_master_SP)
    try:
      per_link = [
        float(np.max(np.abs(q))) if isinstance(q, (list, np.ndarray)) else abs(float(q))
        for q in coord.batch_q(coord.q)
      ]
      worst = int(np.argmax(per_link))
      ends = [int(coord.master_vars[coord._linker[row, worst] - 1].sp_index) for row in (0, 1)]
    except Exception:
      return master
    by_index = {sp.index: sp for sp in self.subProblems}
    best, best_range = ends[0], -1.0
    for idx in dict.fromkeys(ends):
      sp = by_index.get(idx)
      fingerprint = SolveQualityExtractor.extract(getattr(sp, "Il_file", None)) if sp is not None else None
      inc_range = float(fingerprint[3]) if fingerprint is not None else 0.0
      if inc_range > best_range:
        best, best_range = idx, inc_range
    return int(best)

  def _rollback_step(self, iter: int) -> None:
    """Record this iteration's design; on a detected spike with a clearly worse objective, restore a better one.

    Candidates are earlier feasible iterations whose q_max is within rollback_q_factor of the current one (or below
    100x the tolerance), so the restart point is nearly as consistent as the run has become. The penalty weights
    are scaled by rollback_w_scale (floor 1) and the multipliers v are restored with the design. At most rollback_max
    rollbacks per run, at least rollback_gap iterations apart.
    """
    coord = self.Coordinator
    if not (coord.is_AI_regularizer and getattr(coord, "rollback", False)):
      return
    q_now = float(np.max(np.abs(coord.q))) if np.size(coord.q) else 0.0
    obj_now = float(self.fmin[0] if isinstance(self.fmin, (list, tuple, np.ndarray)) else self.fmin)
    hmax = max([self._scalarize_hmin(v) for v in self.hmin_dict.values()] or [0.0])
    history = getattr(self, "_rb_history", [])
    history.append({
      "iter": int(iter), "obj": obj_now, "q": q_now, "hmax": hmax,
      "sp_vars": [[copy.deepcopy(v.value) for v in sp.vars] for sp in self.subProblems],
      "master_vars": copy.deepcopy(coord.master_vars),
      "v": copy.deepcopy(coord.v),
    })
    self._rb_history = history

    n_done = int(getattr(self, "_rb_count", 0))
    last = int(getattr(self, "_rb_last", -10**9))
    if str(getattr(coord, "rollback_trigger", "event")) == "forecast":
      detected = bool(getattr(coord, "_forecast_accepted", False))
    else:
      detected = getattr(coord, "_event_kind", "") in ("spike", "hold")
    # Optional safeguard: a design that has become infeasible is rolled back even without a detector alarm.
    lost_feasibility = bool(getattr(coord, "rollback_on_infeasible", False)) and hmax > 1e-6
    if not (detected or lost_feasibility) or n_done >= int(coord.rollback_max) or iter - last < int(coord.rollback_gap):
      return
    q_cap = max(float(coord.rollback_q_factor) * q_now, 100.0 * float(self.inc_stop))
    candidates = [h for h in history[:-1] if np.isfinite(h["obj"]) and h["hmax"] <= 1e-6 and h["q"] <= q_cap]
    if not candidates:
      return
    best = min(candidates, key=lambda h: h["obj"])
    if not lost_feasibility and not obj_now > best["obj"] + float(coord.rollback_obj_tol) * max(1.0, abs(best["obj"])):
      return
    for sp, values in zip(self.subProblems, best["sp_vars"]):
      for var, value in zip(sp.vars, values):
        var.value = copy.deepcopy(value)
        var.baseline = copy.deepcopy(value)
    coord.master_vars = copy.deepcopy(best["master_vars"])
    # The multipliers belong to the design they were accumulated on; keeping the current ones would push the
    # restored design hard in the direction of the abandoned one.
    coord.v = copy.deepcopy(best["v"])
    coord.w = np.maximum(np.multiply(coord.w, float(coord.rollback_w_scale)), float(getattr(coord, "w_init", 1.0)))
    self._rb_count, self._rb_last = n_done + 1, int(iter)
    why = "feasibility lost" if lost_feasibility and not detected else "detected spike"
    msg = (f"Rollback at iteration {iter} ({why}): objective {obj_now:.6g}, hmax {hmax:.3g} -> design of iteration "
           f"{best['iter']} ({best['obj']:.6g})")
    self.log.log_msg(msg=msg, msg_type=MSG_TYPE.INFO.value)
    print(msg)

  @staticmethod
  def _late_improving(sp_file, tail: float = 0.3, rtol: float = 1e-6) -> bool:
    """True when the solve's best merit (objective + penalty) still improved in its last `tail` fraction of evaluations."""
    if not sp_file or not os.path.isfile(sp_file):
      return False
    try:
      df = pd.read_csv(sp_file, skipinitialspace=True)
      df.columns = [str(c).strip() for c in df.columns]
      merit = (pd.to_numeric(df["Objective"], errors="coerce") + pd.to_numeric(df["Penalty"], errors="coerce")).values
    except Exception:
      return False
    merit = merit[np.isfinite(merit)]
    if merit.size < 4:
      return False
    cut = int(np.floor((1.0 - tail) * merit.size))
    best_before, best_after = float(np.min(merit[:cut])), float(np.min(merit[cut:]))
    return best_after < best_before - rtol * max(1.0, abs(best_before))

  def _update_detector_outer_iteration(self, iter: int) -> None:
    """Feed the detector once per outer iteration (paper Algorithm 1) and publish its outputs."""
    coord = self.Coordinator
    detector = self._get_detector()
    if detector is None or not getattr(coord, "detector_per_outer_iteration", True):
      return

    q = np.abs(np.asarray(coord.q, dtype=float))
    inc = float(np.max(q)) if q.size > 0 else 0.0
    obj = float(self.fmin[0] if isinstance(self.fmin, (list, tuple, np.ndarray)) else self.fmin)
    hmins = [self._scalarize_hmin(self.hmin_dict.get(sp.index, [0.])) for sp in self.subProblems]

    owner = self._attribute_spike_subproblem()
    owner_sp = next((sp for sp in self.subProblems if sp.index == owner), None)
    detector.update_sp_features(getattr(owner_sp, "Il_file", None))
    alarm = detector.update(obj=obj, inc=inc, hmins=hmins)

    metrics = latest_detector_metrics(detector)
    coord.qmax_predicted = metrics["forecast_prob"]
    coord.fmin_predicted = metrics["objective_non_improvement_prob"]
    coord.spike_prob = metrics["spike_prob"]
    coord.forecast_prob = metrics["forecast_prob"]
    coord.non_improvement_prob = metrics["non_improvement_prob"]
    coord.objective_non_improvement_prob = metrics["objective_non_improvement_prob"]
    coord.constraint_non_improvement_prob = metrics["constraint_non_improvement_prob"]
    coord.detector_uncertainty = metrics["detector_uncertainty"]
    coord.constraint_pressure = metrics["constraint_pressure"]
    coord.ai_event_strength = float(max(metrics.values()))
    coord.spike_location = owner
    coord.alarm_detected = bool(alarm)
    if str(getattr(coord, "spike_response", "accelerate")) == "causal":
      coord._sp_late_improving = {sp.index: self._late_improving(getattr(sp, "Il_file", None)) for sp in self.subProblems}
  
  def initialize_OL_res_file(self, file):
    if not hasattr(self.Coordinator, "master_vars"):
      return
    if not os.path.exists(file):
      mode = 'x'
    else:
      mode = 'w'
    with open(file, mode=mode) as csv_file:
      keys = self._get_ol_keys()
      writer = csv.DictWriter(csv_file, fieldnames=keys)
      writer.writeheader()    # add column names in the CSV file
  
  def prepare_post(self):
    pd = self.post_dir
    pfo = os.path.join(pd, 'Coordination_history.out')
    os.makedirs(pd, exist_ok=True)
    self.postDir = pd
    self.Ol_file = pfo
    # self.log.relocate_logger(source_file=self.log.log.handlers[0].baseFilename, Dest_file=pd)
    self.initialize_OL_res_file(pfo)
    
  def Add_OL_res_row(self, r: Dict):
    with open(self.Ol_file, mode='a') as csv_file:
      keys = self._get_ol_keys()
      writer = csv.DictWriter(csv_file, fieldnames=keys)
      writer.writerow(r)    # add column names in the CSV file

  def propose_best_candidates(self, s):
    temp_obj = []
    sp_vnames = [vs.name for vs in self.subProblems[s].vars]
    for nsp in range(len(self.subProblems)):
      ctemp = copy.deepcopy(self.subProblems[nsp].cache)
      if isinstance(ctemp, pd.DataFrame):
        cache_rows = ctemp.to_dict("records")
      else:
        cache_rows = [
          {"objective": ct[0], "penalty": ct[1], "vlist": ct[2], "outputs": ct[3]}
          for ct in ctemp
        ]
      for ct in cache_rows:
        if len(temp_obj) == 0 or ct["objective"] < min(temp_obj):
          temp_obj.append(ct["objective"])
          if s == nsp:
            copy.deepcopy(ct["vlist"])
          else:
            svIndex = -1
            for v in self.subProblems[nsp].vars:
              svIndex += 1
              try:
                vindex = sp_vnames.index(v.name)
                if v.link == self.subProblems[s].index:
                  self.subProblems[s].vars[vindex].baseline = ct["vlist"][svIndex]
              except:  # noqa: E722
                continue
            srIndex = -1
            for v in self.subProblems[nsp].resps:
              srIndex += 1
              try:
                vindex = sp_vnames.index(v.name)
                if v.link == self.subProblems[s].index:
                  self.subProblems[s].vars[vindex].baseline = ct["outputs"][srIndex]
              except:  # noqa: E722
                continue
    
    return

  def _record_subproblem_solve(self, s: int) -> None:
    """Record the constraint violation (and, for the master subproblem, the objective) of a finished solve."""
    sp = self.subProblems[s]
    hmin = sp.reported_hmin()
    self.hmin_dict[sp.index] = hmin
    if sp.index == self.Coordinator.index_of_master_SP:
      self.fmin = sp.fmin_nop
      self.hmin = hmin

  def solve_subproblem(self, s: int):
    self.prepare_SP_jobs(s=s)
    # if iter > 1:
    #   self.propose_best_candidates(s)
    out_sp = self.subProblems[s].solve(self.Coordinator.v, self.Coordinator.w, file=self.file, iter=self.iter)
    
    self.Coordinator = copy.deepcopy(self.subProblems[s].coord)
    
    return out_sp, self.subProblems[s]

  def prepare_SP_jobs(self, s: int):
    """Bring subproblem ``s`` up to date with the coordinator before a (Gauss-Seidel) solve."""
    if self.iter == 0:
      self._initialize_subproblem(s)
    else:
      self.subProblems[s].coord.master_vars = copy.deepcopy(self.Coordinator.master_vars)
    self.subProblems[s].modify_cond_vars(self.Coordinator.master_vars)

  def format_log_message(self, iter, qmax, obj, dx, wmax):
    """Format log message with dynamic spacing for vertical alignment"""
    # Use consistent width for each field
    # Field widths are set based on typical value ranges
    iter_width = 4  # 4 digits for iteration number
    qmax_width = 16  # Enough for scientific notation
    obj_width = 16  # Enough for large objective values
    dx_width = 16  # Enough for large dx values
    wmax_width = 16  # Enough for large wmax values

    # Format each field with consistent width
    iter_str = f'{iter}'.rjust(iter_width)
    qmax_str = f'qmax: {qmax:.8f}'.rjust(qmax_width)
    obj_str = f'Obj: {obj:.8f}'.rjust(obj_width)
    dx_str = f'dx: {dx:.8f}'.rjust(dx_width)
    wmax_str = f'max(w): {wmax:.8f}'.rjust(wmax_width)

    # Combine with || separators
    return f'{iter_str} || {qmax_str} || {obj_str} || {dx_str} || {wmax_str}'


 

  # ------------------------------------------------------------------------------------------------ coordination loop
  @staticmethod
  def _is_serial(mode) -> bool:
    return str(mode).lower() == "serial"

  def _resolve_sweep(self, mode, sweep):
    name = str(sweep if sweep is not None else ("gauss_seidel" if self._is_serial(mode) else "jacobi")).lower()
    if name in ("gauss_seidel", "gauss-seidel", "gs", "sequential"):
      return self._sweep_gauss_seidel
    if name in ("jacobi", "parallel"):
      return self._sweep_jacobi
    raise ValueError(f"Unknown coordination sweep '{sweep}'. Use 'gauss_seidel' or 'jacobi'.")

  def _sweep_gauss_seidel(self, executor) -> None:
    """Solve the subproblems one after another; each one sees the coordinator updated by the previous solves."""
    for s in range(len(self.subProblems)):
      if self.iter > 0:
        self.subProblems[s].coord = self.Coordinator
      self.prepare_SP_jobs(s=s)
      task = SubproblemTask(index=s, subproblem=self.subProblems[s], v=self.Coordinator.v, w=self.Coordinator.w,
                            iteration=self.iter, file=self.file)
      result = executor.map(solve_task, [task])[0]
      self.subProblems[s] = copy.deepcopy(result.subproblem)
      self._record_subproblem_solve(s)
      self.Coordinator = copy.deepcopy(self.subProblems[s].coord)

  def _sweep_jacobi(self, executor) -> None:
    """Solve all subproblems from the same coordinator snapshot, then merge their results in subproblem order.

    The outcome does not depend on how (or in which order) the executor runs the tasks.
    """
    n = len(self.subProblems)
    if self.iter == 0:
      for s in range(n):
        self._initialize_subproblem(s)
    # Consensus targets: every linked group of copies starts the sweep from its mean.
    self.Coordinator.calculate_and_set_average_target_values(None, None)
    tasks = []
    for s in range(n):
      sp = self.subProblems[s]
      sp.coord = copy.deepcopy(self.Coordinator)
      sp.modify_cond_vars(self.Coordinator.master_vars)
      tasks.append(SubproblemTask(index=s, subproblem=sp, v=copy.deepcopy(self.Coordinator.v),
                                  w=copy.deepcopy(self.Coordinator.w), iteration=self.iter, file=self.file))
    results = sorted(executor.map(solve_task, tasks), key=lambda r: r.index)

    merged = copy.deepcopy(self.Coordinator)
    for result in results:
      s, sp = result.index, result.subproblem
      self.subProblems[s] = sp
      self._record_subproblem_solve(s)
      # Each subproblem owns the master-vector entries of its own variable and response copies.
      owned = [v.index for v in sp.vars] + [r.index for r in (sp.MDA_process.responses or [])]
      for idx in owned:
        merged.master_vars[idx - 1] = copy.deepcopy(sp.coord.master_vars[idx - 1])
      if sp.is_main:
        merged.eps_fo = list(sp.coord.eps_fo)
    wb, vb = merged.batch_multipliers()
    merged.calc_inconsistency()
    if np.size(merged.q) != np.size(merged.w):
      merged.modify_multipliers(merged.batch_q(merged.q), wb, vb)
    self.Coordinator = merged

  def _initialize_subproblem(self, s: int) -> None:
    """First-iteration setup of subproblem ``s``: linking pairs, initial multipliers, logging and output paths."""
    sp = self.subProblems[s]
    sp.set_pair()
    self.Coordinator.v = [0.] * len(sp.coord._linker[0])
    # Initial penalty weight (coord.w_init, default 1); problems whose objective is large relative to the
    # range-normalised inconsistencies need a larger start so the couplings are not ignored early on.
    self.Coordinator.w = [float(getattr(self.Coordinator, "w_init", 1.0))] * len(sp.coord._linker[0])
    sp.coord = copy.deepcopy(self.Coordinator)
    sp.log = self.log
    sp.postDir = self.post_dir

  def _after_sweep(self) -> None:
    """Coordinator update once all subproblems of the outer iteration are solved."""
    self.Coordinator.calc_inconsistency()
    if self.Coordinator.is_AI_regularizer:
      self._update_detector_outer_iteration(self.iter)
    if not getattr(self.Coordinator, "single_multiplier_update", False):
      self.Coordinator.update_multipliers(self.iter)

  def _worst_link_index(self) -> int:
    qb = self.Coordinator.batch_q(self.Coordinator.q)
    ql = [max(np.abs(q)) if isinstance(q, list) else abs(q) for q in qb]
    return int(np.argmax(ql))

  def _display_iteration(self, iter: int, dx: float, index: int) -> None:
    coord = self.Coordinator
    formatted_msg = self.format_log_message(
        iter,
        np.max(np.abs(coord.q)),
        self.fmin[0] if isinstance(self.fmin, list) else self.fmin,
        dx,
        np.max(coord.w)
    )
    self.log.log_msg(msg=formatted_msg, msg_type=MSG_TYPE.INFO.value)
    print(formatted_msg)
    a = coord.master_vars[coord._linker[0, index]-1]
    b = coord.master_vars[coord._linker[1, index]-1]
    self.log.log_msg(msg=f'Highest inconsistency : {a.name}_{a.sp_index} to {b.name}_{b.link}',
                     msg_type=MSG_TYPE.INFO.value)
    print(f'Highest inconsistency : {a.name}_{a.link} to {b.name}_{b.link}')

  def _write_iteration_row(self, iter: int, dx: float, index: int) -> None:
    """Append this outer iteration to the coordination history file."""
    coord = self.Coordinator
    hmin_cols = self._get_hmin_columns()
    keys = self._get_ol_keys()
    curr_time = time.strftime("%H:%M:%S", time.localtime())
    hmin = max(self.hmin) if isinstance(self.hmin, list) and self.hmin is not None and len(self.hmin) > 0 else self.hmin
    hstatus = "Feasible" if hmin <= 0 else "Infeasible"
    status = copy.deepcopy(hstatus) if self.fmin != np.inf else "Error"
    a = coord.master_vars[coord.extended_linker[0, index]-1]
    b = coord.master_vars[coord.extended_linker[1, index]-1]
    cmax = f'{a.name}_{a.link} to {b.name}_{b.link}'
    if coord.is_AI_regularizer:
      values = [curr_time, iter, np.max(np.abs(coord.q)), coord.spike_location,
                int(bool(getattr(coord, "alarm_detected", False))), int(bool(getattr(coord, "spike_detected", False))),
                self.fmin, coord.spike_prob, coord.forecast_prob, coord.non_improvement_prob,
                coord.objective_non_improvement_prob, coord.constraint_non_improvement_prob,
                coord.detector_uncertainty, coord.constraint_pressure, status, dx, np.max(coord.w), cmax]
    else:
      values = [curr_time, iter, np.max(np.abs(coord.q)), int(bool(getattr(coord, "alarm_detected", False))),
                int(bool(getattr(coord, "spike_detected", False))), self.fmin, status, dx, np.max(coord.w), cmax]
    row = {keys[0]: str(values[0])}
    for k, value in zip(keys[1:len(values)], values[1:]):
      row[k] = str(value).rjust(30)
    if coord.is_AI_regularizer and getattr(coord, "spike_adaptive_scales", True):
      # Scales actually applied by update_multipliers this iteration (recomputing them here would see the cool-down).
      _eff_w, _eff_g, _eff_b, _event = getattr(coord, "_applied_scales", (1.0, 1.0, 1.0, ""))
      row[f'{"Eff_spike_w_scale".rjust(30)}'] = f'{_eff_w:.6f}'.rjust(30)
      row[f'{"Eff_spike_gamma_scale".rjust(30)}'] = f'{_eff_g:.6f}'.rjust(30)
      row[f'{"Eff_budget_scale".rjust(30)}'] = f'{_eff_b:.6f}'.rjust(30)
      row[f'{"Adapt_event".rjust(30)}'] = str(_event or "none").rjust(30)

    # Add hmin values for each subproblem
    for sp_index, hmin_col in zip(self._get_sp_indices(), hmin_cols):
      row[hmin_col] = str(self.hmin_dict.get(sp_index, [0.])).rjust(30)
    lr = len(row)
    for cv, v in enumerate(coord.master_vars):
      row[keys[lr+cv]] = str(v.value).rjust(30)
    self.Add_OL_res_row(row)

  def run(self, file=None, resume=False, mode="Serial", executor=None, sweep=None, max_workers=None):  # noqa: C901
    """Run the coordination loop.

    :param mode: ``"serial"`` (default sweep Gauss-Seidel, in-process) or ``"parallel"`` (default sweep Jacobi, process
      pool with one worker per subproblem).
    :param executor: an :class:`~DMDO.ports.Executor` instance or a registered executor name (``"inprocess"``,
      ``"process"`` or a plugin) that runs the subproblem solves; overrides the one implied by ``mode``.
    :param sweep: ``"gauss_seidel"`` or ``"jacobi"``; overrides the one implied by ``mode``.
    :param max_workers: worker count for executors created from a name (default: number of subproblems).
    """
    global eps_fio, eps_qio
    if file is not None:
      self.post_dir = os.path.dirname(file)
    if self.log is None:
      self.log: logger = logger()
      self.log.initialize(os.path.join(self.post_dir, "DMDO.log"), handler_name="DMDO")
    if not resume:
      self.log.log_msg(msg="Running MDO ... ", msg_type=MSG_TYPE.INFO.value)
      # Note: once you run MDAO, the data stored in eps_fio and eps_qio shall be deleted.
      # It is recommended to store such data to a different variable before running another MDAO
      eps_fio = []
      eps_qio = []
      self.file = file
    else:
      self.log.log_msg(msg=f"Resuming MDO coordination at iteration #{self.iter} ... ", msg_type=MSG_TYPE.INFO.value)
      file = self.file

    self.fmin = np.inf
    self.mode = mode
    self.Coordinator.mode = mode
    run_sweep = self._resolve_sweep(mode, sweep)
    owns_executor = executor is None or isinstance(executor, str)
    if executor is None:
      executor = "inprocess" if self._is_serial(mode) else "process"
    executor = create_executor(executor, max_workers=max_workers or len(self.subProblems))

    if self.Coordinator.is_AI_regularizer:
      detector = self._get_detector()
      if detector is not None and detector.tol_floor <= 0.0:
        detector.tol_floor = float(self.inc_stop)
      # Tolerance reference for the threshold trigger (coordinator._threshold_spike).
      self.Coordinator._tol_ref = float(self.inc_stop)
    try:
      for iter in range(self.Coordinator.budget):
        self.iter = iter
        # Initialize hmin dictionary for each SP at the start of each iteration
        self.hmin_dict = {}
        if self.Coordinator.is_AI_regularizer:
          self._reset_ai_regularizer_outputs()
        if iter > 0:
          with open(os.path.join(self.post_dir, 'coord_chk.pkl'), "wb") as fh:
            pickle.dump(self, fh)
          self.Coordinator.master_vars_old = copy.deepcopy(self.Coordinator.master_vars)
        else:
          self.Coordinator.master_vars_old = copy.deepcopy(self.variables)
          self.Coordinator.master_vars = copy.deepcopy(self.variables)
          self.prepare_post()

        run_sweep(executor)
        self._after_sweep()

        dx = self.get_master_vars_difference()
        index = self._worst_link_index()
        if self.display:
          self._display_iteration(iter, dx, index)
        # The objective history must include this iteration before the adaptation scales are evaluated.
        self.Coordinator.objective_history.append(float(self.fmin))
        if self.Coordinator.is_AI_regularizer and hasattr(self.Coordinator, "_get_expected_spike_event"):
          self.Coordinator._get_expected_spike_event(iter)
        self._write_iteration_row(iter, dx, index)

        # Update Lagrange multipliers and penalty weights
        self.Coordinator.update_multipliers(iter)
        self._rollback_step(iter)

        # Stopping criteria
        self.tab_inc.append(np.max(np.abs(self.Coordinator.q)))
        if self.check_termination_critt(iter):
          break
        self.eps_qio = copy.deepcopy(eps_qio)
        self.eps_fio = copy.deepcopy(eps_fio)
    finally:
      if owns_executor:
        executor.close()

  def validation(self, vType: int):
    self.term_status = []
    for i in range(len(self.term_critteria)):
      if self.term_type[i] == vType:
          self.term_status.append(self.term_critteria[i])

  def setInputs(self, values: List[Any]):
    self.variables = copy.deepcopy(values)

  def getOutputs(self):
    return self.responses
