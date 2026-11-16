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
from dataclasses import dataclass
import os
import pickle
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..subproblem.subproblem import SubProblem
from ..utils.log import MSG_TYPE, logger
from ..core.protocols import Process_data
from ..coordination.admm import ADMM
from ..core.variables import variableData
from ..ports import IterationRecord
from .execution import SubproblemTask, create_executor, solve_task
from .observers import HISTORY_FILE, ConsoleReporter, CoordinationHistoryRecorder, default_observers, notify

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

  # ------------------------------------------------------------------------------------------------ outputs
  # The coordination history and console output are written by observers (workflow/observers.py); these methods are
  # kept for scripts that call them directly.
  def _prepare_post_dir(self) -> None:
    os.makedirs(self.post_dir, exist_ok=True)
    self.postDir = self.post_dir
    self.Ol_file = os.path.join(self.post_dir, HISTORY_FILE)

  def prepare_post(self, file=None):
    """Create the output directory and the coordination history header (``file`` is accepted and ignored)."""
    self._prepare_post_dir()
    CoordinationHistoryRecorder().write_header(self)

  def _get_ol_keys(self) -> List[str]:
    return CoordinationHistoryRecorder().columns(self)

  def initialize_OL_res_file(self, file):
    CoordinationHistoryRecorder().write_header(self, file)

  def Add_OL_res_row(self, r: Dict):
    CoordinationHistoryRecorder().append_row(self, r)

  def format_log_message(self, iter, qmax, obj, dx, wmax):
    return ConsoleReporter.format_line(iter, qmax, obj, dx, wmax)

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

  def _worst_link_index(self) -> int:
    qb = self.Coordinator.batch_q(self.Coordinator.q)
    ql = [max(np.abs(q)) if isinstance(q, list) else abs(q) for q in qb]
    return int(np.argmax(ql))

  def run(self, file=None, resume=False, mode="Serial", executor=None, sweep=None, max_workers=None,  # noqa: C901
          observers=None):
    """Run the coordination loop.

    :param mode: ``"serial"`` (default sweep Gauss-Seidel, in-process) or ``"parallel"`` (default sweep Jacobi, process
      pool with one worker per subproblem).
    :param executor: an :class:`~DMDO.ports.Executor` instance or a registered executor name (``"inprocess"``,
      ``"process"`` or a plugin) that runs the subproblem solves; overrides the one implied by ``mode``.
    :param sweep: ``"gauss_seidel"`` or ``"jacobi"``; overrides the one implied by ``mode``.
    :param max_workers: worker count for executors created from a name (default: number of subproblems).
    :param observers: :class:`~DMDO.ports.RunObserver` objects notified during the run, after DMDO's own observers.
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
    hooks = default_observers(self) + list(observers or [])

    try:
      notify(hooks, "on_run_start", self)
      for iter in range(self.Coordinator.budget):
        self.iter = iter
        # Initialize hmin dictionary for each SP at the start of each iteration
        self.hmin_dict = {}
        if iter == 0:
          self.Coordinator.master_vars_old = copy.deepcopy(self.variables)
          self.Coordinator.master_vars = copy.deepcopy(self.variables)
          self._prepare_post_dir()
        notify(hooks, "on_iteration_start", self, iter)
        if iter > 0:
          self.Coordinator.master_vars_old = copy.deepcopy(self.Coordinator.master_vars)

        run_sweep(executor)
        self.Coordinator.calc_inconsistency()
        notify(hooks, "on_sweep_end", self, iter)
        if not getattr(self.Coordinator, "single_multiplier_update", False):
          self.Coordinator.update_multipliers(iter)

        dx = self.get_master_vars_difference()
        index = self._worst_link_index()
        # The objective history must include this iteration before the adaptation event is evaluated.
        self.Coordinator.objective_history.append(float(self.fmin))
        assess = getattr(self.Coordinator, "assess_iteration", None)
        if assess is not None:
          assess(iter)
        record = IterationRecord(iteration=iter, objective=self.fmin,
                                 max_inconsistency=float(np.max(np.abs(self.Coordinator.q))),
                                 max_penalty_weight=float(np.max(self.Coordinator.w)), variables_change=dx,
                                 worst_link=index)
        notify(hooks, "on_iteration_end", self, record)

        # Update Lagrange multipliers and penalty weights
        self.Coordinator.update_multipliers(iter)
        notify(hooks, "on_multipliers_updated", self, iter)

        # Stopping criteria
        self.tab_inc.append(np.max(np.abs(self.Coordinator.q)))
        if self.check_termination_critt(iter):
          break
        self.eps_qio = copy.deepcopy(eps_qio)
        self.eps_fio = copy.deepcopy(eps_fio)
      notify(hooks, "on_run_end", self)
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
