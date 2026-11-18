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
import hashlib
from logging import warning
import os
import time
from typing import Any, Callable, Dict, List
import numpy as np

from ..utils.log import MSG_TYPE, logger
from ..core.enums import COUPLING_TYPE, PSIZE_UPDATE, VAR_TYPE, eps_fio
from ..core.protocols import process
from ..coordination.admm import ADMM
from ..core.variables import variableData
import pandas as pd
from ..ai.detector import HybridAnomalyDetector, latest_detector_metrics
from ..analysis.da import analyze_and_validate_types
from ..utils.rng import seed_all_rngs as _seed_all_rngs  # noqa: F401  (backward-compatible alias)
from ..ports import SolverRequest
from ..registry import solvers
from ..solvers import DEFAULT_SOLVER, create_solver


# Kept for backward compatibility; the values live with the OMADS strategy.
from ..solvers.omads_solver import PSIZE_FLOOR_REL as _PSIZE_FLOOR_REL, PSIZE_FLOOR_SPAN_CAP as _PSIZE_FLOOR_SPAN_CAP  # noqa: E402,F401


def _is_number(a) -> bool:
  return isinstance(a, (int, float, np.number)) and not isinstance(a, bool)


def _bool_from_value(value: Any, default: bool = False) -> bool:
  if value is None:
    return default
  if isinstance(value, bool):
    return value
  if isinstance(value, (int, np.integer)):
    return bool(value)
  if isinstance(value, str):
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
      return True
    if lowered in {"0", "false", "no", "off"}:
      return False
  return default

class partitionedProblemData:
  nv: int
  nr: int
  sp_index: int
  vars: List[variableData]
  resps: List[variableData]
  is_main: bool
  MDA_process: process
  coupling: List[float]
  solution: List[Any]
  solver: Any
  realistic_objective: bool = False
  optFunctions: List[Callable] = None
  obj: float = np.inf
  constraints: List[float] = [np.inf]
  frealistic: float = 0.
  scaling: float = 10.
  Il_file: str = None
  conf: Dict = None
  coord: ADMM
  opt: Callable
  fmin_nop: float
  budget: int
  display: bool
  psize: float
  psize_init: int
  tol: float
  scipy: Dict
  sets: Dict
  fmin: float
  cache: pd.DataFrame
  hmin: float
  log: logger = None
  RANDOM_SEED: int = 1234
  update_bl: bool = True
  job: Any = None
  allConf: Dict = None 
  bl: List[float] = None
  tweak_csv: str = None
  _tweak_df: pd.DataFrame = None
  _AI_regularizer: HybridAnomalyDetector = None

def _is_index_key(key) -> bool:
  """True for an integer variable index (not a bool) used as a start_point key."""
  return isinstance(key, (int, np.integer)) and not isinstance(key, bool)


@dataclass
class SubProblem(partitionedProblemData):
  # Constructor
  def __init__(self, nv, index, vars, resps, is_main, analysis, coordination, opt, fmin_nop, budget, display, scipy=None, \
               psize=1, pupdate=PSIZE_UPDATE.LAST, freal=None, tol=1E-12, solver='OMADS', sets=None, conf = None, \
                realistic_objective=False, log: logger=None, tweak_csv: str=None, wd: str = None,
               start_point: Any = "warm"):
    self.nv = nv
    self.index = index
    self.vars = copy.deepcopy(vars)
    self.resps = copy.deepcopy(resps)
    self.is_main = is_main
    self.MDA_process = analysis
    self.coord = coordination
    self.opt = opt
    self.fmin_nop = fmin_nop
    self.budget=budget
    self.display = display
    self.psize = psize
    self.psize_init = pupdate
    self.frealistic = freal
    self.tol = tol
    self.solver = solver
    self.realistic_objective = realistic_objective
    self.fmin = np.inf
    self.hmin = np.inf
    self.cache = self._create_cache_frame()
    self.eval_ID = 0
    self.tweak_csv = tweak_csv
    self.RANDOM_SEED = self._get_base_seed(conf)
    if self.coord.is_AI_regularizer:
      history_file = os.path.join(wd, 'Coordination_history.out') if wd is not None else None
      warm_start_history = _bool_from_value(
        getattr(self.coord, "detector_warm_start_from_history", False),
        default=False,
      )
      if not warm_start_history and isinstance(conf, dict):
        options = conf.get("options")
        if isinstance(options, dict):
          warm_start_history = _bool_from_value(options.get("detector_warm_start_from_history"), default=False)
      existing_regularizer = getattr(self.coord, "ai_regularizer", None)
      if existing_regularizer is None:
        self.coord.ai_regularizer = HybridAnomalyDetector(
          data_file=history_file if warm_start_history else None,
          window=int(getattr(self.coord, "detector_window", 8)),
          min_history=int(getattr(self.coord, "detector_min_history", 5)),
          seed=self.RANDOM_SEED + int(self.index),
          refit_every=int(getattr(self.coord, "refit_every", 20)),
          **dict(getattr(self.coord, "detector_options", None) or {}),
        )
      # Keep the legacy SP-local alias for compatibility with older checkpoints.
      self._AI_regularizer = self.coord.ai_regularizer
    if self.tweak_csv is not None:
      self._tweak_df = pd.read_csv(self.tweak_csv)
    else:
      self._tweak_df = None
    if log is not None:
      self.log = log
    self.sets = sets
    self.conf = conf
    canonical = solvers.canonical_name(solver) if isinstance(solver, str) else None
    if canonical is None and isinstance(solver, str) and solver in solvers:
      canonical = solver  # "package.module:SolverClass" import path
    if canonical is None:
      self._warn(f'Inappropriate solver method definition for subproblem # {self.index}! OMADS will be used.')
      canonical = DEFAULT_SOLVER
    self.solver = canonical
    self.scipy = self._solver_impl().prepare_settings(scipy, self.index, self._warn)
    self.set_start_point(start_point)

  # ------------------------------------------------------------------------------------------------ start point
  START_POINT_MODES = ("warm", "initial")

  def set_start_point(self, start_point: Any = "warm") -> None:
    """Choose where every solve of this subproblem starts.

    * ``"warm"`` (default): from the last solution (the current values of the design variables).
    * ``"initial"``: always from the design-variable values the subproblem was built with (YAML ``bl``).
    * a mapping ``{variable: value}``: always from these values; design variables not listed keep their initial
      values. A variable is given by its index (``variableData.index``, which is what a YAML variable key such as
      ``v27`` is translated to when the problem is built) or by its name. A name shared by several scalar design
      variables takes one value for all of them or a list with one value each, in the subproblem's order; an
      index entry wins over a name entry for the same variable. Vector variables take a list of ``dim`` values.

    The initial values are those at the time of the call, so call it before the run (the YAML ``start_point`` key
    of the subproblem does this when the problem is built).
    """
    if isinstance(start_point, str):
      mode = start_point.strip().lower()
      if mode not in self.START_POINT_MODES:
        raise ValueError(f"Subproblem {self.index}: start_point must be 'warm', 'initial' or a mapping "
                         f"{{variable: value}}, got '{start_point}'.")
      start_point = mode
    elif isinstance(start_point, dict):
      design = self._start_point_design_vars()
      indices = {int(v.index) for v in design}
      names = {v.name for v in design}
      unknown = sorted((str(k) for k in start_point
                        if not ((_is_index_key(k) and int(k) in indices) or (isinstance(k, str) and k in names))))
      if unknown:
        raise ValueError(f"Subproblem {self.index}: start_point refers to {unknown}, which are not design "
                         f"variables of this subproblem; design variables: "
                         f"{[(int(v.index), v.name) for v in design]} (index, name).")
    elif start_point is not None:
      raise TypeError(f"Subproblem {self.index}: start_point must be a string or a mapping, got "
                      f"{type(start_point).__name__}.")
    self.start_point = "warm" if start_point is None else copy.deepcopy(start_point)
    self._initial_x0 = self._start_vector(self.start_point)

  def _start_point_design_vars(self) -> List[variableData]:
    return [v for v in self.get_design_vars() if v.coupling_type != COUPLING_TYPE.CONSTANT]

  def _start_vector(self, start_point: Any) -> List[Any]:
    """Flattened design vector (``get_list_vars`` order) of the initial values, with a mapping's values applied."""
    design = self._start_point_design_vars()
    by_index = ({int(k): v for k, v in start_point.items() if _is_index_key(k)}
                if isinstance(start_point, dict) else {})
    by_name = ({k: v for k, v in start_point.items() if isinstance(k, str)}
               if isinstance(start_point, dict) else {})
    copies = {name: sum(1 for v in design if v.name == name and not isinstance(v.baseline, list))
              for name in {v.name for v in design}}
    seen = {name: 0 for name in copies}
    x0: List[Any] = []
    for var in design:
      values = copy.deepcopy(var.baseline)
      label = None
      if int(var.index) in by_index:
        values, label = copy.deepcopy(by_index[int(var.index)]), f"variable {int(var.index)} ({var.name})"
      elif var.name in by_name:
        values, label = copy.deepcopy(by_name[var.name]), f"'{var.name}'"
        if not isinstance(var.baseline, list) and isinstance(values, list):
          # A list for a name shared by several scalar variables: one value each, in the subproblem's order.
          if len(values) != copies[var.name]:
            raise ValueError(f"Subproblem {self.index}: start_point value of '{var.name}' has {len(values)} "
                             f"entries, but the subproblem has {copies[var.name]} design variable(s) named "
                             f"'{var.name}'.")
          values = values[seen[var.name]]
          seen[var.name] += 1
      if not isinstance(var.baseline, list) and isinstance(values, list):
        raise ValueError(f"Subproblem {self.index}: start_point value of {label} must be a single value.")
      if label is not None and isinstance(var.baseline, list) and (
              not isinstance(values, list) or len(values) != len(var.baseline)):
        raise ValueError(f"Subproblem {self.index}: start_point value of {label} must be a list of "
                         f"{len(var.baseline)} entries (the variable's dim).")
      typ = var.type[0].lower() if var.dim == 1 else var.type[0][0].lower()
      categorical = typ in ("c", "d")
      for value in (values if isinstance(values, list) else [values]):
        x0.append(self.sets[var.set].index(value) if categorical else value)
    return x0

  def _solve_start_point(self) -> List[Any]:
    """Start point of the next solve: the last solution, or the fixed one chosen with :meth:`set_start_point`."""
    if getattr(self, "start_point", "warm") == "warm":  # checkpoints from before 2612.1 have no start_point
      return self.get_list_vars(self.get_design_vars())
    return copy.deepcopy(self._initial_x0)

  def _warn(self, msg: str) -> None:
    log = getattr(self, "log", None)
    if log is not None and getattr(log, "log", None) is not None:
      log.log_msg(msg=msg, msg_type=MSG_TYPE.WARNING.value)
    warning(msg)

  def _solver_impl(self):
    """The solver strategy for the current ``self.solver`` name (resolved through :data:`DMDO.registry.solvers`)."""
    cached = self.__dict__.get("_solver_cache")
    if cached is None or cached[0] != self.solver:
      cached = (self.solver, create_solver(self.solver))
      self.__dict__["_solver_cache"] = cached
    return cached[1]

  def _get_base_seed(self, conf: Dict) -> int:
    if isinstance(conf, dict):
      options = conf.get("options")
      if isinstance(options, dict) and "seed" in options:
        try:
          return int(options["seed"])
        except (TypeError, ValueError):
          return int(self.RANDOM_SEED)
    return int(self.RANDOM_SEED)

  def __getstate__(self):
    # Remove or replace non-picklable attributes
    state = self.__dict__.copy()
    # Example: remove or replace RLock
    if '_lock' in state:
        del state['_lock']
    # Solver strategies are re-resolved from the registry by name after unpickling.
    state.pop('_solver_cache', None)
    # Or replace with a placeholder
    # state['_lock'] = None
    return state

  def __setstate__(self, state):
      self.__dict__.update(state)
      if not isinstance(self.cache, pd.DataFrame):
        self.cache = self._migrate_legacy_cache(self.cache)
      # Reconstruct non-picklable objects if needed
      # self._lock = threading.RLock()
        

  def _create_cache_frame(self) -> pd.DataFrame:
    cache = pd.DataFrame(columns=["objective", "penalty", "vlist", "outputs"])
    cache.index.name = "cache_address"
    return cache

  def _migrate_legacy_cache(self, cache_entries: Any) -> pd.DataFrame:
    cache = self._create_cache_frame()
    if not isinstance(cache_entries, list):
      return cache
    for entry in cache_entries:
      if isinstance(entry, list) and len(entry) >= 4:
        self._append_cache_row(entry[0], entry[1], entry[2], entry[3], cache)
    return cache

  def _normalize_cache_value(self, value: Any):
    if isinstance(value, np.ndarray):
      return self._normalize_cache_value(value.tolist())
    if isinstance(value, list):
      return [self._normalize_cache_value(item) for item in value]
    if isinstance(value, tuple):
      return tuple(self._normalize_cache_value(item) for item in value)
    if isinstance(value, (np.integer, np.floating)):
      return value.item()
    return value

  def get_cache_address(self, vlist: List[Any]) -> int:
    normalized_vlist = self._normalize_cache_value(vlist)
    payload = repr(normalized_vlist).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), byteorder="big", signed=False)

  def cache_row(self, cache_address: int) -> pd.Series:
    return self.cache.loc[cache_address]

  def get_cached_row(self, vlist: List[Any]) -> pd.Series:
    return self.cache_row(self.get_cache_address(vlist))

  def _append_cache_row(self, objective: float, penalty: float, vlist: List[Any], outputs: Any, cache: pd.DataFrame = None):
    cache_frame = self.cache if cache is None else cache
    cache_address = self.get_cache_address(vlist)
    cache_frame.loc[cache_address, ["objective", "penalty", "vlist", "outputs"]] = [
      objective,
      penalty,
      copy.deepcopy(vlist),
      copy.deepcopy(outputs),
    ]


  def get_minimizer(self):
    v = []
    for i in range(len(self.vars)):
      v.append(self.vars[i])
    return v

  def get_coupling_vars(self):
    v = []
    for i in range(len(self.MDA_process.responses)):
      if self.MDA_process.responses[i].coupling_type == COUPLING_TYPE.FEEDFORWARD:
        v.append(self.MDA_process.responses[i])
    return v

  def get_design_vars(self):
    v = []
    for i in range(len(self.vars)):
      if self.vars[i].coupling_type != COUPLING_TYPE.FEEDFORWARD:
        v.append(self.vars[i])
    return v
  
  def modify_cond_vars(self, V: variableData):
     # TODO: Add a check to support conditionality on other variable properties
    for i in range(len(self.vars)):
      if self.vars[i].cond_on is not None:
        for v in V:
          if self.index == v.sp_index and v.name == self.vars[i].name and v.cond_on is not None:
            if self.vars[i].dim > v.dim:
              temp = self.vars[i].value[:v.dim]
              self.vars[i].__update__(temp)
            elif self.vars[i].dim < v.dim:
              temp = copy.deepcopy(self.vars[i].value)
              temp += [0]*(v.dim-self.vars[i].dim)
              self.vars[i].__update__(temp)

  def set_variables_value(self, vlist: List, clist: List=None):
    # FIXME: scipy doesn't consider constants list, if defined the routine crashes
    kv = 0
    kc = 0
    for i in range(len(self.vars)):
      if self.vars[i].coupling_type != COUPLING_TYPE.FEEDFORWARD and self.vars[i].coupling_type != COUPLING_TYPE.CONSTANT:
        if self.vars[i].dim > 1:
          for ik in range(self.vars[i].dim):
            self.vars[i].value[ik] = vlist[kv]
            self.vars[i].baseline[ik] = vlist[kv]
            kv += 1
        else:
          self.vars[i].value = vlist[kv]
          self.vars[i].baseline = vlist[kv]
          kv += 1
      elif self.vars[i].coupling_type == COUPLING_TYPE.CONSTANT and clist is not None:
        if self.vars[i].dim > 1:
          for ik in range(self.vars[i].dim):
            self.vars[i].value[ik] = clist[kc]
            self.vars[i].baseline[ik] = clist[kc]
            kc += 1
        else:
          self.vars[i].value = clist[kc]
          self.vars[i].baseline = clist[kc]
          kc += 1

  def set_variables_baseline(self, vlist: List, clist: List=None):
    # FIXME: scipy doesn't consider constants list, if defined the routine crashes
    kv = 0
    kc = 0
    for i in range(len(self.vars)):
      if self.vars[i].coupling_type != COUPLING_TYPE.FEEDFORWARD and self.vars[i].coupling_type != COUPLING_TYPE.CONSTANT:
        if self.vars[i].dim > 1:
          for ik in range(self.vars[i].dim):
            self.vars[i].baseline[ik] = vlist[kv]
            kv += 1
        else:
          self.vars[i].baseline = vlist[kv]
          kv += 1
      elif self.vars[i].coupling_type == COUPLING_TYPE.CONSTANT and clist is not None:
        if self.vars[i].dim > 1:
          for ik in range(self.vars[i].dim):
            self.vars[i].baseline[ik] = clist[kc]
            kc += 1
        else:
          self.vars[i].baseline = clist[kc]
          kc += 1

  def set_pair(self):
    indices1 = []
    indices2 = []
    sp_link =  []
    sp_link_to = []
    for i in range(len(self.coord.master_vars)):
      check: bool = False
      if self.coord.master_vars[i].link and isinstance(self.coord.master_vars[i].link, list):
        nl = len(self.coord.master_vars[i].link)
        check = any(self.coord.master_vars[i].link >= np.ones(nl))
      elif self.coord.master_vars[i].link:
        check = self.coord.master_vars[i].link >= 1
      if (self.index == self.coord.master_vars[i].sp_index \
          or self.coord.master_vars[i].coupling_type != COUPLING_TYPE.UNCOUPLED) \
            and check and self.coord.master_vars[i].index not in indices2:
        sp_link.append(self.coord.master_vars[i].sp_index)
        linked_to = (self.coord.master_vars[i].link)
        if linked_to and isinstance(linked_to, list):
          for linki in range(len(linked_to)):
            indices1.append(self.coord.master_vars[i].index)
        else:
          indices1.append(self.coord.master_vars[i].index)
        for j in range(len(self.coord.master_vars)):
          check = False
          if linked_to and isinstance(linked_to, list):
            nl = len(linked_to)
            check = any(linked_to == np.multiply(self.coord.master_vars[j].sp_index, np.ones(nl)))
          else:
            check = (linked_to == self.coord.master_vars[j].sp_index)
          if check and (self.coord.master_vars[j].name == self.coord.master_vars[i].name):
            sp_link_to.append(self.coord.master_vars[i].link)
            indices2.append(self.coord.master_vars[j].index)
    self.coord._linker = copy.deepcopy(np.array([indices1, indices2]))
    # Keep only paired links when list lengths diverge.
    # pair_count = min(len(indices1), len(indices2))
    # self.coord._linker = copy.deepcopy(np.array([indices1[:pair_count], indices2[:pair_count]], dtype=int))

  def getLocalIndices(self):
    local_link = []
    for i in range(len(self.coord._linker[0])):
      for j in range(len(self.vars)):
        if (self.coord._linker[0])[i] == self.vars[j].index or (self.coord._linker[1])[i] == self.vars[j].index:
          local_link.append(i)
      for k in range(len(self.resps)):
        if (self.coord._linker[0])[i] == self.resps[k].index or (self.coord._linker[1])[i] == self.resps[k].index:
          local_link.append(i)
    local_link.sort()
    return local_link
    

  def initialize_IL_res_file(self, file):
    if not os.path.exists(file):
      mode = 'x'
    else:
      mode = 'w'
    with open(file, mode=mode) as csv_file:
      keys = [f'{"Time"}', f'{"Iteration #".rjust(30)}', f'{"Evaluation #".rjust(30)}', f'{"Max. inconsistency".rjust(30)}', \
              f'{"Penalty".rjust(30)}', f'{"Objective".rjust(30)}', f'{"Coupling Status".rjust(30)}'] + \
                [f'{f"{x.name}".rjust(30)}' for x in self.vars] + \
                  [f'{f"{y.name}".rjust(30)}' for y in self.resps] 
      writer = csv.DictWriter(csv_file, fieldnames=keys)
      writer.writeheader()    # add column names in the CSV file
  
  def prepare_post(self, file):
    ht = os.path.split(file)
    # Base name only: a relative or dotted path in `file` would otherwise be nested inside its own directory.
    name = os.path.splitext(os.path.basename(file))[0]
    pd = os.path.join(ht[0], f'{name}_post')
    pdsb = os.path.join(pd, f'SP_{self.index}')
    pfo = os.path.join(pdsb, f'SP_{self.index}_{self.iter}.out')
    # exist_ok: subproblems solved concurrently (Jacobi sweep in worker processes) share the post directory.
    os.makedirs(pdsb, exist_ok=True)
    self.postDir = pdsb
    self.Il_file = pfo
    self.initialize_IL_res_file(pfo)
    
  def Add_IL_res_row(self, r: Dict):
    with open(self.Il_file, mode='a') as csv_file:
      keys = [f'{"Time"}', f'{"Iteration #".rjust(30)}', f'{"Evaluation #".rjust(30)}', f'{"Max. inconsistency".rjust(30)}', \
              f'{"Penalty".rjust(30)}', f'{"Objective".rjust(30)}', f'{"Coupling Status".rjust(30)}'] + \
                [f'{f"{x.name}".rjust(30)}' for x in self.vars] + [f'{f"{y.name}".rjust(30)}' 
                                                                   for y in self.MDA_process.responses] 
      writer = csv.DictWriter(csv_file, fieldnames=keys)
      writer.writerow(r)    # add column names in the CSV file

  def _scalarize_hmin(self, value: Any) -> float:
    array = np.atleast_1d(np.asarray(value, dtype=float))
    if array.size == 0:
      return 0.0
    finite = array[np.isfinite(array)]
    if finite.size > 0:
      return float(np.max(finite))
    if np.any(array > 0.0):
      return float(np.inf)
    return 0.0

  def _get_detector_slot(self, raw_index: int, count: int, prefer_one_based: bool = True) -> int:
    if count <= 0:
      return 0
    if prefer_one_based and 1 <= raw_index <= count:
      return raw_index - 1
    if 0 <= raw_index < count:
      return raw_index
    return min(max(raw_index - 1, 0), count - 1)

  def _get_runtime_hmins_for_detector(self) -> List[float]:
    count = self.coord.nsp if hasattr(self.coord, "nsp") else 1
    hmin_values = [0.0] * count
    history_file = (
      os.path.join(os.path.dirname(self.postDir), "Coordination_history.out")
      if hasattr(self, "postDir")
      else None
    )
    prefer_one_based = True

    if history_file is not None and os.path.exists(history_file):
      try:
        df = pd.read_csv(history_file, sep=None, engine="python")
        if not df.empty:
          columns = []
          for column in df.columns:
            label = str(column).strip()
            if label.startswith("hmin_"):
              try:
                columns.append((int(label.split("_")[-1]), column))
              except ValueError:
                continue
          columns.sort(key=lambda item: item[0])
          if any(idx == 0 for idx, _ in columns):
            prefer_one_based = False
          last_row = df.iloc[-1]
          for idx, column in columns:
            slot = self._get_detector_slot(idx, len(hmin_values), prefer_one_based)
            hmin_values[slot] = self._scalarize_hmin(last_row[column])
      except Exception:
        pass

    local_index = self._get_detector_slot(self.index, len(hmin_values), prefer_one_based)
    if 0 <= local_index < len(hmin_values):
      hmin_values[local_index] = self._scalarize_hmin(self.hmin)
    elif hmin_values:
      hmin_values[0] = self._scalarize_hmin(self.hmin)

    return hmin_values

  def _resolve_ai_regularizer(self):
    if not getattr(self.coord, "is_AI_regularizer", False):
      return None
    regularizer = getattr(self.coord, "ai_regularizer", None)
    if regularizer is None and self._AI_regularizer is not None:
      regularizer = self._AI_regularizer
      self.coord.ai_regularizer = regularizer
    return regularizer

  def _collect_regularizer_metrics(self, regularizer) -> Dict[str, float]:
    return latest_detector_metrics(regularizer)

  def _apply_regularizer_metrics_to_coord(self, metrics: Dict[str, float]):
    current_event_strength = float(max(
      metrics["forecast_prob"],
      metrics["non_improvement_prob"],
      metrics["objective_non_improvement_prob"],
      metrics["constraint_non_improvement_prob"],
      metrics["spike_prob"],
    ))
    best_event_strength = float(getattr(self.coord, "ai_event_strength", -1.0))
    if current_event_strength < best_event_strength:
      return

    if hasattr(self.coord, "ai_event_strength"):
      self.coord.ai_event_strength = current_event_strength
    if hasattr(self.coord, "qmax_predicted"):
      self.coord.qmax_predicted = metrics["forecast_prob"]
    if hasattr(self.coord, "fmin_predicted"):
      self.coord.fmin_predicted = metrics["objective_non_improvement_prob"]
    if hasattr(self.coord, "spike_prob"):
      self.coord.spike_prob = metrics["spike_prob"]
    if hasattr(self.coord, "forecast_prob"):
      self.coord.forecast_prob = metrics["forecast_prob"]
    if hasattr(self.coord, "non_improvement_prob"):
      self.coord.non_improvement_prob = metrics["non_improvement_prob"]
    if hasattr(self.coord, "objective_non_improvement_prob"):
      self.coord.objective_non_improvement_prob = metrics["objective_non_improvement_prob"]
    if hasattr(self.coord, "constraint_non_improvement_prob"):
      self.coord.constraint_non_improvement_prob = metrics["constraint_non_improvement_prob"]
    if hasattr(self.coord, "detector_uncertainty"):
      self.coord.detector_uncertainty = metrics["detector_uncertainty"]
    if hasattr(self.coord, "constraint_pressure"):
      self.coord.constraint_pressure = metrics["constraint_pressure"]
    if hasattr(self.coord, "spike_location"):
      self.coord.spike_location = self.index

  def _update_ai_regularizer(self):
    # With the outer-iteration feed the coordinator updates the detector once per outer iteration.
    if getattr(self.coord, "detector_per_outer_iteration", True):
      return
    regularizer = self._resolve_ai_regularizer()
    if regularizer is None:
      return

    inc_value = float(np.max(np.abs(self.coord.q))) if self.coord.q is not None and np.size(self.coord.q) > 0 else 0.0
    obj_value = self.fmin_nop if np.isfinite(self.fmin_nop) else self.fmin
    hmin_values = self._get_runtime_hmins_for_detector()
    # Feed the completed SP-solve trajectory before the coordination-level update
    sp_file = getattr(self, "Il_file", None)
    if sp_file is not None:
        regularizer.update_sp_features(sp_file)
    alarm = regularizer.update(obj=obj_value, inc=inc_value, hmins=hmin_values)
    metrics = self._collect_regularizer_metrics(regularizer)
    self._apply_regularizer_metrics_to_coord(metrics)

    if hasattr(self.coord, "spike_location") and alarm and int(getattr(self.coord, "spike_location", 0)) <= 0:
      self.coord.spike_location = self.index
    if hasattr(self.coord, "alarm_detected"):
      self.coord.alarm_detected = bool(getattr(self.coord, "alarm_detected", False) or alarm)

  def get_con(self, x):
    # Opt-in (scipy setting ``cache_constraints``): reuse the constraints of an objective evaluation at the same
    # point in this solve instead of running the analysis again (SLSQP asks for f and c separately at every point).
    memo = self.__dict__.get("_con_memo")
    if memo is not None:
      hit = memo.get(self._memo_key(x))
      if hit is not None:
        return list(hit)
    return self.evaluate(x, None, "con_only")

  @staticmethod
  def _memo_key(x) -> tuple:
    return tuple(float(a) for a in np.ravel(np.asarray(x, dtype=float)))

  def _con_memo_enabled(self) -> bool:
    return (self._solver_impl().objective_form == "scalar" and isinstance(self.scipy, dict)
            and bool(self.scipy.get("is_con")) and bool(self.scipy.get("cache_constraints", False)))
  
  @staticmethod
  def _is_out_of_bounds(vlist, lb, ub) -> bool:
    return any(_is_number(x) and _is_number(lo) and _is_number(hi) and not (lo <= x <= hi)
               for x, lo, hi in zip(vlist, lb, ub))

  def evaluate(self, vlist: List[float]=None, clist: List[float]=None, *argv):  # noqa: C901
    is_conOnly = False
    if argv is not None and len(argv)>0 and argv[0] == "con_only":
      is_conOnly = True
    if self.coord.save_q_in_out:
      global eps_fio
    self.eval_ID += 1
    # If no variables were provided use existing value of the 
    # variables of the current subproblem (might happen during initialization)
    if vlist is None:
      v: List = self.get_design_vars()
      vlist: list = self.get_list_vars(v)
    elif getattr(self, "_omads_bounds", None) is not None and self._is_out_of_bounds(vlist, *self._omads_bounds):
      # Same output OMADS assigns to a failed evaluation.
      return [np.inf, [np.inf]]
    
    self.set_variables_value(vlist, clist)
    self.MDA_process.setInputs(self.vars)
    if argv is not None and len(argv) > 0 and argv[0] == "evalOnly":
      self.MDA_process.run()
      y = self.MDA_process.getOutputs()
      fun = self.opt(vlist, y, clist)
      return fun
    wb, vb = self.coord.batch_multipliers()
    if self.MDA_process.responses is not None:
      self.MDA_process.run()
      y = self.MDA_process.getOutputs()
    else:
      y = None

    inputs_type_analysis = analyze_and_validate_types(self.opt)
    if len(inputs_type_analysis) > 0 :
      if len(inputs_type_analysis) > 0:
        if inputs_type_analysis[list(inputs_type_analysis.keys())[0]] == "dict":
          inp1 = self.get_dict_vars(self.get_design_vars())
        else:
          inp1 = vlist
        
        if self.MDA_process.responses is not None \
        and len(inputs_type_analysis) > 1 \
        and inputs_type_analysis[list(inputs_type_analysis.keys())[1]] == "dict":
          inp2 = self.get_dict_outputs(self.MDA_process.responses)
        else:
          inp2 = y
        
        
    fun = self.opt(inp1, inp2, clist)
    
    self.fmin_nop = fun[0]
    con = fun[1]
    if self.MDA_process.responses is not None:
      con = self.get_coupling_vars_diff(con)
    con_coupling = self.get_coupling_vars_diff([0])

    if self.is_main:
      if self.frealistic is not None and self.frealistic != 0.:
        self.coord.eps_fo.append(abs(fun[0]-self.frealistic)/abs(self.frealistic))
        if self.coord.save_q_in_out:
          eps_fio.append(abs(fun[0]-self.frealistic)/abs(self.frealistic))
      else:
        self.coord.eps_fo.append(abs(fun[0]))
        if self.coord.save_q_in_out:
          eps_fio.append(abs(fun[0]))

    self.set_pair()
    self.coord.update_master_vector(self.vars, self.MDA_process.responses)
    self.coord.calc_inconsistency()
    qb = self.coord.batch_q(self.coord.q)
    qtemp = []
    for i in range(len(qb)):
      if isinstance(qb[i], list):
        qtemp.append(max(qb[i]))
      else:
        qtemp.append(qb[i])
    self.coord.modify_multipliers(qb, wb, vb)
    if self.Il_file is not None:
      curr_time = time.strftime("%H:%M:%S", time.localtime()) 
      hstatus = "Feasible" if max(con_coupling) <= 0 else "Infeasible"
      status = copy.deepcopy(hstatus) if fun[0] != np.inf else "Error"
      keys = [f'{"Time"}', f'{"Iteration #".rjust(30)}', f'{"Evaluation #".rjust(30)}', f'{"Max. inconsistency".rjust(30)}', \
              f'{"Penalty".rjust(30)}', f'{"Objective".rjust(30)}', \
                f'{"Coupling Status".rjust(30)}'] + \
                  [f'{f"{x.name}".rjust(30)}' for x in self.vars] + \
                    [f'{f"{y.name}".rjust(30)}' for y in self.MDA_process.responses] 

      row = {keys[0]: f'{f"{curr_time}"}', 
             keys[1]: f'{f"{self.iter}".rjust(30)}', 
             keys[2]: f'{f"{self.eval_ID}".rjust(30)}', 
             keys[3]: f'{f"{max(np.abs(self.coord.q))}".rjust(30)}', 
             keys[4]: f'{f"{self.coord.phi}".rjust(30)}', 
             keys[5]: f'{f"{fun[0]}".rjust(30)}', 
             keys[6]: f'{f"{status}".rjust(30)}'}
      cv = 0
      lr = len(row)
      for v in vlist:
        row[keys[lr+cv]] = f'{f"{v}".rjust(30)}'
        cv += 1
      cr = 0
      lr = len(row)
      for rs in self.MDA_process.responses:
        row[keys[lr+cr]] = f'{f"{rs.value}".rjust(30)}'
        cr += 1
      self.Add_IL_res_row(row)
    
    # TODO: Fix the local indices to select from a batched q list
    q_indices: List = self.getLocalIndices()
    self.coord.calc_inconsistency()
    self.coord.calc_penalty(q_indices, self.fmin_nop)
    # TODO: change the name of this routine
    self.coord.update_master_vector_val(self.vars, self.MDA_process.responses, self.sets)
    mvs = self.coord.master_vars
    for i in range(len(mvs)):
      for j in range(len(self.vars)):
        if mvs[i].index == self.vars[j].index:
          self.vars[j] = copy.deepcopy(mvs[i])

    self._append_cache_row(fun[0], self.coord.phi, vlist, self.MDA_process.getOutputs())
    # if self.realistic_objective:
    #   fun[0] =(y[0]-self.frealistic)
    if self._solver_impl().objective_form == "scalar":
      if self.scipy["is_con"]:
        if is_conOnly:
          return con
        else:
          memo = self.__dict__.get("_con_memo")
          if memo is not None and vlist is not None:
            memo[self._memo_key(vlist)] = list(con)
          return fun[0] + self.coord.phi
      else:
        hmax = 0.001
        h = max(max(con),hmax)
        return fun[0]+self.coord.phi + (max(max(con),0)**2 if h <= hmax else np.inf) 
    else:
      return [fun[0]+self.coord.phi, con]

  def _budget_scale(self, iter: int) -> float:  # noqa: C901
    """Inner-budget multiplier for this solve (AI regularizer: budget expansion after an attributed spike)."""
    budget_scale = 1
    configured_budget_scale = float(getattr(self.coord, "budget_scale", 6))
    # Intensity-modulated value computed by the coordinator at the previous outer iteration.
    spike_budget_scale = float(getattr(self.coord, "_effective_budget_scale", configured_budget_scale))

    # Budget expansion only follows a detected spike attributed to this subproblem.
    if not self.coord.is_AI_regularizer:
      return budget_scale
    min_iter = getattr(self.coord, "budget_min_iter", None)
    min_iter = int(self.coord.refit_every if min_iter is None else min_iter)
    if iter > min_iter and getattr(self.coord, "last_spike_detected", False) and getattr(self.coord, "adapt_budget", True):
      last_location = int(getattr(self.coord, "last_spike_location", 0))
      improving = getattr(self.coord, "_sp_late_improving", None)
      causal = str(getattr(self.coord, "spike_response", "accelerate")) == "causal"
      if causal and getattr(self.coord, "budget_neutral", False) and improving and spike_budget_scale > 1.0:
        # Recipients: subproblems still improving when their last budget ran out. Donors: those that had
        # settled. The total stays nsp * budget; no reallocation when every subproblem is in the same state.
        n_rec = sum(1 for v in improving.values() if v)
        n_don = len(improving) - n_rec
        if n_rec > 0 and n_don > 0:
          floor = float(getattr(self.coord, "budget_floor", 0.25))
          gain = min(spike_budget_scale, 1.0 + n_don * (1.0 - floor) / n_rec)
          budget_scale = gain if improving.get(self.index, False) else max(floor, 1.0 - (gain - 1.0) * n_rec / n_don)
      elif getattr(self.coord, "budget_neutral", False) and last_location > 0 and spike_budget_scale > 1.0:
        # Move evaluations to the attributed subproblem; the total over all subproblems stays nsp * budget.
        nsp = max(int(getattr(self.coord, "nsp", 1)), 1)
        floor = float(getattr(self.coord, "budget_floor", 0.25))
        if nsp > 1:
          gain = min(spike_budget_scale, 1.0 + (nsp - 1) * (1.0 - floor))
          budget_scale = gain if last_location == self.index else max(floor, (nsp - gain) / (nsp - 1))
      elif last_location == self.index:
        budget_scale = spike_budget_scale
    if budget_scale > 1.0 and getattr(self.coord, "budget_feasible_only", False):
      # Expand only a subproblem whose previous solve ended feasible; extra evaluations on an infeasible one
      # tend to trade its own constraints for objective and penalty terms.
      last_h = np.ravel(np.asarray(getattr(self, "hmin", [0.0]), dtype=float))
      last_h = last_h[np.isfinite(last_h)]
      if last_h.size and float(np.max(last_h)) > 1e-6:
        budget_scale = 1.0
    return budget_scale

  def _apply_tweaks(self) -> None:
    """Per-iteration overrides (solver, budget, gamma, beta) from the optional ``tweak_csv`` schedule."""
    if self._tweak_df is None or (self.iter+1) not in self._tweak_df['iteration'].values:
      return
    row = self._tweak_df.query(f'iteration == {self.iter+1}')
    name = row['solver'].iloc[0]
    canonical = solvers.canonical_name(name)
    if canonical is None:
      raise ValueError(f"tweak_csv selects unknown solver '{name}' for subproblem #{self.index}")
    self.solver = canonical
    self.budget = row['budget'].iloc[0]
    self.coord.gamma = row['gamma'].iloc[0]
    self.coord.beta = row['beta'].iloc[0]

  def _build_solver_request(self, impl, x0, budget: int, budget_scale: float, iter: int) -> SolverRequest:
    design = self.get_design_vars()
    return SolverRequest(
      name=f"SP_{self.index}",
      objective=self.evaluate,
      constraints=self.get_con,
      x0=x0,
      lb=self.get_list_vars_lb(design),
      ub=self.get_list_vars_ub(design),
      var_names=self.get_list_vars_names(design),
      var_types=self.get_vars_types(design),
      var_sets=self.sets,
      constants=self.get_list_constant_updates(design),
      constants_names=self.get_list_const_names(design),
      budget=budget,
      budget_scale=budget_scale,
      seed=int(self.RANDOM_SEED + int(iter)),
      iteration=int(iter) if iter is not None else 0,
      tol=self.tol,
      display=self.display,
      post_dir=getattr(self, "postDir", None),
      config=impl.select_config(self.conf, self.scipy),
      warm_start={"psize": self.psize, "psize_policy": self.psize_init},
    )

  def solve(self, v, w, file: str = None, iter: int = None):
    """Solve this subproblem for the given multipliers ``v`` and penalty weights ``w``.

    The optimization itself is delegated to the solver strategy registered under ``self.solver``
    (see :mod:`DMDO.solvers`). Returns the strategy's raw output (OMADS output dict / scipy ``x``).
    """
    if file is not None:
      self.iter = iter
      self.prepare_post(file)
    if self._solver_impl().refresh_dependent_baseline and self.is_main:
      self.set_dependent_baseline(self.coord.master_vars)
    self.cache = self._create_cache_frame()
    self.coord.v = copy.deepcopy(v)
    self.coord.w = copy.deepcopy(w)
    bl = self._solve_start_point()
    # Values carried over from other subproblems can land a round-off outside the bounds, which OMADS rejects
    # ("mesh center values must satisfy: lb <= x^mesh <= ub"), so clamp the start point onto the box.
    lbs = self.get_list_vars_lb(self.get_design_vars())
    ubs = self.get_list_vars_ub(self.get_design_vars())
    bl = [min(max(x, lo), hi) if _is_number(x) and _is_number(lo) and _is_number(hi) else x
          for x, lo, hi in zip(bl, lbs, ubs)]
    if not hasattr(self, "xmin_old"):
      self.xmin_old = copy.deepcopy(bl)
    if not hasattr(self, "fmin_old"):
      self.fmin_old = np.inf
    self.coord.ai_spike_objective_jump = False  # Initialize it to False at the start of each solve

    budget_scale = self._budget_scale(iter)
    effective_budget = max(1, int(round(float(self.budget) * float(budget_scale))))

    self.log.log_msg(msg=f"Coordination iteration # {iter}: running subproblem {self.index}", msg_type=MSG_TYPE.INFO.value)
    # A tweak's budget takes effect from the next solve (effective_budget is already fixed above).
    self._apply_tweaks()

    impl = self._solver_impl()
    request = self._build_solver_request(impl, bl, effective_budget, budget_scale, iter)
    log_copy = self.log
    # Some solvers evaluate trial points they fail to snap onto the bounds; reject them in evaluate() so they can
    # never become the next poll center (which OMADS refuses with "lb <= x^mesh <= ub").
    self._omads_bounds = (request.lb, request.ub) if impl.rejects_out_of_bounds else None
    # Constraint memo lives for one solve only (the multipliers and the other subproblems' values change between).
    self.__dict__["_con_memo"] = {} if self._con_memo_enabled() else None
    try:
      outcome = impl.solve(request)
    finally:
      self._omads_bounds = None
      self.__dict__["_con_memo"] = None
    self.log = log_copy
    self.fmin = outcome.fmin
    self.hmin = outcome.hmin
    if "psize" in outcome.warm_start:
      self.psize = outcome.warm_start["psize"]

    # Commit evaluation at the solution: updates the inconsistencies and the master vector from this solve.
    x_best = outcome.x if len(outcome.x) > 0 else bl
    self.evaluate(x_best, self.get_list_constant_updates(self.get_design_vars()))
    if self.fmin_nop < self.fmin_old:
      self.xmin_old = copy.deepcopy(x_best)
      self.fmin_old = copy.deepcopy(self.fmin_nop)
      if "psuccess" in outcome.warm_start:
        self.psize_old = outcome.warm_start["psuccess"]

    if getattr(self.coord, "is_AI_regularizer", False):
      self._update_ai_regularizer()
    self.log.log_msg(msg=f"Coordination iteration # {iter}: completed subproblem {self.index}", msg_type=MSG_TYPE.INFO.value)
    return outcome.raw

  def reported_hmin(self):
    """Constraint violation of the last solve in the form the coordination loop records."""
    return self._solver_impl().report_hmin(self.hmin)

  def get_coupling_vars_diff(self, con):
    # TODO: Add categorical coupling difference
    vc: List[variableData] = self.get_coupling_vars()
    for i in range(len(vc)):
      if isinstance(vc[i].value, list):
        for j in range(len(vc[i].value)):
          if vc[i].type[j][0].lower() == "c":
            con.append(float(self.sets[vc[i].set].index(vc[i].value[j])-vc[i].ub[j]))
          else:
            con.append(float(vc[i].value[j]-vc[i].ub[j]))
      else:
        if vc[i].type[0].lower() == "c":
          con.append(float(self.sets[vc[i].set].index(vc[i].value)-vc[i].ub))
        else:
          con.append(float(vc[i].value-vc[i].ub))

    for i in range(len(vc)):
      if isinstance(vc[i].value, list):
        for j in range(len(vc[i].value)):
          if vc[i].type[j][0].lower() == "c":
            con.append(float(vc[i].lb[j]-self.sets[vc[i].set].index(vc[i].value[j])))
          else:
            con.append(float(vc[i].lb[j]-vc[i].value[j]))
      else:
        if vc[i].type[0].lower() == "c":
          con.append(float(vc[i].lb-self.sets[vc[i].set].index(vc[i].value)))
        else:
          con.append(float(vc[i].lb-vc[i].value))
    return con

  def set_dependent_baseline(self, vars: List[variableData]):
    for j in range(len(self.vars)):
      if self.vars[j].coupling_type == COUPLING_TYPE.FEEDBACK or self.vars[j].coupling_type == COUPLING_TYPE.SHARED \
        or self.vars[j].coupling_type == COUPLING_TYPE.UNCOUPLED or self.vars[j].coupling_type == COUPLING_TYPE.CONSTANT:
        for i in range(len(vars)):
          if isinstance(self.vars[j].link , list):
            found = False
            for inx in self.vars[j].link:
              if inx == vars[i].sp_index:
                found = True
          else:
            found = self.vars[j].link == vars[i].sp_index
          if self.vars[j].name == vars[i].name and found:
            self.vars[j].baseline = vars[i].value
            self.vars[j].value = vars[i].value
    
    # for j in range(len(self.resps)):
    #   if self.resps[j].coupling_type == COUPLING_TYPE.FEEDFORWARD 
    # or self.resps[j].coupling_type == COUPLING_TYPE.SHARED or self.resps[j].coupling_type == COUPLING_TYPE.CONSTANT:
    #     for i in range(len(vars)):
    #       if isinstance(self.resps[j].link , list):
    #         found = False
    #         for inx in self.resps[j].link:
    #           if inx == vars[i].sp_index:
    #             found = True
    #       else:
    #         found = self.resps[j].link == vars[i].sp_index
    #       if self.resps[j].name == vars[i].name and found:
    #         self.resps[j].baseline = vars[i].value

    # for i in range(len(vars)):
    #   if vars[i].sp_index == self.index:
    #     if vars[i].coupling_type == COUPLING_TYPE.FEEDFORWARD 
    # or vars[i].coupling_type == COUPLING_TYPE.SHARED or vars[i].coupling_type == COUPLING_TYPE.CONSTANT:
    #       for j in range(len(self.vars)):
    #         if isinstance(self.vars[j].link , list):
    #           found = False
    #           for inx in self.vars[j].link:
    #             if inx == vars[i].sp_index:
    #               found = True
    #         else:
    #           found = self.vars[j].link == vars[i].sp_index
    #         if self.vars[j].name == vars[i].name and found:
    #           self.vars[j].baseline = vars[i].value

  def get_list_vars(self, vars:List[variableData]):
    v = []
    for i in range(len(vars)):
      if vars[i].coupling_type != COUPLING_TYPE.CONSTANT:
        if isinstance(vars[i].baseline, list):
          for j in range(len(vars[i].baseline)):
            typ = vars[i].type[0].lower() if vars[i].dim == 1 else vars[i].type[0][0].lower()
            temp = vars[i].baseline[j] if (typ != "c" and typ != "d") else self.sets[vars[i].set].index(vars[i].baseline[j])
            v.append(temp)
        else:
          typ = vars[i].type[0].lower() if vars[i].dim == 1 else vars[i].type[0][0].lower()
          v.append(vars[i].baseline if (typ != "c" and typ != "d") else self.sets[vars[i].set].index(vars[i].baseline))
    return v
  
  def get_dict_vars(self, vars:List[variableData]):
    v = {}
    for i in range(len(vars)):
      if vars[i].coupling_type != COUPLING_TYPE.CONSTANT:
        if isinstance(vars[i].baseline, list):
          for j in range(len(vars[i].baseline)):
            typ = vars[i].type[0].lower() if vars[i].dim == 1 else vars[i].type[0][0].lower()
            temp = vars[i].baseline[j] if (typ != "c" and typ != "d") else self.sets[vars[i].set].index(vars[i].baseline[j])
            v[vars[i].name] = temp
        else:
          typ = vars[i].type[0].lower() if vars[i].dim == 1 else vars[i].type[0][0].lower()
          v[vars[i].name] = vars[i].baseline if (typ != "c" and typ != "d") else self.sets[vars[i].set].index(vars[i].baseline)
    return v
  
  def get_dict_outputs(self, vars:List[variableData]):
    v = {}
    for i in range(len(vars)):
      if vars[i].coupling_type != COUPLING_TYPE.CONSTANT:
        if isinstance(vars[i].baseline, list):
          for j in range(len(vars[i].baseline)):
            typ = vars[i].type[0].lower() if vars[i].dim == 1 else vars[i].type[0][0].lower()
            temp = vars[i].baseline[j] if (typ != "c" and typ != "d") else self.sets[vars[i].set].index(vars[i].baseline[j])
            v[vars[i].name] = temp
        else:
          typ = vars[i].type[0].lower() if vars[i].dim == 1 else vars[i].type[0][0].lower()
          v[vars[i].name] = vars[i].value if (typ != "c" and typ != "d") else self.sets[vars[i].set].index(vars[i].baseline)
    return v


  def get_list_vars_ub(self, vars:List[variableData]):
    v = []
    for i in range(len(vars)):
      if vars[i].coupling_type != COUPLING_TYPE.CONSTANT:
        if isinstance(vars[i].ub, list):
          for j in range(len(vars[i].ub)):
            v.append(vars[i].ub[j])
        else:
          v.append(vars[i].ub)
    return v

  def get_list_vars_lb(self, vars:List[variableData]):
    v = []
    for i in range(len(vars)):
      if vars[i].coupling_type != COUPLING_TYPE.CONSTANT:
        if isinstance(vars[i].lb, list):
          for j in range(len(vars[i].lb)):
            v.append(vars[i].lb[j])
        else:
          v.append(vars[i].lb)
    return v

  def get_design_vars_scaling(self, vars:List[variableData]):
    v = []
    for i in range(len(vars)):
      if vars[i].coupling_type != COUPLING_TYPE.CONSTANT:
        if isinstance(vars[i].scaling, list):
          for j in range(len(vars[i].scaling)):
            v.append(vars[i].scaling[j])
        else:
          v.append(vars[i].scaling)
    return v
  
  # def get_sets(self, vars:List[variableData]):
  #   v = []
  #   for i in range(len(self.vars)):
  #     if vars[i].coupling_type != COUPLING_TYPE.CONSTANT and vars[i].set is not in v:
  #       v.append(self.vars[i].set)
  #   return v

  def get_list_vars_names(self,vars:List[variableData]):
    v = []
    for i in range(len(vars)):
      if vars[i].coupling_type != COUPLING_TYPE.CONSTANT:
        if isinstance(vars[i].value, list):
          for j in range(len(vars[i].value)):
            v.append(vars[i].name)
        else:
          v.append(vars[i].name)
    return v
  
  def get_list_const_names(self,vars:List[variableData]):
    v = []
    for i in range(len(vars)):
      if vars[i].coupling_type == COUPLING_TYPE.CONSTANT:
        if isinstance(vars[i].value, list):
          for j in range(len(vars[i].value)):
            v.append(vars[i].name)
        else:
          v.append(vars[i].name)
    if isinstance(v, list) and len(v)>0:
      return v
    else:
      return None
  
  def get_list_constant_updates(self, vars: List[variableData]):
    v = []
    for i in range(len(vars)):
      if vars[i].coupling_type == COUPLING_TYPE.CONSTANT:
        if isinstance(vars[i].value, list):
          for j in range(len(vars[i].value)):
            v.append(vars[i].value[j] if (vars[i].type[j].lower() == 'r' \
                                          or vars[i].type[j].lower() == 'i') else vars[i].value[j])
        else:
          v.append(vars[i].value if (vars[i].type[0].lower() == 'r' or vars[i].type[0].lower() == 'i') else vars[i].value)
    if isinstance(v, list) and len(v)>0:
      return v
    else:
      return None

  def get_vars_types(self,vars:List[variableData]):
    v = []
    for i in range(len(vars)):
      if vars[i].coupling_type != COUPLING_TYPE.CONSTANT:
        if isinstance(vars[i].type, list):
          for j in range(len(vars[i].type)):
            if vars[i].type[j] == VAR_TYPE.REAL.name:
              v.append("REAL")
            elif vars[i].type[j] == VAR_TYPE.INTEGER.name:
              v.append("INTEGER")
            elif vars[i].type[j] == VAR_TYPE.CATEGORICAL.name:
              v.append("CATEGORICAL")
            else:
              v.append(vars[i].type[j])
        else:
          if vars[i].type == VAR_TYPE.REAL.name:
            v.append("REAL")
          elif vars[i].type == VAR_TYPE.INTEGER.name:
            v.append("INTEGER")
          elif vars[i].type == VAR_TYPE.CATEGORICAL.name:
            v.append("CATEGORICAL")
          else:
            v.append(vars[i].type)
    return v
