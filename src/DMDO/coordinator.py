
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
import math
from logging import warning
from typing import Any, Dict, List, Optional

import numpy as np

from ._globals import COUPLING_TYPE, w_scheme, eps_qio
from .variables import variableData
from ._detector import HybridAnomalyDetector
from ._decisor import resolve_level
from dataclasses import dataclass, field

@dataclass
class coordinationData:
  nsp: int
  budget: int = 20
  index_of_master_SP: int = 1
  display: bool = True
  scaling: Any = 10.0
  mode: str = "serial"
  var_group: List[variableData] = field(default_factory=list)
  _linker: List[List[int]] = field(default_factory=list)
  n_dim: int = 0
  index: int = None
  is_AI_regularizer: bool = False
  qmax_predicted: float = np.inf
  spike_location: int = 0
  alarm_detected: bool = False
  spike_detected: bool = False
  last_spike_detected: bool = False
  last_spike_location: int = 0
  objective_history: List[float] = field(default_factory=list)
  ai_event_strength: float = -1.
  fmin_predicted: float = 0.
  spike_prob: float = 0.
  forecast_prob: float = 0.
  non_improvement_prob: float = 0.
  objective_non_improvement_prob: float = 0.
  constraint_non_improvement_prob: float = 0.
  detector_uncertainty: float = 1.
  constraint_pressure: float = 0.
  ai_regularizer: HybridAnomalyDetector = None
  AI_spike_policy: str = "L2_Normal"
  spike_confidence_level: str = "L2_Normal"
  spike_threshold: float = 0.5
  spike_influence_on_multipliers: bool = True
  spike_w_damp_scale: float = 2.2
  spike_w_amplify_scale: float = 0.8
  spike_gamma_scale: float = 0.05
  budget_scale: float = 6.0
  spike_adaptive_scales: bool = True
  refit_every: int = 20
  detector_warm_start_from_history: bool = False
  detector_window: int = 8
  detector_min_history: int = 5
  detector_options: Dict = field(default_factory=dict)
  detector_per_outer_iteration: bool = True
  adapt_beta: bool = True
  adapt_gamma: bool = True
  adapt_budget: bool = True
  adapt_cooldown: int = 0
  penalty_w_cap: float = np.inf
  w_stall_exponent: float = 2.0
  single_multiplier_update: bool = False
  spike_response: str = "accelerate"
  spike_w_hold_scale: float = 0.0
  spike_freeze_beta_scale: float = 0.5
  spike_freeze_gamma_scale: float = 1.5
  freeze_obj_rtol: float = 1e-4
  freeze_detected: bool = False
  ni_gate: str = "association"
  max_adaptations: int = 0
  rollback: bool = False
  rollback_w_scale: float = 0.5
  rollback_obj_tol: float = 0.1
  rollback_q_factor: float = 10.0
  rollback_max: int = 2
  rollback_gap: int = 5
  rollback_trigger: str = "event"
  w_init: float = 1.0
  budget_feasible_only: bool = False
  adapt_trigger: str = "forecast"
  trigger_rise: float = 0.3
  harm_threshold: float = 0.45
  harm_window: int = 5
  harm_prob: float = 0.0
  placebo_rate: float = 0.0
  placebo_seed: int = 0
  rollback_on_infeasible: bool = False
  budget_neutral: bool = False
  budget_floor: float = 0.25
  budget_min_iter: Optional[int] = None
  ai_spike_objective_jump: bool = False
  vk_offset: float = 0.0
  wk_offset: float = 1.0


@dataclass
class ADMM_data(coordinationData):
  beta: float = 1.3
  gamma: float = 0.5
  q: np.ndarray = None
  qold: np.ndarray = None
  phi: float = 1.0
  v: np.ndarray = None
  w: np.ndarray = None
  update_w: bool = False
  M_update_scheme: int = w_scheme.MEDIAN
  eps_qo: List = None
  save_q_in: bool = False
  save_q_in_out: bool = False
  eps_fo: List = None
  delta_q: List[float] = None



# COMPLETE: ADMM needs to be customized for this code
@dataclass
class ADMM(ADMM_data):
  " Alternating directions method of multipliers "
  # Constructor
  def __init__(self, nsp, beta, budget, index_of_master_SP, display, scaling, \
                mode, M_update_scheme, store_q_o=False, store_q_io=False, index = None, gamma=0.5, is_AI_regularizer=False,
                AI_spike_policy="L2_Normal", spike_influence_on_multipliers=True,
                spike_w_damp_scale=2.2, spike_w_amplify_scale=0.8,
                spike_gamma_scale=0.05, budget_scale=6.0, spike_threshold=0.5,
                spike_adaptive_scales=True, refit_every=20, detector_warm_start_from_history=False,
                ai_options: Optional[Dict] = None):
    global eps_fio, eps_qio
    """ Initialize the multiplier vectors """
    self.nsp = nsp
    self.beta = beta
    self.gamma = gamma
    self.budget = budget
    self.index_of_master_SP = index_of_master_SP
    self.display = display
    self.scaling = scaling
    self.mode = mode
    self.started: bool = False
    self.M_update_scheme = M_update_scheme
    self.objective_history = []
    self.eps_qo = []
    self.save_q_out = store_q_o
    self.save_q_in_out = store_q_io
    self.eps_fo = []
    self.index = index

    self.q = np.zeros([0,0])
    self.qold = np.zeros([0,0])
    self.phi: float = 1.0
    self.v = np.zeros([0,0])
    self.w = np.zeros([0,0])
    self.is_AI_regularizer = is_AI_regularizer
    self.AI_spike_policy = AI_spike_policy
    self.spike_confidence_level = AI_spike_policy
    self.spike_influence_on_multipliers = spike_influence_on_multipliers
    self.spike_w_damp_scale = spike_w_damp_scale
    self.spike_w_amplify_scale = spike_w_amplify_scale
    self.spike_gamma_scale = spike_gamma_scale
    self.budget_scale = budget_scale
    self.spike_adaptive_scales = bool(spike_adaptive_scales)
    self.spike_threshold = spike_threshold
    self.refit_every = int(refit_every)
    self.detector_warm_start_from_history = bool(detector_warm_start_from_history)
    # Detector/adaptation controls. Defaults reproduce the paper; ai_options overrides them (see preprocess.py).
    self.detector_window = 8
    self.detector_min_history = 5
    self.detector_options = {}
    self.detector_per_outer_iteration = True
    self.adapt_beta = True
    self.adapt_gamma = True
    self.adapt_budget = True
    self.adapt_cooldown = 0
    self.penalty_w_cap = np.inf
    self.w_stall_exponent = 2.0
    self.single_multiplier_update = False
    # "accelerate" (paper): every accepted spike speeds up penalty growth. "bidirectional": a spike that follows a
    # penalty increase (q_max rose right after w grew) holds w instead, scaled by spike_w_hold_scale.
    # "causal": as "bidirectional", plus a premature-freeze response (penalties grow, q_max falls, objective
    # stalls) that slows penalty growth (spike_freeze_beta_scale) and relaxes the stall test (spike_freeze_gamma_scale).
    self.spike_response = "accelerate"
    self.spike_w_hold_scale = 0.0
    self.spike_freeze_beta_scale = 0.5
    self.spike_freeze_gamma_scale = 1.5
    self.freeze_obj_rtol = 1e-4
    self.freeze_detected = False
    self._event_kind = ""
    # "association" (paper): non-improvement gate from the Beta-Binomial probabilities. "direct": objective worsened
    # or q_max stalled (see _direct_non_improvement).
    self.ni_gate = "association"
    self.max_adaptations = 0  # 0: unlimited
    # Rollback (MDO._rollback_step): on a detected spike with a worse objective, restart from the best earlier
    # design that was nearly as consistent, with penalty weights scaled by rollback_w_scale.
    self.rollback = False
    self.rollback_w_scale = 0.5
    self.rollback_obj_tol = 0.1
    self.rollback_q_factor = 10.0
    self.rollback_max = 2
    self.rollback_gap = 5
    # "event": an adaptation event (decisor + non-improvement gate) this iteration. "forecast": the decisor
    # accepted the forecast; the rollback's own objective test then plays the role of the non-improvement gate.
    self.rollback_trigger = "event"
    self.w_init = 1.0
    self.budget_feasible_only = False  # expand only subproblems whose last solve was feasible
    # Action trigger: "forecast" (decisor accepts the one-step forecast) or "threshold" (q_max rose by >= trigger_rise
    # decades). ni_gate "objective_drop" uses the harm gate (_harm_gate). Validated on recorded no-AI histories.
    self.adapt_trigger = "forecast"
    self.trigger_rise = 0.3
    self.harm_threshold = 0.45
    self.harm_window = 5
    self.harm_prob = 0.0
    self.placebo_rate = 0.0  # adapt_trigger "random": per-iteration probability of a (placebo) adaptation
    self.placebo_seed = 0
    self._tol_ref = 0.0
    self.rollback_on_infeasible = False  # roll back an infeasible design even without a detector alarm
    self._forecast_accepted = False
    self._n_adapt = 0
    self._qlog_hist = []
    # Budget-neutral expansion: the attributed subproblem's extra evaluations are taken from the others.
    self.budget_neutral = False
    self.budget_floor = 0.25
    self.budget_min_iter = None  # None: only after the first backbone re-fit (iter > refit_every)
    self._last_adapt_iter = -10**9
    self._w_grew_last = False
    self._qmax_prev = np.inf
    for key, value in (ai_options or {}).items():
      setattr(self, key, value)
    if is_AI_regularizer:
      resolve_level(AI_spike_policy)

  def __getstate__(self):
    # Remove or replace non-picklable attributes
    state = self.__dict__.copy()
    # Example: remove or replace RLock
    if '_lock' in state:
        del state['_lock']
    # Or replace with a placeholder
    # state['_lock'] = None
    return state

  def __setstate__(self, state):
      self.__dict__.update(state)
      # Reconstruct non-picklable objects if needed
      # self._lock = threading.RLock()

  def clone_point(self, p: variableData):
    self.var_group.append(p)

  def create_linking_list(self):
    a = []
    b = []
    for i in range(self.nsp):
      for j in range(self.nsp):
        if i != j:
          a.append(j)
          b.append(i)

    self._linker.append(a)
    self._linker.append(b)

  def get_target_values(self, vname: str, sp_index: int) -> List[float]:
    if self.save_q_in_out:
      global eps_qio
    targets = []
    if self.master_vars:
      for i in range(self._linker.shape[1]):
        if self.master_vars[self._linker[0, i]-1].sp_index == sp_index:
          if self.master_vars[self._linker[0, i]-1].name == vname:
            targets.append(self.master_vars[self._linker[1, i]-1].value) 
        elif self.master_vars[self._linker[1, i]-1].sp_index == sp_index:
          if self.master_vars[self._linker[1, i]-1].name == vname:
            targets.append(self.master_vars[self._linker[0, i]-1].value) 
    else:
        raise Exception(IOError, "Master variables vector have to be non-empty to calculate inconsistencies!")
    return targets
  
  def set_pair(self):
    indices1 = []
    indices2 = []
    sp_link =  []
    sp_link_to = []
    for i in range(len(self.master_vars)):
      check: bool = False
      if self.master_vars[i].link and isinstance(self.master_vars[i].link, list):
        nl = len(self.master_vars[i].link)
        check = any(self.master_vars[i].link >= np.ones(nl))
      elif self.master_vars[i].link:
        check = self.master_vars[i].link >= 1
      if (self.index == self.master_vars[i].sp_index \
          or self.master_vars[i].coupling_type != COUPLING_TYPE.UNCOUPLED) \
            and check and self.master_vars[i].index not in indices2:
        sp_link.append(self.master_vars[i].sp_index)
        linked_to = (self.master_vars[i].link)
        if linked_to and isinstance(linked_to, list):
          for linki in range(len(linked_to)):
            indices1.append(self.master_vars[i].index)
        else:
          indices1.append(self.master_vars[i].index)
        for j in range(len(self.master_vars)):
          check = False
          if linked_to and isinstance(linked_to, list):
            nl = len(linked_to)
            check = any(linked_to == np.multiply(self.master_vars[j].sp_index, np.ones(nl)))
          else:
            check = (linked_to == self.master_vars[j].sp_index)
          if check and (self.master_vars[j].name == self.master_vars[i].name):
            sp_link_to.append(self.master_vars[i].link)
            indices2.append(self.master_vars[j].index)
    
    # Remove redundant links
    self._linker = copy.deepcopy(np.array([indices1, indices2]))

  def build_average_matrix(self):
      """
      Build averaging matrix A such that y = A @ x returns
      averaged values for linked variables.

      Parameters
      ----------
      n_vars : int
          Number of variables
      linker : list[list]
          Two lists [src, dst] describing linked indices

      Returns
      -------
      A : np.ndarray
          Averaging matrix (n_vars x n_vars)
      """ 
      n_vars = len(self.master_vars)
      self.set_pair()
      src, dst = self._linker

      # ---- Union-Find structure ----
      parent = list(range(n_vars))

      def find(x):
          while parent[x] != x:
              parent[x] = parent[parent[x]]
              x = parent[x]
          return x

      def union(a, b):
          pa = find(a)
          pb = find(b)
          if pa != pb:
              parent[pb] = pa

      # ---- Apply links ----
      for a, b in zip(src, dst):
          union(a-1, b-1)

      # ---- Build groups manually ----
      groups = []
      roots = []

      for i in range(n_vars):
          r = find(i)

          if r not in roots:
              roots.append(r)
              groups.append([i])
          else:
              idx = roots.index(r)
              groups[idx].append(i)

      # ---- Build averaging matrix ----
      A = np.zeros((n_vars, n_vars))

      for group in groups:
          k = len(group)
          w = 1.0 / k

          for i in group:
              for j in group:
                  A[i, j] = w

      return A

  def get_target_variables(self, vname: str, sp_index: int) -> List[variableData]:
    if self.save_q_in_out:
      global eps_qio
    targets = []
    if self.master_vars:
      for i in range(self._linker.shape[1]):
        if self.master_vars[self._linker[0, i]-1].sp_index-1 == sp_index \
          and self.master_vars[self._linker[0, i]-1].coupling_type != COUPLING_TYPE.UNCOUPLED:
          if self.master_vars[self._linker[0, i]-1].name == vname:
            targets.append(self.master_vars[self._linker[1, i]-1]) 
    else:
        raise Exception(IOError, "Master variables vector have to be non-empty to calculate inconsistencies!")
    return targets
  
  def get_average_target_variables(self, vname: str, sp_index: int) ->List[variableData]:
    targets: List[variableData] = self.get_target_variables(vname, sp_index)
    current = 0
    for v in self.master_vars:
      if v.sp_index == sp_index and v.name == vname:
        current = v.value

    avg_t = (sum([x.value for x in targets]) + current) / (len(targets)+1)
    
    for i, t in enumerate(targets):
      targets[i].value = avg_t
      

    return targets
  
  def calculate_and_set_average_target_values(self, vname: str, sp_index: int):
    avg_matrix = self.build_average_matrix()
    m_values= []
    if self.master_vars:
      for v in self.master_vars:
        m_values.append(v.value)
      m_avg = avg_matrix @ m_values
      for i, v in enumerate(self.master_vars):
        self.master_vars[i].value = m_avg[i]

    # tgts = self.get_average_target_variables(vname=vname, sp_index=sp_index)
    # if self.save_q_in_out:
    #   global eps_qio
    # if self.master_vars:
    #   self.update_master_vector(tgts, None)
    else:
        raise Exception(IOError, "Master variables vector have to be non-empty to calculate inconsistencies!")
    self.calc_inconsistency()
    self.delta_q = np.multiply(-1, self.v)\
        /(np.multiply(2, np.square(self.w)))- self.q
    # a = 0

  def get_target_lb(self, vname: str, sp_index: int) -> List[float]:
    if self.save_q_in_out:
      global eps_qio
    targets = []
    if self.master_vars:
      for i in range(self._linker.shape[1]):
        if self.master_vars[self._linker[0, i]-1].sp_index == sp_index:
          if self.master_vars[self._linker[0, i]-1].name == vname:
            targets.append(self.master_vars[self._linker[1, i]-1].lb) 
        elif self.master_vars[self._linker[1, i]-1].sp_index == sp_index:
          if self.master_vars[self._linker[1, i]-1].name == vname:
            targets.append(self.master_vars[self._linker[0, i]-1].lb) 
    else:
        raise Exception(IOError, "Master variables vector have to be non-empty to calculate inconsistencies!")
    return targets
  
  def get_target_ub(self, vname: str, sp_index: int) -> List[float]:
    if self.save_q_in_out:
      global eps_qio
    targets = []
    if self.master_vars:
      for i in range(self._linker.shape[1]):
        if self.master_vars[self._linker[0, i]-1].sp_index == sp_index:
          if self.master_vars[self._linker[0, i]-1].name == vname:
            targets.append(self.master_vars[self._linker[1, i]-1].ub) 
        elif self.master_vars[self._linker[1, i]-1].sp_index == sp_index:
          if self.master_vars[self._linker[1, i]-1].name == vname:
            targets.append(self.master_vars[self._linker[0, i]-1].ub) 
    else:
        raise Exception(IOError, "Master variables vector have to be non-empty to calculate inconsistencies!")
    return targets

  def get_target_names(self, vname: str, sp_index: int) -> List[float]:
    if self.save_q_in_out:
      global eps_qio
    targets = []
    if self.master_vars:
      for i in range(self._linker.shape[1]):
        if self.master_vars[self._linker[0, i]-1].sp_index == sp_index:
          if self.master_vars[self._linker[0, i]-1].name == vname:
            targets.append(self.master_vars[self._linker[1, i]-1].name) 
        elif self.master_vars[self._linker[1, i]-1].sp_index == sp_index:
          if self.master_vars[self._linker[1, i]-1].name == vname:
            targets.append(self.master_vars[self._linker[0, i]-1].name) 
    else:
        raise Exception(IOError, "Master variables vector have to be non-empty to calculate inconsistencies!")
    return targets
  
  def get_target_SP_indices(self, vname: str, sp_index: int) -> List[float]:
    if self.save_q_in_out:
      global eps_qio
    targets = []
    if self.master_vars:
      for i in range(self._linker.shape[1]):
        if self.master_vars[self._linker[0, i]-1].sp_index == sp_index:
          if self.master_vars[self._linker[0, i]-1].name == vname:
            if self.master_vars[self._linker[0, i]-1].link not in targets:
              targets.append(self.master_vars[self._linker[0, i]-1].link) 
        elif self.master_vars[self._linker[1, i]-1].sp_index == sp_index:
          if self.master_vars[self._linker[1, i]-1].name == vname:
            if self.master_vars[self._linker[1, i]-1].link not in targets:
              targets.append(self.master_vars[self._linker[1, i]-1].link) 
    else:
        raise Exception(IOError, "Master variables vector have to be non-empty to calculate inconsistencies!")
    return targets
  
  def get_target_multipliers(self, vname: str, sp_index: int) -> List[float]:
    if self.save_q_in_out:
      global eps_qio
    targets_w = []
    targets_v = []
    counter = 0
    if self.master_vars:
      for i in range(self._linker.shape[1]):
        if self.master_vars[self._linker[0, i]-1].sp_index == sp_index:
          if self.master_vars[self._linker[0, i]-1].name == vname:
            targets_w.append(self.w[counter]) 
            targets_v.append(self.v[counter]) 
        elif self.master_vars[self._linker[1, i]-1].sp_index == sp_index:
          if self.master_vars[self._linker[1, i]-1].name == vname:
            targets_w.append(self.w[counter]) 
            targets_v.append(self.v[counter])
        counter += 1
    else:
        raise Exception(IOError, "Master variables vector have to be non-empty to calculate inconsistencies!")
    
    return targets_v, targets_w
  
  def get_coupling_type(self, vname: str, sp_index: int) -> List[float]:
    if self.save_q_in_out:
      global eps_qio
    targets = []
    if self.master_vars:
      for i in range(self._linker.shape[1]):
        if self.master_vars[self._linker[0, i]-1].sp_index == sp_index:
          if self.master_vars[self._linker[0, i]-1].name == vname:
            targets.append(self.master_vars[self._linker[0, i]-1].coupling_type.name) 
        elif self.master_vars[self._linker[1, i]-1].sp_index == sp_index:
          if self.master_vars[self._linker[1, i]-1].name == vname:
            targets.append(self.master_vars[self._linker[1, i]-1].coupling_type.name) 
    else:
        raise Exception(IOError, "Master variables vector have to be non-empty to calculate inconsistencies!")
    if len(targets) > 0:
      return targets
    return [COUPLING_TYPE.UNCOUPLED.name]

  
  def get_scaling_factor(self, vname: str, sp_index: int) -> List[float]:
    if self.save_q_in_out:
      global eps_qio
    targets = []
    if self.master_vars:
      for i in range(self._linker.shape[1]):
        if self.master_vars[self._linker[0, i]-1].sp_index == sp_index:
          if self.master_vars[self._linker[0, i]-1].name == vname:
            targets.append(self.master_vars[self._linker[0, i]-1].scaling) 
        elif self.master_vars[self._linker[1, i]-1].sp_index == sp_index:
          if self.master_vars[self._linker[1, i]-1].name == vname:
            targets.append(self.master_vars[self._linker[1, i]-1].scaling) 
    else:
        raise Exception(IOError, "Master variables vector have to be non-empty to calculate inconsistencies!")
    
    return targets
  
  def are_master_dims_consistent(self):
    for i in range(self._linker.shape[1]):
      dim1: int = self.master_vars[self._linker[0, i]-1].dim
      dim2: int = self.master_vars[self._linker[1, i]-1].dim
      name1: int = self.master_vars[self._linker[0, i]-1].name
      name2: int = self.master_vars[self._linker[1, i]-1].name
      if dim1 != dim2:
        raise IOError(f"Global master variables {name1} and {name2} should have the same dimensions!")
  
  def get_total_n_dimensions(self)->int:
    out:int = 0
    for i in range(self._linker.shape[1]):
      out+=self.master_vars[self._linker[0, i]-1].dim
    return out

  def batch_q(self, qin) -> list:
    """ This routine should be called after calculating the unbatched inconsistency vector """
    self.are_master_dims_consistent()
    ntot = self.get_total_n_dimensions()
    if ntot != len(qin):
      raise IOError("The size of variables inconsistency vector doesn't match the total number of variables dimension!")
    qout: List = []
    c = 0
    for i in range(self._linker.shape[1]):
      dim: int = self.master_vars[self._linker[0, i]-1].dim
      if dim >1:
        subvect = []
        for _ in range(dim):
          subvect.append(qin[c])
          c+=1
        qout.append(subvect)
      else:
        qout.append(qin[c])
        c+=1
    return qout
  
  def get_total_n_dimensions_old(self)->int:
    out:int = 0
    for i in range(self._linker.shape[1]):
      out+=self.master_vars_old[self._linker[0, i]-1].dim
    return out

  def batch_q_old(self, qin) -> list:
    """ This routine should be called after calculating the unbatched inconsistency vector """
    self.are_master_dims_consistent()
    ntot = self.get_total_n_dimensions_old()
    if ntot != len(qin):
      raise IOError("The size of variables inconsistency vector doesn't match the total number of variables dimension!")
    qout: List = []
    c = 0
    for i in range(self._linker.shape[1]):
      dim: int = self.master_vars_old[self._linker[0, i]-1].dim
      if dim >1:
        subvect = []
        for _ in range(dim):
          subvect.append(qin[c])
          c+=1
        qout.append(subvect)
      else:
        qout.append(qin[c])
        c+=1
    return qout
  

  def unbatch_q(self, qin):
    self.are_master_dims_consistent()
    nlinks = self._linker.shape[1]
    if nlinks != len(qin):
      raise IOError("The size of the introduced batched variables "
                    "inconsistency vector doesn't match the total number of links available!")
    qout = []
    for i in range(nlinks):
      dim: int = self.master_vars[self._linker[0, i]-1].dim
      if dim >1:
        subvect = qin[i]
        for j in range(len(subvect)):
          qout.append(subvect[j])
      else:
        qout.append(qin[i])

    return qout
  
  def unbatch_multipliers(self, win, vin):
    """ This routine should be called before calculating the unbatched multipliers vector vector """
    wout: list = []
    vout: list = []
    for i in range(self._linker.shape[1]):
      dim: int = self.master_vars[self._linker[0, i]-1].dim
      if dim >1:
        wsv = win[i]
        vsv = vin[i]
        for j in range(len(wsv)):
          wout.append(wsv[j])
          vout.append(vsv[j])
      else:
        wout.append(win[i])
        vout.append(vin[i])

    return wout, vout
  
  def batch_multipliers(self):
    """ This routine should be called before calculating the unbatched multipliers vector vector """
    wout: list = []
    vout: list = []
    c = 0
    for i in range(self._linker.shape[1]):
      dim: int = self.master_vars[self._linker[0, i]-1].dim
      if dim >1:
        wsv = []
        vsv = []
        for _ in range(dim):
          wsv.append(self.w[c])
          vsv.append(self.v[c])
          c+=1
        wout.append(wsv)
        vout.append(vsv)
      else:
        wout.append(self.w[c])
        vout.append(self.v[c])
        c+=1
    return wout, vout

  def modify_multipliers(self, qin, win, vin):
    for i in range(len(qin)):
      if isinstance(qin[i], list) and len(win[i]) != len(qin[i]):
        if len(qin[i]) > len(win[i]):
          for _ in range(len(qin[i])-len(win[i])):
            win[i].append(1.)
            vin[i].append(0.)
        elif len(win[i]) > len(qin[i]):
          win[i] = win[i][:len(qin[i])]
          vin[i] = vin[i][:len(qin[i])]
    self.w, self.v = self.unbatch_multipliers(win, vin)
  
  def update_all_cond_linked(self, index):
    name = self.master_vars[index].name
    dim = self.master_vars[index].dim
    value = self.master_vars[index].value

    for i in range(len(self.master_vars)):
      if index != i:
        namet = self.master_vars[i].name
      else:
        continue
      if namet == name:
        dt = self.master_vars[i].dim
        vt = self.master_vars[i].value
        tt = self.master_vars[i].type
        ubt = self.master_vars[i].ub
        lbt = self.master_vars[i].lb
        blt = self.master_vars[i].baseline
        st = self.master_vars[i].scaling
        sett = self.master_vars[i].set 
        if dim > dt:
          l = len(vt)  # noqa: E741
          dif = dim - dt
          vt = copy.deepcopy(value)
          tt += [tt[l-1]]*dif
          lbt += [lbt[l-1]]*dif
          ubt += [ubt[l-1]]*dif
          blt += [blt[l-1]]*dif
          st += [st[l-1]]*dif
          # TODO: Enhance the set logic
          if isinstance(sett, list):
            sett += [sett[l-1]]*dif
        elif dim < dt:
          l = len(value)  # noqa: E741
          dif = dt - dim
          vt = copy.deepcopy(value)
          tt = tt[:len(tt)-dif]
          lbt = lbt[:len(lbt)-dif]
          ubt = ubt[:len(ubt)-dif]
          st = st[:len(st)-dif]
          blt = blt[:len(blt)-dif]
        else:
          continue
          # TODO: Enhance the set logic
        dt = len(vt)
        self.master_vars[i].type = tt
        self.master_vars[i].value = vt
        self.master_vars[i].dim = dt
        self.master_vars[i].set = sett
        self.master_vars[i].baseline = blt
        self.master_vars[i].scaling = st
        self.master_vars[i].lb = lbt
        self.master_vars[i].ub = ubt

  def update_conditional_vars(self):
    for i in range(self._linker.shape[1]):
      dim1: int = self.master_vars[self._linker[0, i]-1].dim
      dim2: int = self.master_vars[self._linker[1, i]-1].dim

      # Check whether linked variables have same dimension
      if dim1 != dim2 and self.master_vars[self._linker[0, i]-1].cond_on is None:
          raise Exception(IOError, 
                          f'The variable {self.master_vars[self._linker[0, i]-1].name} '
                          f'linked between subproblems index #{self._linker[0, i]} and #{self._linker[1, i]}'
                          ' does not have the same dimension!')
      else:
        if self.master_vars[self._linker[0, i]-1].coupling_type == COUPLING_TYPE.FEEDFORWARD:
          self.update_all_cond_linked(self._linker[0, i]-1)
    for i in range(self._linker.shape[1]):
      dim1: int = self.master_vars[self._linker[0, i]-1].dim
      dim2: int = self.master_vars[self._linker[1, i]-1].dim

      # Check whether linked variables have same dimension
      if dim1 != dim2 and self.master_vars[self._linker[0, i]-1].cond_on is None:
          raise Exception(IOError, f'The variable {self.master_vars[self._linker[0, i]-1].name} '
                          f'linked between subproblems index #{self._linker[0, i]} and #{self._linker[1, i]} '
                          'does not have the same dimension!')
      else:
        if self.master_vars[self._linker[1, i]-1].coupling_type == COUPLING_TYPE.FEEDFORWARD:
          self.update_all_cond_linked(self._linker[1, i]-1)

  def calc_inconsistency(self):  # noqa: C901
    if self.save_q_in_out:
      global eps_qio
    q_temp : np.ndarray = np.zeros([0,0])
    tl0: List = []
    tl1: List = []
    self.update_conditional_vars()
    for i in range(self._linker.shape[1]):
      if self.master_vars:
        # TODO: Add a sanity check early on to ensure that linked parameters has the same type and linked to the same set
        # TODO: if they were of discrete type and may be add a dunder methed to handle all the necessary equality checks
        # TODO: Check if linked parameters have same type
        type1: Any = self.master_vars[self._linker[0, i]-1].type
        type2: Any = self.master_vars[self._linker[1, i]-1].type
        val1: Any = self.master_vars[self._linker[0, i]-1].value
        val2: Any = self.master_vars[self._linker[1, i]-1].value
        dim1: int = self.master_vars[self._linker[0, i]-1].dim

        #COMPLETED: The variables coupling relationships and their size dependencies need a review
        for ik in range(dim1):
          t1 = type1[ik] if isinstance(type1, list) else type1
          t2 = type2[ik] if isinstance(type2, list) else type2
          tl0.append(self._linker[0, i])
          tl1.append(self._linker[1, i])
          if t1[0] != t2[0]:
            raise Exception(IOError, f'The variable {self.master_vars[self._linker[0, i]-1].name}'
                            f' linked between subproblems index #{self._linker[0, i]} '
                            f'and #{self._linker[1, i]} does not have the same type(s)!')
          if t1[0].lower() != "r" and t1[0].lower() != "i" and t1[0].lower() != "d" and t1[0].lower() != "c":
            raise Exception(IOError, f'The variable {self.master_vars[self._linker[0, i]-1].name} '
                            f'linked between subproblems index #{self._linker[0, i]} and #{self._linker[1, i]}'
                            ' has unknown type(s)!')
          if t1[0].lower() == "c":
            v1 = val1[ik] if isinstance(val1, list) else val1
            v2 = val2[ik] if isinstance(val2, list) else val2
            # if isinstance(val1, list) and isinstance(val2, list):
            #   q_temp = np.append(q_temp, sum([not x for x in np.equal(val1,val2)]))
            # else:
            q_temp = np.append(q_temp, 0 if v1 == v2 else 1)
            continue
          else:
            v1 = val1[ik] if isinstance(val1, list) else val1
            v2 = val2[ik] if isinstance(val2, list) else val2
          if t1[0].lower() == "d":
            i1 = int(v1)#self.sets[sn1].index(v1)
            i2 = int(v1)#self.sets[sn2].index(v2)
          if  (isinstance(self.scaling, list) and len(self.scaling) == len(self.master_vars)):
            qscale = np.multiply(np.add(self.scaling[self._linker[0, i]-1], self.scaling[self._linker[1, i]-1]), 0.5)
            if t1[0].lower() == "d":
              q_temp = np.append(q_temp, np.multiply(np.subtract(i1, i2), qscale))
            else:
              q_temp = np.append(q_temp, np.multiply(np.subtract(v1, v2), qscale))
          elif isinstance(self.scaling, float) or  isinstance(self.scaling, int):
            if t1[0].lower() == "d":
              q_temp = np.append(q_temp, np.multiply(np.subtract(i1, i2), self.scaling))
            else:
              q_temp = np.append(q_temp, np.multiply(np.subtract(v1, v2), self.scaling))
          else:
            if t1[0].lower() == "d":
              q_temp = np.append(q_temp, np.multiply(np.subtract(i1, i2), min(self.scaling)))
            else:
              q_temp = np.append(q_temp, np.multiply(np.subtract(v1, v2), min(self.scaling)))
            warning("The inconsistency scaling factors are defined in a list which has a different "
                    "size from the master variables vector! The minimum value of the provided scaling "
                    "list will be used to scale the inconsistency vector.")
        
          # if  (isinstance(self.scaling, list) and len(self.scaling) == len(self.master_vars)):
          #   qscale = np.multiply(np.add(self.scaling[self._linker[0, i]-1], self.scaling[self._linker[1, i]-1]), 0.5)
          #   q_temp = np.append(q_temp, np.multiply(subtract(((self.master_vars[self._linker[0, i]-1].value)),
          #           ((self.master_vars[self._linker[1, i]-1].value))), qscale))
          # elif isinstance(self.scaling, float) or  isinstance(self.scaling, int):
          #   q_temp = np.append(q_temp, np.multiply(subtract(((self.master_vars[self._linker[0, i]-1].value)),
          #           ((self.master_vars[self._linker[1, i]-1].value))), self.scaling))
          # else:
          #   q_temp = np.append(q_temp, np.multiply(subtract(((self.master_vars[self._linker[0, i]-1].value)),
          #           ((self.master_vars[self._linker[1, i]-1].value))), min(self.scaling)))
          #   warning("The inconsistency scaling factors are defined in a list which has a different size 
          # from the master variables vector! The minimum value of the provided 
          # scaling list will be used to scale the inconsistency vector.")
      else:
        raise Exception(IOError, "Master variables vector have to be non-empty to calculate inconsistencies!")
    # qb: list = self.batch_q(q_temp)
    # self.modify_multipliers(qb, wb, vb)
    # correction = np.array([0 for _ in range(len(q_temp))]) if self.delta_q is None else self.delta_q
    self.q = copy.deepcopy(q_temp)# + correction
    self.extended_linker = copy.deepcopy(np.array([tl0, tl1]))
    if self.save_q_out:
      self.eps_qo.append(np.max([abs(x) for x in q_temp]))
    if self.save_q_in_out:
      eps_qio.append(np.max([abs(x) for x in q_temp]))

  #COMPLETED: This should be consistent with the calc_inc
  def calc_inconsistency_old(self):
    if self.save_q_in_out:
      global eps_qio
    q_temp : np.ndarray = np.zeros([0,0])
    tl0: List = []
    tl1: List = []
    for i in range(self._linker.shape[1]):
      if self.master_vars_old:
        # TODO: Add a sanity check early on to ensure that linked parameters has the same 
        # TODO: type and linked to the same set if they were of discrete type and may be add 
        # TODO: a dunder methed to handle all the necessary equality checks
        #  TODO: Check if linked parameters have same type
        type1: Any = self.master_vars_old[self._linker[0, i]-1].type
        type2: Any = self.master_vars_old[self._linker[1, i]-1].type
        val1: Any = self.master_vars_old[self._linker[0, i]-1].value
        val2: Any = self.master_vars_old[self._linker[1, i]-1].value
        dim1: int = self.master_vars_old[self._linker[0, i]-1].dim
        set1_name: Any = self.master_vars_old[self._linker[1, i]-1].set
        set2_name: Any = self.master_vars_old[self._linker[1, i]-1].set


        
        # Check whether linked variables have same dimension
        for ik in range(dim1):
          t1 = type1[ik] if isinstance(type1, list) else type1
          t2 = type2[ik] if isinstance(type2, list) else type2
          sn1 = set1_name[ik] if isinstance(set1_name, list) else set1_name
          sn2 = set2_name[ik] if isinstance(set2_name, list) else set2_name
          tl0.append(self._linker[0, i])
          tl1.append(self._linker[1, i])
          if t1 != t2:
            raise Exception(IOError, f'The variable {self.master_vars_old[self._linker[0, i]-1].name} linked '
                            f'between subproblems index #{self._linker[0, i]} and #{self._linker[1, i]} '
                            'does not have the same type(s)!')
          if t1[0].lower() != "r" and t1[0].lower() != "i" and t1[0].lower() != "d" and t1[0].lower() != "c":
            raise Exception(IOError, f'The variable {self.master_vars_old[self._linker[0, i]-1].name} linked '
                            f'between subproblems index #{self._linker[0, i]} and #{self._linker[1, i]} has unknown type(s)!')
          if t1[0].lower() == "c":
            v1 = val1[ik] if isinstance(val1, list) else val1
            v2 = val2[ik] if isinstance(val2, list) else val2
            if v1 == v2:
                q_temp = np.append(q_temp, 0.)
            else:
              q_temp = np.append(q_temp, 1.)
            continue
          else:
            v1 = val1[ik] if isinstance(val1, list) else val1
            v2 = val2[ik] if isinstance(val2, list) else val2
          if t1[0].lower() == "d":
            i1 = self.sets[sn1].index(v1)
            i2 = self.sets[sn2].index(v2)
          if  (isinstance(self.scaling, list) and len(self.scaling) == len(self.master_vars_old)):
            qscale = np.multiply(np.add(self.scaling[self._linker[0, i]-1], self.scaling[self._linker[1, i]-1]), 0.5)
            if t1[0].lower() == "d":
              q_temp = np.append(q_temp, np.multiply(np.subtract(i1, i2), qscale))
            else:
              q_temp = np.append(q_temp, np.multiply(np.subtract(v1, v2), qscale))
          elif isinstance(self.scaling, float) or  isinstance(self.scaling, int):
            if t1[0].lower() == "d":
              q_temp = np.append(q_temp, np.multiply(np.subtract(i1, i2), self.scaling))
            else:
              q_temp = np.append(q_temp, np.multiply(np.subtract(v1, v2), self.scaling))
          else:
            if t1[0].lower() == "d":
              q_temp = np.append(q_temp, np.multiply(np.subtract(i1, i2), min(self.scaling)))
            else:
              q_temp = np.append(q_temp, np.multiply(np.subtract(v1, v2), min(self.scaling)))
            warning("The inconsistency scaling factors are defined in a list which has a different'" \
            "' size from the master variables vector! The minimum value of the provided scaling list will be used to "
            "scale the inconsistency vector.")
      else:
        raise Exception(IOError, "Master variables vector have to be non-empty to calculate inconsistencies!")
    self.qold = copy.deepcopy(q_temp)

  def update_master_vector_val(self, vars: List[variableData], resps: List[variableData], sets):
    if self.master_vars:
      for i in range(len(vars)):
        self.master_vars[vars[i].index-1] = copy.deepcopy(vars[i])
        # self.master_vars[vars[i].index-1].value = vars[i].value 
        # #if (typ != "c" and typ != "d") else sets[vars[i].set].index(vars[i].value)
      if resps is not None:
        for i in range(len(resps)):
          self.master_vars[resps[i].index-1] = copy.deepcopy(resps[i])
          # self.master_vars[resps[i].index-1].value = resps[i].value 
          # #if (typ != "c" and typ != "d") else sets[resps[i].set].index(resps[i].value)
    else:
      raise Exception(IOError, "Master variables vector have to be non-empty to calculate inconsistencies!")

  def update_master_vector(self, vars: List[variableData], resps: List[variableData]):
    if self.master_vars:
      for i in range(len(vars)):
        self.master_vars[vars[i].index-1] = copy.deepcopy(vars[i])
      if resps is not None:
        for i in range(len(resps)):
          self.master_vars[resps[i].index-1] = copy.deepcopy(resps[i])
    else:
      raise Exception(IOError, "Master variables vector have to be non-empty to calculate inconsistencies!")


  def calc_penalty(self, q_indices, fobj: float = -np.inf):
    if len(self.w) != len(self.v):
      raise RuntimeError('The multipliers vectors w and v have inconsistent size!')
    if len(self.q) != len(self.w):
      raise RuntimeError("The variables inconsistency vector has different size from the multipliers vectoe w and v!")
    correction = np.array([0 for _ in range(len(self.q))]) if self.delta_q is None else self.delta_q
    q_corrected = self.q + correction
    phi_obj_guard = 0.0
    # if getattr(self, "ai_spike_objective_jump", False):
    #   incumbent_offset = np.clip(fobj - min(self.objective_history), 0.0, np.inf)
    #   phi_obj_guard = self.vk_offset * incumbent_offset + self.wk_offset*self.wk_offset*incumbent_offset*incumbent_offset
    phi = np.add(np.multiply(self.v, q_corrected), np.multiply(np.multiply(self.w, self.w), 
                                                               np.multiply(q_corrected, q_corrected)))
    #COMPLETE: Sum relevant components of q to accelerate the convergence of variables consistency
    s = 0.0
    for i in q_indices:
      if isinstance(i, list):
        s += sum(phi[i])
      else:
        s += phi[i]
    self.phi = s + phi_obj_guard

    if np.iscomplex(self.phi) or np.isnan(self.phi):
      self.phi = np.inf
  
  def sigmoid(self, x: np.ndarray):
    """
    Compute the sigmoid of x

    Parameters
    ----------
    x : array_like
        A scalar or numpy array of any size.

    Returns
    -------
     s : array_like
         sigmoid(x)
    """
    x = np.clip( x, -500, 500 )           # protect against overflow
    s = 1.0/(1.0+np.exp(-x))

    return s
  
  def tanh(self, x: np.ndarray):
    """
    Compute the sigmoid of x

    Parameters
    ----------
    x : array_like
        A scalar or numpy array of any size.

    Returns
    -------
     s : array_like
         tanh(x)
    """
    x = np.clip( x, -500, 500 )           # protect against overflow
    s = np.arctan(x)

    return s

  def _get_expected_spike_event(self, iter: int) -> tuple[bool, int, int]:
    """
    Return (is_expected_spike, spike_subproblem, spike_future_iteration).

    A spike is considered expected when forecast confidence is accepted by the
    decisor and associated with a non-improvement condition.
    """
    if not getattr(self, "is_AI_regularizer", False):
      return False, 0, -1

    detector = getattr(self, "ai_regularizer", None)
    if detector is None:
      return False, 0, -1

    level_name = str(getattr(self, "AI_spike_policy", getattr(self, "spike_confidence_level", "L2_Normal")))
    decision = detector.decide(level_name=level_name)
    # Kept for the rollback trigger "forecast", which does not use the non-improvement gate.
    self._forecast_accepted = bool(decision.accept_forecast_prob)

    spike_threshold = float(getattr(self, "spike_threshold", 0.5))
    spike_location = int(getattr(self, "spike_location", 0))
    non_improvement_prob = float(getattr(self, "non_improvement_prob", 0.0))
    objective_non_improvement_prob = float(getattr(self, "objective_non_improvement_prob", 0.0))
    constraint_non_improvement_prob = float(getattr(self, "constraint_non_improvement_prob", 0.0))

    non_improvement_gate = max(
      non_improvement_prob,
      objective_non_improvement_prob,
      constraint_non_improvement_prob,
    ) >= spike_threshold
    if str(getattr(self, "ni_gate", "association")) == "direct":
      # The association probabilities hover near 0.5 on some problems (GP), so read non-improvement directly.
      non_improvement_gate = self._direct_non_improvement()

    if str(getattr(self, "ni_gate", "association")) == "objective_drop":
      non_improvement_gate = self._harm_gate()
    elif str(getattr(self, "ni_gate", "association")) == "none":
      non_improvement_gate = True  # act on every triggered event (no usefulness gate)
    if str(getattr(self, "adapt_trigger", "forecast")) == "random":
      # Placebo control: act at random iterations (rate placebo_rate), no detector decision and no gate. The draw
      # depends only on (placebo_seed, iteration), so repeated calls within an iteration agree.
      u = np.random.default_rng([int(getattr(self, "placebo_seed", 0)), int(iter)]).random()
      is_expected_spike = bool(u < float(getattr(self, "placebo_rate", 0.0)))
    elif str(getattr(self, "adapt_trigger", "forecast")) == "threshold":
      # Act on a confirmed spike (q_max rose by >= trigger_rise decades) rather than on the one-step forecast,
      # which validation showed to be late and noisy.
      is_expected_spike = self._threshold_spike() and non_improvement_gate
    else:
      is_expected_spike = (
        decision.accept_forecast_prob
        and non_improvement_gate
      )

    # Both the decisor and the non-improvement gate must pass (paper Eq. ni_gate).
    if not is_expected_spike:
      return False, 0, -1

    if spike_location <= 0:
      spike_location = int(getattr(self, "last_spike_subproblem", 0))

    # forecast_prob is a one-step-ahead signal, so expected spike timing is t+1.
    spike_future_iteration = int(iter) + 1
    self.last_spike_subproblem = spike_location
    self.last_spike_iteration = spike_future_iteration
    self.spike_detected = True
    return True, spike_location, spike_future_iteration

  def _compute_ai_scales(self, iter: int) -> tuple[float, float, float, bool]:
    """
    Return (beta_scale, gamma_scale, budget_scale, spike) for this outer iteration.

    Adaptation is applied only when the decisor accepts the forecast AND the non-improvement
    gate holds. In adaptive mode each configured value is an upper bound reached at full
    intensity. Nothing is committed here, so the logger can call it without side effects.

    The beta rule is keyed on the objective: if it just decreased, spike_w_damp_scale (> 1 in the
    benchmark configs) accelerates penalty growth on stalled couplings; otherwise
    spike_w_amplify_scale (about 1) leaves it nearly unchanged.
    """
    neutral = (1.0, 1.0, 1.0)
    if not (getattr(self, "is_AI_regularizer", False) and getattr(self, "spike_influence_on_multipliers", False)):
      return (*neutral, False)

    is_expected_spike, _, _ = self._get_expected_spike_event(iter)
    max_adapt = int(getattr(self, "max_adaptations", 0))
    if max_adapt > 0 and int(getattr(self, "_n_adapt", 0)) >= max_adapt:
      self._event_kind = "spike" if is_expected_spike else ""
      return (*neutral, is_expected_spike)
    freeze = (not is_expected_spike) and self._premature_freeze(iter)
    self._event_kind = "spike" if is_expected_spike else ("freeze" if freeze else "")
    if not (is_expected_spike or freeze):
      return (*neutral, False)

    cooldown = int(getattr(self, "adapt_cooldown", 0))
    if cooldown > 0 and int(iter) - int(getattr(self, "_last_adapt_iter", -10**9)) <= cooldown:
      return (*neutral, is_expected_spike)

    if freeze:
      # Consistency is being bought at the expense of the objective: slow penalty growth and relax the stall test,
      # and move evaluations to the subproblem that was still improving (see SP.solve).
      beta_scale = float(getattr(self, "spike_freeze_beta_scale", 0.5)) if getattr(self, "adapt_beta", True) else 1.0
      gamma_scale = float(getattr(self, "spike_freeze_gamma_scale", 1.5)) if getattr(self, "adapt_gamma", True) else 1.0
      budget_scale = float(getattr(self, "budget_scale", 6.0)) if getattr(self, "adapt_budget", True) else 1.0
      return beta_scale, gamma_scale, budget_scale, False

    obj_hist = getattr(self, "objective_history", [])
    obj_decreased = (
      len(obj_hist) >= 2
      and np.isfinite(obj_hist[-1])
      and np.isfinite(obj_hist[-2])
      and obj_hist[-1] < obj_hist[-2]
    )
    if obj_decreased:
      configured_beta_scale = float(getattr(self, "spike_w_damp_scale", 1.8))
    else:
      configured_beta_scale = float(getattr(self, "spike_w_amplify_scale", 1.05))
    configured_gamma_scale = float(getattr(self, "spike_gamma_scale", 0.05))
    configured_budget_scale = float(getattr(self, "budget_scale", 6.0))

    if not getattr(self, "spike_adaptive_scales", True):
      beta_scale, gamma_scale, budget_scale = configured_beta_scale, configured_gamma_scale, configured_budget_scale
      if self._penalty_caused_spike():
        beta_scale, gamma_scale = float(getattr(self, "spike_w_hold_scale", 0.0)), 1.0
    else:
      forecast_prob = float(getattr(self, "forecast_prob", 1.0))
      non_improvement_prob = float(getattr(self, "non_improvement_prob", 1.0))
      intensity = float(np.clip(0.5 * forecast_prob + 0.5 * non_improvement_prob, 0.0, 1.0))
      budget_intensity = float(np.clip(1.5 * non_improvement_prob, 0.0, 1.0))
      beta_scale = 1.0 + (configured_beta_scale - 1.0) * intensity
      # gamma_scale tightens the stall threshold; the configured value is the tightest extreme.
      gamma_scale = 1.0 - (1.0 - configured_gamma_scale) * intensity
      budget_scale = 1.0 + (configured_budget_scale - 1.0) * budget_intensity
      if self._penalty_caused_spike():
        # Accelerating the penalty would feed the spike: hold w (toward spike_w_hold_scale) and keep gamma.
        hold = float(getattr(self, "spike_w_hold_scale", 0.0))
        beta_scale = 1.0 - (1.0 - hold) * intensity
        gamma_scale = 1.0

    if not getattr(self, "adapt_beta", True):
      beta_scale = 1.0
    if not getattr(self, "adapt_gamma", True):
      gamma_scale = 1.0
    if not getattr(self, "adapt_budget", True):
      budget_scale = 1.0
    return beta_scale, gamma_scale, budget_scale, True

  def _threshold_spike(self) -> bool:
    """q_max rose by at least trigger_rise decades at this iteration, from above twice the tolerance."""
    hist = list(getattr(self, "_qlog_hist", []))
    if not hist or self.q is None or not np.size(self.q):
      return False
    tol = float(getattr(self, "_tol_ref", 0.0))
    q_now = max(float(np.max(np.abs(self.q))), 1e-300)
    q_prev = 10.0 ** hist[-1]
    if q_prev < 2.0 * tol:
      return False
    return bool(np.log10(q_now) - hist[-1] >= float(getattr(self, "trigger_rise", 0.3)))

  def _harm_gate(self) -> bool:
    """Act when the objective dropped together with the spike, relative to its recent noise.

    p_harm = Phi(-dS / s), dS = slog f_k - slog f_(k-1), s = median |dS| over the previous harm_window changes (floor
    1e-6). Validation: an objective drop at a spike predicts that the objective will be worse a few iterations later.
    """
    obj = np.asarray(getattr(self, "objective_history", []), dtype=float)
    if obj.size < 3 or not np.all(np.isfinite(obj[-2:])):
      return False
    so = np.sign(obj) * np.log1p(np.abs(obj))
    dso = np.diff(so)
    window = int(getattr(self, "harm_window", 5))
    prev = dso[-window - 1:-1]
    prev = prev[np.isfinite(prev)]
    scale = max(float(np.median(np.abs(prev))) if prev.size else 0.0, 1e-6)
    p_harm = 0.5 * (1.0 + math.erf((-dso[-1] / scale) / math.sqrt(2.0)))
    self.harm_prob = float(p_harm)
    return p_harm >= float(getattr(self, "harm_threshold", 0.45))

  def _direct_non_improvement(self) -> bool:
    """Objective worsened this iteration, or log q_max has been flat (span < 0.05) over the last four iterations."""
    obj_hist = getattr(self, "objective_history", [])
    worsened = (
      len(obj_hist) >= 2 and np.isfinite(obj_hist[-1]) and np.isfinite(obj_hist[-2])
      and float(obj_hist[-1]) > float(obj_hist[-2]) + 1e-4 * max(1.0, abs(float(obj_hist[-2])))
    )
    qlog = list(getattr(self, "_qlog_hist", []))
    if self.q is not None and np.size(self.q):
      qlog.append(float(np.log10(max(float(np.max(np.abs(self.q))), 1e-30))))
    stalled = len(qlog) >= 4 and (max(qlog[-4:]) - min(qlog[-4:])) < 0.05
    return bool(worsened or stalled)

  def _premature_freeze(self, iter: int) -> bool:
    """Causal response only: penalties grew last iteration and q_max fell, yet the objective did not improve.

    That is the signature of the coordination converging on consistency before optimality, which the paper's
    accelerating actions make worse.
    """
    if str(getattr(self, "spike_response", "accelerate")) != "causal":
      return False
    if int(iter) < int(getattr(self, "detector_min_history", 5)):
      return False
    if not getattr(self, "_w_grew_last", False) or self.q is None or not np.size(self.q):
      return False
    if float(np.max(np.abs(self.q))) >= float(getattr(self, "_qmax_prev", np.inf)):
      return False
    obj_hist = getattr(self, "objective_history", [])
    if len(obj_hist) < 2 or not (np.isfinite(obj_hist[-1]) and np.isfinite(obj_hist[-2])):
      return False
    tol = float(getattr(self, "freeze_obj_rtol", 1e-4)) * max(1.0, abs(float(obj_hist[-2])))
    return float(obj_hist[-1]) >= float(obj_hist[-2]) - tol

  def _penalty_caused_spike(self) -> bool:
    """True when the bidirectional or causal response is on and q_max rose right after the last penalty increase."""
    if str(getattr(self, "spike_response", "accelerate")) not in ("bidirectional", "causal"):
      return False
    if not getattr(self, "_w_grew_last", False) or self.q is None or not np.size(self.q):
      return False
    return float(np.max(np.abs(self.q))) > float(getattr(self, "_qmax_prev", np.inf))

  def _get_ai_spike_multiplier_scales(self, iter: int) -> tuple[float, float, bool]:
    """Return (beta_scale, gamma_scale, spike) and commit the budget scale and cool-down state."""
    beta_scale, gamma_scale, budget_scale, spike = self._compute_ai_scales(iter)
    if spike:
      self.spike_detected = True
    if getattr(self, "_event_kind", "") == "freeze" and (beta_scale != 1.0 or gamma_scale != 1.0 or budget_scale != 1.0):
      self.freeze_detected = True
    # The subproblem reads this on the next outer iteration (one-step-ahead adaptation).
    self._effective_budget_scale = budget_scale
    adapted = beta_scale != 1.0 or gamma_scale != 1.0 or budget_scale != 1.0
    if adapted:
      self._last_adapt_iter = int(iter)
      self._n_adapt = int(getattr(self, "_n_adapt", 0)) + 1
    kind = getattr(self, "_event_kind", "")
    if adapted and kind == "spike" and self._penalty_caused_spike():
      kind = "hold"
    self._applied_scales = (beta_scale, gamma_scale, budget_scale, kind if adapted else "")
    return beta_scale, gamma_scale, spike

  def _build_increase_w(self, update_scheme):
    kappa = float(getattr(self, "w_stall_exponent", 2.0))
    if update_scheme == w_scheme.MEDIAN:
      return [
        kappa * (
          self.q_stall[i]
          and np.greater_equal(np.abs(self.q[i]), np.median(np.abs(self.q)))
        )
        for i in range(self.q_stall.shape[0])
      ]
    
    if update_scheme == w_scheme.MEAN:
      return [
        kappa * (
          self.q_stall[i]
          and np.greater_equal(np.abs(self.q[i]), np.mean(np.abs(self.q)))
        )
        for i in range(self.q_stall.shape[0])
      ]

    if update_scheme == w_scheme.NORM_2:
      return [
        kappa * (
          self.q_stall[i]
          and np.greater_equal(np.abs(self.q[i]), np.linalg.norm(np.abs(self.q), 2))
        )
        for i in range(self.q_stall.shape[0])
      ]

    if update_scheme == w_scheme.MAX:
      tc = np.greater_equal(np.abs(self.q), np.max(np.abs(self.q)))
      return [
        len(self.q) * (self.q_stall[i] and tc[i])
        for i in range(self.q_stall.shape[0])
      ]

    if update_scheme == w_scheme.NORMAL:
      return self.q_stall

    if update_scheme == w_scheme.RANK:
      temp = np.argsort(self.q)
      rank = np.empty_like(temp)
      rank[temp] = np.arange(len(self.q))
      return np.multiply(np.multiply(2, self.q_stall), np.divide(rank, np.max(rank)))

    if update_scheme == w_scheme.NO_UPDATE:
      return [1.0] * len(self.w)

    raise Exception(IOError, "Multipliers update scheme is not recognized!")
  
  def calculate_multipliers_weights_main_objective(self, iter):
    incumbent_offset = max(0, self.objective_history[-1] - min(self.objective_history))
    slack = max(0, incumbent_offset)
    incumbent_offset_old = max(0, self.objective_history[-2] - min(self.objective_history))
    
    self.vk_offset = self.vk_offset + 2*self.wk_offset*self.wk_offset*(incumbent_offset+slack)
    inc_offset_stall = np.greater(np.abs(incumbent_offset), self.gamma * np.abs(incumbent_offset_old))
    if inc_offset_stall:
      self.wk_offset = self.wk_offset * self.beta

  def update_multipliers(self, iter):
    spike_detected = False
    # Apply AI spike policy from helper utility.
    beta_scale, gamma_scale, spike_detected = self._get_ai_spike_multiplier_scales(iter)
    self.v = np.add(self.v, np.multiply(
            np.multiply(np.multiply(2, self.w), self.w), self.q))
    self.calc_inconsistency_old()
    old = self.batch_q_old(self.qold)
    current = self.batch_q(self.q)
    for i in range(len(current)):
      if isinstance(current[i], list):
        if len(current[i]) > len(old[i]):
          old[i] += [0]*(len(current[i])-len(old[i]))
        elif len(current[i]) < len(old[i]):
          current[i] += [0]*(len(old[i])-len(current[i]))
        else:
          continue
      else:
        continue
    
    self.q = self.unbatch_q(current)
    self.qold = self.unbatch_q(old)

    
    
    gamma_scaled = self.gamma * gamma_scale
    if gamma_scale > 1.0:
      # A relaxed stall test must still require some progress.
      gamma_scaled = min(gamma_scaled, 0.95)
    self.q_stall = np.greater(np.abs(self.q), gamma_scaled * np.abs(self.qold))
    increase_w = []
    wold = copy.deepcopy(self.w)
    update_scheme = self.M_update_scheme

    # if spike_detected:
    #   update_scheme = w_scheme.NORM_2 # Use MAX scheme for spike events to accelerate convergence.

    increase_w = self._build_increase_w(update_scheme)
    # if spike_detected:
    #   update_scheme = w_scheme.MAX  # Use MAX scheme for spike events to accelerate convergence.
    #   return
    # if iter > 1:
    #   self.calculate_multipliers_weights_main_objective(iter=iter)

    

    w_cap = float(getattr(self, "penalty_w_cap", np.inf))
    for i in range(len(self.w)):
      beta_exponent = np.multiply(increase_w[i], beta_scale)
      self.w[i] = copy.deepcopy(np.multiply(self.w[i], self.beta**beta_exponent))
      if np.isfinite(w_cap):
        self.w[i] = np.minimum(self.w[i], w_cap)
    # State for the bidirectional spike response at the next outer iteration.
    self._w_grew_last = bool(np.any(np.greater(np.asarray(self.w, dtype=float), np.asarray(wold, dtype=float))))
    self._qmax_prev = float(np.max(np.abs(self.q))) if np.size(self.q) else np.inf
    if np.size(self.q):
      self._qlog_hist = (list(getattr(self, "_qlog_hist", [])) + [float(np.log10(max(self._qmax_prev, 1e-30)))])[-20:]

    if iter > 0 and False:
      dq = (np.abs(self.q) - np.abs(self.qold))
      # Forget gate
      f =self.sigmoid(dq+np.abs(np.subtract(self.w,wold)))
      # Input gate for the next iteration
      for i in range(len(self.w)):
        self.w[i] *= f[i]
        self.w[i] += np.tanh(dq[i]+np.abs(np.subtract(self.w[i],wold[i])))*f[i] 
