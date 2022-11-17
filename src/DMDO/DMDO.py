# ------------------------------------------------------------------------------------#
#  Distributed Multidisciplinary Design Optimization - DMDO                           #
#                                                                                     #
#  Author: Ahmed H. Bayoumy                                                           #
#  email: ahmed.bayoumy@mail.mcgill.ca                                                #
#                                                                                     #
#  This program is free software: you can redistribute it and/or modify it under the  #
#  terms of the GNU Lesser General Public License as published by the Free Software   #
#  Foundation, either version 3 of the License, or (at your option) any later         #
#  version.                                                                           #
#                                                                                     #
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY    #
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A    #
#  PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.   #
#                                                                                     #
#  You should have received a copy of the GNU Lesser General Public License along     #
#  with this program. If not, see <http://www.gnu.org/licenses/>.                     #
#                                                                                     #
#  You can find information on simple_mads at                                         #
#  https://github.com/Ahmed-Bayoumy/DMDO                                              #
# ------------------------------------------------------------------------------------#

import copy
from genericpath import isfile
import json
from logging import warning
import logging
from multiprocessing.dummy import Process
from multiprocessing.sharedctypes import Value

from dataclasses import dataclass, field
import os
import platform
import subprocess
import sys
from typing import List, Dict, Any, Callable, Protocol, Optional

import numpy as np
from numpy import cos, exp, pi, prod, sin, sqrt, subtract, inf
import math

import OMADS
from enum import Enum, auto
from scipy.optimize import minimize, Bounds
import yaml


@dataclass
class BMMDO:
  """ Simple testing problems """

@dataclass
class USER:
  """
  Custom class to introduce user data
  """
# Global instant of user data
user =USER
# Global lists to store the evolution of inconsistency and the discripency from best known solution for each iteration within the nested inner-outer loop
eps_qio = []
eps_fio = []

@dataclass
class double_precision:
    decimals: int
    """ Decimal precision control """

    def truncate(self, value: float) -> float:
        if abs(value) != inf:
            multiplier = 10 ** self.decimals
            return int(value * multiplier) / multiplier if value != inf else inf
        else:
            return value

class VAR_TYPE(Enum):
  CONTINUOUS = auto()
  INTEGER = auto()
  BINARY = auto()
  CATEGORICAL = auto()
  ORDINAL = auto()

class VALIDATOR(Enum):
  PRE = auto()
  RUNNING = auto()
  POST = auto()

class BARRIER_TYPE(Enum):
  EXTREME = auto()
  PROGRESSIVE = auto()
  FILTER = auto()

class PSIZE_UPDATE(Enum):
  DEFAULT = auto()
  SUCCESS = auto()
  MAX = auto()
  LAST = auto()

class w_scheme(Enum):
  MEDIAN = auto()
  MAX = auto()
  NORMAL = auto()
  RANK = auto()

class MODEL_TYPE(Enum):
  SURROGATE = auto()
  DATA = auto()
  SIMULATION = auto()
  NEUTRAL = auto()

class COUPLING_TYPE(Enum):
  SHARED = auto()
  FEEDBACK = auto()
  FEEDFORWARD = auto()
  UNCOUPLED = auto()
  DUMMY = auto()

class COUPLING_STRENGTH(Enum):
  TIGHT = auto()
  LOOSE = auto()

class MDO_ARCHITECTURE(Enum):
  MDF = auto()
  IDF = auto()

@dataclass
class variableData:
  name: str
  sp_index: int
  coupling_type: int
  link: str
  dim: int
  value: float
  baseline: float
  scaling: float
  lb: float = None
  ub: float = None
  type: int = VAR_TYPE.CONTINUOUS
  index: int = None


  def __sub__(self, other):
    return self.value - other.value

  def __add__(self, other):
    return self.value + other.value

  def __mul__(self, other):
    return self.value * other.value

  def __truediv__(self, other):
    if isinstance(other, variableData):
      return self.value / other.value
    else:
      return self.value / other

@dataclass
class Process_data(Protocol):
  term_critteria: List[Callable] = field(init=False)
  term_type: List[int] = field(init=False)
  term_status: List[bool] = field(init=False)
  variables: List[variableData]
  responses: List[variableData]

class coordinator(Protocol):

  def clone_point(self):
    ...

  def create_linking_list(self):
    ...

  def calc_inconsistency(self):
    ...

  def calc_penalty(self):
    ...

  def update_multipliers(self):
    ...

class process(Process_data, Protocol):

  def run(self):
    ...

  def validation(self):
    ...

  def setInputs(self):
    ...

  def getOutputs(self):
    ...

  def setup(self):
    ...

class search(Protocol):

  def evaluateSamples(self):
    ...

  def run(self):
    ...

@dataclass
class ModelInadequacyData:
  type: int
  relative_inadequacies: np.ndarray
  absolute_inadequacies: np.ndarray
  approx_rel_inadeq: np.ndarray
  approx_abs_inadeq: np.ndarray
  reference_model_index: int
  errorSurrogateType: int

@dataclass
class DA_Data:
  inputs: List[variableData]
  outputs: List[variableData]
  blackbox: Callable
  links: List[int]
  coupling_type: List[int]
  preCondition: Optional[Callable] = field(init=False)
  runningCondition: Optional[Callable]  = field(init=False)
  postCondition: Optional[Callable] = field(init=False)
  model: Optional[object] = field(init=False)
  modelType: Optional[int] = field(init=False)
  validation_list: Optional[List[Callable]]  = field(init=False)
  validation_type: Optional[List[int]] = field(init=False)
  validation_status: Optional[List[bool]] = field(init=False)
  index: int = None
  timeout: int = 1000000

@dataclass
class optimizationData:
  objectives: List[Callable]
  constraints: List[Callable]
  objectiveWeights: List[Any]
  constraintsHandling: int
  solver: Callable


@dataclass
class DA(DA_Data):

  def setup(self, input):
    data: Dict = {}
    if isfile(input):
      data = json.load(input)

    if isinstance(input, dict):
      self(**data)

  def run(self):
    if callable(self.blackbox):
      outs = self.blackbox(self.getInputsList())
    else:
      raise IOError("Callables are the only evaluator type currently allowed. Enabling evaluating BB executables is still in progress!")
      evalerr = False
      try:
        p = subprocess.run(self.blackbox, shell=True, timeout=self.timeout)
        if p.returncode != 0:
          evalerr = True
          logging.error("Evaluation # {self.blackbox} is errored!")
      except subprocess.TimeoutExpired:
        timouterr = True 
        logging.error(f'Timeout for {self.blackbox} ({self.timeout}s) expired!')
    self.setOutputsValue(outs)
    return outs

  def validation(self, vType: int):
    self.validation_status = []
    for i in range(len(self.validation_list)):
      if self.validation_type[i] == vType:
          self.validation_status.append(self.validation_list[i])

    return all(self.validation_status)

  def setInputsValue(self, values: List[Any]):
    for i in range(len(self.inputs)):
      self.inputs[i].value = copy.deepcopy(values[i])

  def setOutputsValue(self, values: Any):
    if len(self.outputs) > 1 and isinstance(values, list):
      for i in range(len(self.outputs)):
        self.outputs[i].value = copy.deepcopy(values[i])
    elif len(self.outputs) == 1 and not isinstance(values, list):
      self.outputs[0].value = copy.deepcopy(values)


  def getInputsList(self):
    v = []
    for i in range(len(self.inputs)):
      v.append(self.inputs[i].value)
    return v

  def getOutputs(self):
    return self.outputs

  def getOutputsList(self):
    o = []
    for i in range(len(self.outputs)):
      o.append(self.outputs[i].value)
    return o

@dataclass
class MDA_data(Process_data):
  nAnalyses: int
  analyses: List[DA]
  index: int = None

@dataclass
class MDA(MDA_data):

  def setup(self, input):
    data: Dict = {}
    if isfile(input):
      data = json.load(input)

    if isinstance(input, dict):
      self(**data)

  def run(self):
    for i in range(self.nAnalyses):
      for j in range(len(self.variables)):
        for k in range(len(self.analyses[i].inputs)):
          if self.analyses[i].inputs[k].index == self.variables[j].index:
            self.analyses[i].inputs[k] = copy.deepcopy(self.variables[j])
      self.analyses[i].run()

  def validation(self, vType: int):
    self.term_status = []
    for i in range(len(self.term_critteria)):
      if self.term_type[i] == vType:
          self.term_status.append(self.term_critteria[i])

  def setInputs(self, values: List[Any]):
    self.variables = copy.deepcopy(values)

  def getOutputs(self):
    out = []
    for i in range(len(self.responses)):
      out.append(self.responses[i].value)
    return out

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

@dataclass
class ADMM_data(coordinationData):
  beta: float = 1.3
  gamma: float = 0.5
  q: np.ndarray = np.zeros([0,0])
  qold: np.ndarray = np.zeros([0,0])
  phi: float = 1.0
  v: np.ndarray = np.zeros([0,0])
  w: np.ndarray = np.zeros([0,0])
  update_w: bool = False
  M_update_scheme: int = w_scheme.MEDIAN
  eps_qo: List = None
  save_q_in: bool = False
  save_q_in_out: bool = False
  eps_fo: List = None

# COMPLETE: ADMM needs to be customized for this code
@dataclass
class ADMM(ADMM_data):
  " Alternating directions method of multipliers "
  # Constructor
  def __init__(self, nsp, beta, budget, index_of_master_SP, display, scaling, mode, M_update_scheme, store_q_o=False, store_q_io=False, index = None):
    global eps_fio, eps_qio
    """ Initialize the multiplier vectors """
    self.nsp = nsp
    self.beta = beta
    self.budget = budget
    self.index_of_master_SP = index_of_master_SP
    self.display = display
    self.scaling = scaling
    self.mode = mode
    self.started: bool = False
    self.M_update_scheme = M_update_scheme
    self.eps_qo = []
    self.save_q_out = store_q_o
    self.save_q_in_out = store_q_io
    self.eps_fo = []
    self.index = index

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


  def calc_inconsistency(self):
    if self.save_q_in_out:
      global eps_qio
    q_temp : np.ndarray = np.zeros([0,0])
    for i in range(self._linker.shape[1]):
      if self.master_vars:
        if  (isinstance(self.scaling, list) and len(self.scaling) == len(self.master_vars)):
          qscale = np.multiply(np.add(self.scaling[self._linker[0, i]-1], self.scaling[self._linker[1, i]-1]), 0.5)
          q_temp = np.append(q_temp, np.multiply(subtract(((self.master_vars[self._linker[0, i]-1].value)),
                  ((self.master_vars[self._linker[1, i]-1].value))), qscale))
        elif isinstance(self.scaling, float) or  isinstance(self.scaling, int):
          q_temp = np.append(q_temp, np.multiply(subtract(((self.master_vars[self._linker[0, i]-1].value)),
                  ((self.master_vars[self._linker[1, i]-1].value))), self.scaling))
        else:
          q_temp = np.append(q_temp, np.multiply(subtract(((self.master_vars[self._linker[0, i]-1].value)),
                  ((self.master_vars[self._linker[1, i]-1].value))), min(self.scaling)))
          warning("The inconsistency scaling factors are defined in a list which has a different size from the master variables vector! The minimum value of the provided scaling list will be used to scale the inconsistency vector.")
      else:
        raise Exception(IOError, "Master variables vector have to be non-empty to calculate inconsistencies!")

    self.q = copy.deepcopy(q_temp)
    if self.save_q_out:
      self.eps_qo.append(np.max([abs(x) for x in q_temp]))
    if self.save_q_in_out:
      eps_qio.append(np.max([abs(x) for x in q_temp]))




  def calc_inconsistency_old(self):
    q_temp : np.ndarray = np.zeros([0,0])
    for i in range(self._linker.shape[1]):
      if self.master_vars_old:
        if (isinstance(self.scaling, list) and len(self.scaling) == len(self.master_vars_old)):
          qscale = np.multiply(np.add(self.scaling[self._linker[0, i]-1], self.scaling[self._linker[1, i]-1]), 0.5)
          q_temp = np.append(q_temp, np.multiply(subtract(((self.master_vars_old[self._linker[0, i]-1].value)),
                  ((self.master_vars_old[self._linker[1, i]-1].value))), qscale))
        elif isinstance(self.scaling, float) or  isinstance(self.scaling, int):
          q_temp = np.append(q_temp, np.multiply(subtract(((self.master_vars_old[self._linker[0, i]-1].value)),
                  ((self.master_vars_old[self._linker[1, i]-1].value))), self.scaling))
        else:
          q_temp = np.append(q_temp, np.multiply(subtract(((self.master_vars_old[self._linker[0, i]-1].value)),
                  ((self.master_vars_old[self._linker[1, i]-1].value))), np.min(self.scaling)))
          warning("The inconsistency scaling factors are defined in a list which has a different size from the master variables vector! The minimum value of the provided scaling list will be used to scale the inconsistency vector.")
      else:
        raise Exception(IOError, "Master variables vector have to be non-empty to calculate inconsistencies!")
    self.qold = copy.deepcopy(q_temp)

  def update_master_vector(self, vars: List[variableData], resps: List[variableData]):
    if self.master_vars:
      for i in range(len(vars)):
        self.master_vars[vars[i].index-1] = copy.deepcopy(vars[i])
      for i in range(len(resps)):
        self.master_vars[resps[i].index-1] = copy.deepcopy(resps[i])
    else:
      raise Exception(IOError, "Master variables vector have to be non-empty to calculate inconsistencies!")


  def calc_penalty(self, q_indices):
    phi = np.add(np.multiply(self.v, self.q), np.multiply(np.multiply(self.w, self.w), np.multiply(self.q, self.q)))
    #COMPLETE: Sum relevant components of q to accelerate the convergence of variables consistency
    self.phi = np.sum(phi[q_indices])

    if np.iscomplex(self.phi) or np.isnan(self.phi):
      self.phi = np.inf

  def update_multipliers(self):
    self.v = np.add(self.v, np.multiply(
            np.multiply(np.multiply(2, self.w), self.w), self.q))
    self.calc_inconsistency_old()
    self.q_stall = np.greater(np.abs(self.q), self.gamma*np.abs(self.qold))
    increase_w = []
    if self.M_update_scheme == w_scheme.MEDIAN:
      for i in range(self.q_stall.shape[0]):
        increase_w.append(2. * ((self.q_stall[i]) and (np.greater_equal(np.abs(self.q[i]), np.median(np.abs(self.q))))))
    elif self.M_update_scheme == w_scheme.MAX:
      tc = np.greater_equal(np.abs(self.q), np.max(np.abs(self.q)))
      for i in range(self.q_stall.shape[0]):
        increase_w.append(self.q.shape[0] * (self.q_stall[i] and tc[i]))
    elif self.M_update_scheme == w_scheme.NORMAL:
      increase_w = self.q_stall
    elif self.M_update_scheme == w_scheme.RANK:
      temp = self.q.argsort()
      rank = np.empty_like(temp)
      rank[temp] = np.arange(len(self.q))
      increase_w = np.multiply(np.multiply(2, self.q_stall), np.divide(rank, np.max(rank)))
    else:
      raise Exception(IOError, "Multipliers update scheme is not recognized!")

    for i in range(len(self.w)):
      self.w[i] = copy.deepcopy(np.multiply(self.w[i], np.power(self.beta, increase_w[i])))

    self.w

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
  coord: ADMM
  opt: Callable
  fmin_nop: float
  budget: int
  display: bool
  psize: float
  psize_init: int
  tol: float
  scipy: Dict

@dataclass
class SubProblem(partitionedProblemData):
  # Constructor
  def __init__(self, nv, index, vars, resps, is_main, analysis, coordination, opt, fmin_nop, budget, display, psize, pupdate, scipy=None, freal=None, tol=1E-12, solver='OMADS'):
    self.nv = nv
    self.index = index
    self.vars = copy.deepcopy(vars)
    self.resps = copy.deepcopy(resps)
    self.is_main = is_main
    self.MDA_process = analysis
    self.coord = coordination
    self.opt = opt
    self.optimizer = OMADS.main
    self.fmin_nop = fmin_nop
    self.budget=budget
    self.display = display
    self.psize = psize
    self.psize_init = pupdate
    self.frealistic = freal
    self.tol = tol
    self.solver = solver
    if solver == 'OMADS':
      self.scipy = None
    elif solver == 'scipy':
      self.scipy = scipy
    else:
      warning(f'Inappropriate solver method definition for subproblem # {self.index}! OMADS will be used.')
      self.solver = 'OMADS'
    

    if (self.solver == 'scipy' and (scipy == None or "options" not in self.scipy or "method" not in self.scipy)):
      warning(f'Inappropriate definition of the scipy settings for subproblem # {self.index}! scipy default settings shall be used!')
      self.scipy: Dict = {}
      self.scipy["options"] = {'disp': False}
      self.scipy["method"] = 'SLSQP'
        

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

  def set_variables_value(self, vlist: List):
    for i in range(len(self.vars)):
      if self.vars[i].coupling_type != COUPLING_TYPE.FEEDFORWARD:
        self.vars[i].value = vlist[i]

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
      if (self.index == self.coord.master_vars[i].sp_index or self.coord.master_vars[i].coupling_type != COUPLING_TYPE.UNCOUPLED)  and check and self.coord.master_vars[i].index not in indices2:
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
    
    # Remove redundant links
    self.coord._linker = copy.deepcopy(np.array([indices1, indices2]))

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
    


  def evaluate(self, vlist: List[float]):
    if self.coord.save_q_in_out:
      global eps_fio
    # If no variables were provided use existing value of the variables of the current subproblem (might happen during initialization)
    if vlist is None:
      v: List = self.get_design_vars()
      vlist = self.get_list_vars(v)
    self.set_variables_value(vlist)
    self.MDA_process.setInputs(self.vars)
    self.MDA_process.run()
    y = self.MDA_process.getOutputs()

    fun = self.opt(vlist, y)
    self.fmin_nop = fun[0]
    con = fun[1]
    con = self.get_coupling_vars_diff(con)

    if self.is_main == True:
      if self.frealistic != None and self.frealistic != 0.:
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
    q_indices: List = self.getLocalIndices()
    self.coord.calc_penalty(q_indices)

    if self.realistic_objective:
      con.append(y-self.frealistic)
    if self.solver == "OMADS":
      return [fun[0]+self.coord.phi, con]
    else:
      return fun[0]+self.coord.phi+max(max(con),0)**2

  def solve(self, v, w):
    self.coord.v = copy.deepcopy(v)
    self.coord.w = copy.deepcopy(w)
    bl = self.get_list_vars(self.get_design_vars())
    if self.solver == 'OMADS' or self.solver != 'scipy':
      eval = {"blackbox": self.evaluate}
      param = {"baseline": bl,
                  "lb": self.get_list_vars_lb(self.get_design_vars()),
                  "ub": self.get_list_vars_ub(self.get_design_vars()),
                  "var_names": self.get_list_vars_names(self.get_design_vars()),
                  "scaling": self.get_design_vars_scaling(self.get_design_vars()),
                  "post_dir": "./post"}
      pinit = min(max(self.tol, self.psize), 1)
      options = {
        "seed": 0,
        "budget": 2*self.budget*len(self.vars),
        "tol": max(pinit/1000, self.tol),
        "psize_init": pinit,
        "display": self.display,
        "opportunistic": False,
        "check_cache": True,
        "store_cache": False,
        "collect_y": False,
        "rich_direction": False,
        "precision": "high",
        "save_results": False,
        "save_coordinates": False,
        "save_all_best": False,
        "parallel_mode": False
      }
      data = {"evaluator": eval, "param": param, "options":options}

      out = {}
      pinit = self.psize
      out = self.optimizer(data)
      if self.psize_init == PSIZE_UPDATE.DEFAULT:
        self.psize = 1.
      elif self.psize_init == PSIZE_UPDATE.SUCCESS:
        self.psize = out["psuccess"]
      elif self.psize_init == PSIZE_UPDATE.MAX:
        self.psize = out["pmax"]
      elif self.psize_init == PSIZE_UPDATE.LAST:
        self.psize = out["psize"]
      else:
        self.psize = 1.
      
      # COMPLETE: Coordinator forgets q after calling the optimizer, possible remedy is to update the subproblem variables from the optimizer output and the master variables too
      # then at the end of each outer loop iteration we can calculate q of that subproblem before updating penalty parameters

      #  We need this extra evaluation step to update inconsistincies and the master_variables vector
      self.evaluate(out["xmin"])
    elif self.solver == 'scipy':
      if self.scipy != None and isinstance(self.scipy, dict):
        opts = self.scipy["options"]
        bnds = Bounds(lb=self.get_list_vars_lb(self.get_design_vars()), ub=self.get_list_vars_ub(self.get_design_vars()))
        res = minimize(self.evaluate, method=self.scipy["method"], x0=bl, options=opts, tol=self.tol, bounds=bnds)
        out = copy.deepcopy(res.x)
        self.evaluate(out)
      else:
        raise IOError(f'Scipy solver is selected but its dictionary settings is inappropriately defined!')

    return out

  def get_coupling_vars_diff(self, con):
    vc: List[variableData] = self.get_coupling_vars()
    for i in range(len(vc)):
      con.append(float(vc[i].value-vc[i].ub))

    for i in range(len(vc)):
      con.append(float(vc[i].lb-vc[i].value))
    return con

  def set_dependent_vars(self, vars: List[variableData]):
    for i in range(len(vars)):
      if vars[i].link == self.index:
        if vars[i].coupling_type == COUPLING_TYPE.FEEDFORWARD or vars[i].coupling_type == COUPLING_TYPE.SHARED:
          for j in range(len(self.vars)):
            if self.vars[j].name == vars[i].name:
              self.vars[j].value = vars[i].value

  def get_list_vars(self, vars:List[variableData]):
    v = []
    for i in range(len(vars)):
      v.append(vars[i].value)
    return v

  def get_list_vars_ub(self, vars:List[variableData]):
    v = []
    for i in range(len(vars)):
      v.append(vars[i].ub)
    return v

  def get_list_vars_lb(self, vars:List[variableData]):
    v = []
    for i in range(len(vars)):
      v.append(vars[i].lb)
    return v

  def get_design_vars_scaling(self, vars:List[variableData]):
    v = []
    for i in range(len(self.vars)):
      v.append(self.vars[i].scaling)
    return v

  def get_list_vars_names(self,vars:List[variableData]):
    v = []
    for i in range(len(vars)):
      v.append(vars[i].name)
    return v

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


@dataclass
class MDO(MDO_data):
  def setup(self, input):
    data: Dict = {}
    if isfile(input):
      data = json.load(input)

  def get_list_of_var_values(x: List[variableData]):
    x_temp = []
    for i in range(len(x)):
      x_temp.append(x[i].value)
    return x_temp

  def get_master_vars_difference(self):
    dx = []
    x =[]
    xold =[]
    for i in range(len(self.Coordinator.master_vars)):
      dx.append(self.Coordinator.master_vars[i] - self.Coordinator.master_vars_old[i])
      x.append(self.Coordinator.master_vars[i].value)
      xold.append(self.Coordinator.master_vars_old[i].value)
    return np.linalg.norm(dx, 2)

  def check_termination_critt(self, iter):
    if iter > 1 and np.abs(self.tab_inc[iter]) < self.inc_stop:
      print(f'Stop: qmax = {np.max(np.abs(self.Coordinator.q))} < {self.inc_stop}')
      self.stop = "Max inconsitency is below stopping threshold"
      return True

    i = self.noprogress_stop

    if iter > i + 2 and np.log(np.min(self.Coordinator.w) > 6.) and np.less_equal(self.tab_inc[iter-i], np.min(self.tab_inc[iter-i+1:iter])):
      print(f'Stop: no progress after {i} iterations.')
      self.stop = "No progress after several iterations"
      return True

    return False
  
  def run(self):
    global eps_fio, eps_qio
    # Note: once you run MDAO, the data stored in eps_fio and eps_qio shall be deleted. It is recommended to store such data to a different variable before running another MDAO
    eps_fio = []
    eps_qio = []
    """ Run the MDO process """
    #  COMPLETE: fix the setup of the local (associated with SP) and global (associated with MDO) coordinators
    for iter in range(self.Coordinator.budget):
      if iter > 0:
        self.Coordinator.master_vars_old = copy.deepcopy(self.Coordinator.master_vars)
      else:
        self.Coordinator.master_vars_old = copy.deepcopy(self.variables)
        self.Coordinator.master_vars = copy.deepcopy(self.variables)

      """ ADMM inner loop """
      for s in range(len(self.subProblems)):
        if iter == 0:
          self.subProblems[s].coord = copy.deepcopy(self.Coordinator)
          self.subProblems[s].set_pair()
          self.xavg = self.subProblems[s].coord._linker
          self.Coordinator.v = [0.] * len(self.subProblems[s].coord._linker[0])
          self.Coordinator.w = [1.] * len(self.subProblems[s].coord._linker[0])
        else:
          self.subProblems[s].coord.master_vars = copy.deepcopy(self.Coordinator.master_vars)

        out_sp = self.subProblems[s].solve(self.Coordinator.v, self.Coordinator.w)
        self.Coordinator = copy.deepcopy(self.subProblems[s].coord)
        if self.subProblems[s].index == self.Coordinator.index_of_master_SP:
          self.fmin = self.subProblems[s].fmin_nop
          if self.subProblems[s].solver == "OMADS":
            self.hmin = out_sp["hmin"]
          else:
            self.hmin = [0.]

      """ Display convergence """
      dx = self.get_master_vars_difference()
      if self.display:
        print(f'{iter} || qmax: {np.max(np.abs(self.Coordinator.q))} || Obj: {self.fmin} || dx: {dx} || max(w): {np.max(self.Coordinator.w)}')
        index = np.argmax(self.Coordinator.q)
        print(f'Highest inconsistency : {self.Coordinator.master_vars[self.Coordinator._linker[0,index]-1].name}_'
          f'{self.Coordinator.master_vars[self.Coordinator._linker[0,index]-1].link} to '
            f'{self.Coordinator.master_vars[self.Coordinator._linker[1,index]-1].name}_'
          f'{self.Coordinator.master_vars[self.Coordinator._linker[1,index]-1].link}')
      """ Update LM and PM """
      self.Coordinator.update_multipliers()

      """ Stopping criteria """
      self.tab_inc.append(np.max(np.abs(self.Coordinator.q)))
      stop: bool = self.check_termination_critt(iter)

      if stop:
        break
      self.eps_qio = copy.deepcopy(eps_qio)
      self.eps_fio = copy.deepcopy(eps_fio)
    return self.Coordinator.q

  def validation(self, vType: int):
    self.term_status = []
    for i in range(len(self.term_critteria)):
      if self.term_type[i] == vType:
          self.term_status.append(self.term_critteria[i])

  def setInputs(self, values: List[Any]):
    self.variables = copy.deepcopy(values)

  def getOutputs(self):
    return self.responses

@dataclass
class problemSetup:
  data: Dict = None
  Vs: List[variableData] = None
  DAs: List[process] = None
  MDAs: List[process] = None
  Coords: List[coordinator] = None
  SPs: List[SubProblem] = None
  MDAO: MDO = None
  Qscaling: List = None
  userData: USER = None

  def setup_couplingList(self)->List:
    """ Build the list of the coupling types """
    ct = []
    vin: Dict = self.data["variables"]
    for i in vin:
      if vin[i][3] == 's':
        ct.append(COUPLING_TYPE.SHARED)
      elif vin[i][3] == 'u':
        ct.append(COUPLING_TYPE.UNCOUPLED)
      elif vin[i][3] == 'ff':
        ct.append(COUPLING_TYPE.FEEDFORWARD)
      elif vin[i][3] == 'fb':
        ct.append(COUPLING_TYPE.FEEDBACK)
      elif vin[i][3] == 'dummy':
        ct.append(COUPLING_TYPE.DUMMY)
      else:
        raise IOError(f'Unrecognized coupling type {vin[i][2]} is introduced to the variables dictionary at this key {i}')
    return ct
  
  def getWupdateScheme(self, inp: str) -> int:
    if inp.lower() == "max":
      return w_scheme.MAX
    elif inp.lower() == "normal":
      return w_scheme.NORMAL
    elif inp.lower() == "rank":
      return w_scheme.RANK
    elif inp.lower() == "median":
      return w_scheme.MEDIAN
    else:
      return None

  def getCouplingType(self, inp: str) -> int:
    if inp.lower() == "ff":
      return COUPLING_TYPE.FEEDFORWARD
    elif inp.lower() == "fb":
      return COUPLING_TYPE.FEEDBACK
    elif inp.lower() == "s":
      return COUPLING_TYPE.SHARED
    elif inp.lower() == "un":
      return COUPLING_TYPE.UNCOUPLED
    elif inp.lower() == "dummy":
      return COUPLING_TYPE.DUMMY
    else:
      return None
  
  def getPollUpdate(self, inp: str) -> int:
    if inp.lower() == "last":
      return PSIZE_UPDATE.LAST
    elif inp.lower() == "default":
      return PSIZE_UPDATE.DEFAULT
    elif inp.lower() == "success":
      return PSIZE_UPDATE.SUCCESS
    elif inp.lower() == "max":
      return PSIZE_UPDATE.MAX
    else:
      return None
  
  def getMDOArch(self, inp: str) -> int:
    if inp.lower() == "idf":
      return MDO_ARCHITECTURE.IDF
    elif inp.lower() == "MDF":
      return MDO_ARCHITECTURE.MDF
    else:
      MDO_ARCHITECTURE.IDF
  
  def variablesSetup(self):
    """ Setup the list of global variables struct """
    vin: Dict = self.data["variables"]
    # del vin["index"]
    v = {}
    self.V = []
    names: List[str] = [vin[i][0] for i in vin]
    spi: List[int] = [vin[i][1] for i in vin]
    links: List = [vin[i][2] if vin[i][2] != "None" else None for i in vin]
    coupling_t: List = self.setup_couplingList()
    lb: List = [vin[i][4] for i in vin]
    bl: List = [vin[i][5] for i in vin]
    ub: List = [vin[i][6] for i in vin]
    dim: List = [vin[i][7] for i in vin]

    scaling = np.subtract(ub,lb)
    self.Qscaling = []
    c = 0
    for i in range(len(names)):
      if dim[i] > 1:
        for j in range(dim[i]):
          v[f"var{i+c+1}"] = {"index": i+c+1,
          "sp_index": spi[i],
          f"name": names[i],
          "dim": 1,
          "value": 0.,
          "coupling_type": coupling_t[i],
          "link": links[i],
          "baseline": bl[i],
          "scaling": scaling[i],
          "lb": lb[i],
          "value": bl[i],
          "ub": ub[i]}
          c += 1
          self.Qscaling.append(.1/scaling[i] if .1/scaling[i] != np.inf and .1/scaling[i] != np.nan else 1.)
        c -=1
      else:
        v[f"var{i+c+1}"] = {"index": i+c+1,
        "sp_index": spi[i],
        f"name": names[i],
        "dim": 1,
        "value": 0.,
        "coupling_type": coupling_t[i],
        "link": links[i],
        "baseline": bl[i],
        "scaling": scaling[i],
        "lb": lb[i],
        "value": bl[i],
        "ub": ub[i]}
        self.Qscaling.append(.1/scaling[i] if .1/scaling[i] != np.inf and .1/scaling[i] != np.nan else 1.)
    
    for i in range(len(names)):
      self.V.append(variableData(**v[f"var{i+1}"]))
  
  def getVariables(self, v: List[str]) -> List[variableData]:
    """ Find the variables by the input key assigned to them"""
    out = []
    for i in v:
      namet = self.data["variables"][i][0]
      indext = self.data["variables"][i][1]
      for k in self.V:
        if k.name == namet and k.sp_index == indext:
          out.append(k)
    
    return out
  
  def getDA(self, d: List[int]) -> List[DA]:
    """ Get the list of corresponding disciplinary analyses """
    out = []
    for i in d:
      for k in self.DAs:
        if k.index == i:
          out.append(k)
    
    return out


  def getSPs(self, sp: List[int]) -> List[SubProblem]:
    """ Get the list of corresponding disciplinary analyses """
    out = []
    for i in sp:
      for k in self.SPs:
        if k.index == i:
          out.append(k)
    
    return out
  
  def getMDA(self, M: int) -> MDA:
    """ Get the MDA process from the provided index """
    k: MDA = None
    for k in self.MDAs:
      if k.index == M:
        return k
    return k
  
  def getCoord(self, c: int) -> ADMM:
    """ Get the coordinator from the provided index """
    k: ADMM = None
    for k in self.Coords:
      if k.index == c:
        return k
    return k
  
  def setBlackboxes(self, bb:str, bb_type: str, copts: str = None):
    if bb_type == "callable":
      return bb
    else:
      isWin = platform.platform().split('-')[0] == 'Windows'
      #  Check if the file is executable
      executable = os.access(bb, os.X_OK)
      if not executable:
        raise IOError(f"The blackbox file {str(bb)} is not an executable! Please provide a valid executable file.")
      # Prepare the execution command based on the running machine's OS
      if isWin and copts is None:
        cmd = bb
      elif isWin:
        cmd = f'{bb} {copts}'
      elif copts is None:
        cmd = f'./{bb}'
      else:
        cmd =  f'./{bb} {copts}'
      return cmd



  def DASetup(self):
    """ Setup discipliary Analyses """
    D = self.data["DA"]
    self.DAs = []
    for i in D:
      self.DAs.append(DA(
        index=D[i]["index"], 
        inputs=self.getVariables(D[i]["inputs"]), 
        outputs=self.getVariables(D[i]["outputs"]), 
        blackbox=self.setBlackboxes(D[i]["blackbox"], bb_type=D[i]["type"], copts = None), 
        links=D[i]["links"], 
        coupling_type=self.getCouplingType(D[i]["coupling_type"])
        ))
  
  def MDASetup(self):
    """ Setup the MDA process """
    MD = self.data["MDA"]
    self.MDAs = []
    for i in MD:
      self.MDAs.append(MDA(
        index=MD[i]["index"], 
        nAnalyses=MD[i]["nAnalyses"], 
        analyses=self.getDA(MD[i]["analyses"]), 
        variables=self.getVariables(MD[i]["variables"]), 
        responses=self.getVariables(MD[i]["responses"])))

  def COORDSetup(self):
    """ Setup coordinators definition """
    c = self.data["coord"]
    self.Coords = []
    for i in c:
      if c[i]["type"] != "ADMM":
        raise IOError(f'Currently DMDO only supports ADMM coordinator. Please change the coordination type under {i} to ADMM.')
      self.Coords.append(ADMM(
        index= c[i]["index"],
        beta=  c[i]["beta"],
        nsp=  c[i]["nsp"],
        index_of_master_SP= c[i]["index_of_master_SP"],
        display= c[i]["display"],
        scaling= c[i]["scaling"] if isinstance(c[i]["scaling"], float)  else copy.deepcopy(self.Qscaling),
        mode=c[i]["mode"],
        M_update_scheme=self.getWupdateScheme(c[i]["M_update_scheme"]),
        store_q_io=c[i]["store_q_io"],
        budget=c[i]["budget"] if "budget" in c[i] else IOError(f'The budget key could not be found for {c[i]}.')
      ))

  def SPSetup(self):
    """ Setup subproblems definition """
    SP = self.data["subproblem"]
    self.SPs = []
    for i in SP:
      self.SPs.append(SubProblem(
        nv= SP[i]["nv"],
        index= SP[i]["index"],
        vars= self.getVariables(SP[i]["vars"]),
        resps= self.getVariables(SP[i]["resps"]),
        is_main= SP[i]["is_main"],
        analysis=self.getMDA(SP[i]["MDA"]) if self.getMDA(SP[i]["MDA"]) is not None else IOError(f'Could not find the MDA with index {SP[i]["MDA"]} assigned to the subproblem {SP[i]["index"]} MDA key.'),
        coordination=self.getCoord(SP[i]["coordinator"]) if self.getMDA(SP[i]["coordinator"]) is not None else IOError(f'Could not find the coordinator with index {SP[i]["coordinator"]} assigned to the subproblem {SP[i]["index"]} coordinator key.'),
        opt=self.setBlackboxes(SP[i]["opt"], bb_type=SP[i]["type"], copts = None) if "opt" in SP[i] else IOError(f'The optimization blackbox key could not be found for {SP[i]}.'),
        fmin_nop=np.inf if not isinstance(SP[i]["fmin_nop"], float) and not isinstance(SP[i]["fmin_nop"], int) else SP[i]["fmin_nop"],
        budget= SP[i]["budget"] if "budget" in SP[i] else IOError(f'The budget key could not be found for {SP[i]}.'),
        display=SP[i]["display"] if "display" in SP[i] else False,
        psize=SP[i]["psize"] if "psize" in SP[i] else 1.,
        pupdate=self.getPollUpdate(SP[i]["pupdate"]) if "pupdate" in SP[i] and self.getPollUpdate(SP[i]["pupdate"]) is not None else PSIZE_UPDATE.LAST,
        freal=SP[i]["freal"] if "freal" in SP[i] else None,
        solver=SP[i]["solver"] if "solver" in SP[i] else "OMADS",
        scipy= SP[i]["scipy"] if "scipy" in SP[i] else None
      ))

  def MDOSetup(self):
    """ Setup MDO process """
    MDAO = self.data["MDO"]
    self.MDAO = MDO(
      Architecture= self.getMDOArch(MDAO["architecture"]) if MDAO["architecture"] is not None else MDO_ARCHITECTURE.IDF,
      Coordinator= self.getCoord(MDAO["coordinator"]) if self.getMDA(MDAO["coordinator"]) is not None else IOError(f'Could not find the coordinator with index {MDAO["coordinator"]} assigned to the MDO coordinator key.'),
      subProblems= self.getSPs(MDAO["subproblems"]),
      variables= self.V,
      responses= self.getVariables(MDAO["responses"]),
      fmin = np.inf if not isinstance(MDAO["fmin"], int) and not isinstance(MDAO["fmin"], float) else MDAO["fmin"],
      hmin = np.inf if not isinstance(MDAO["hmin"], int) and not isinstance(MDAO["hmin"], float) else MDAO["hmin"],
      display=MDAO["display"] if "display" in MDAO else True,
      inc_stop=MDAO["inc_stop"] if "inc_stop" in MDAO and isinstance(MDAO["inc_stop"], float) else 1E-9,
      stop=MDAO["stop"] if "stop" in MDAO and isinstance(MDAO["stop"], str) else "Iteration budget exhausted",
      tab_inc = MDAO["tab_inc"] if "tab_inc" in MDAO and isinstance(MDAO["tab_inc"], list) else [],
      noprogress_stop= MDAO["noprogress_stop"] if "noprogress_stop" in MDAO and isinstance(MDAO["noprogress_stop"], int) else 100,
    )
  
  def UserData(self):
    """ Set user data attr """
    u = self.data["USER"]
    if u is not None:
      for i in u:
        setattr(self.userData, i, u[i])
    else:
      self.userData = None

  def autoProbSetup(self) -> MDO:
    """ Setup the MDO problem """
    self.variablesSetup()
    self.DASetup()
    self.MDASetup()
    self.COORDSetup()
    self.SPSetup()
    self.MDOSetup()
    self.UserData()

    return self.MDAO


# TODO: MDO setup will be simplified when the N2 chart UI is implemented
def main(*args) -> Dict[str, Any]:
  """ The DMDO main routine """
  if not isinstance(args[0], str): 
    raise IOError(f'{args[0]} is not a string input! Please use an appropriate DMDO yaml file!')
  
  file: str = args[0]
  fext = file.split('.')[1]
  if not (fext == "yaml" or fext == "yml"):
    raise IOError(f'Cannot use files with {fext} extension. Please use an appropriate yaml file with yml or yaml extension!')
  
  if not os.path.exists(file):
    raise IOError(f'Could not find {file}! Please make sure that the file exists!')
  
  with open(file, "r") as stream:
    try:
        data: Dict = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise IOError(exc)

  MDAO: MDO  
  MDAO= problemSetup(data=data).autoProbSetup()

  if args[1].lower() != "run":
    return MDAO

  MDAO.run()

  if MDAO.display == True:
    print(f'------Run_Summary------')
    print(MDAO.stop)
    print(f'q = {MDAO.Coordinator.q}')
    for i in MDAO.Coordinator.master_vars:
      print(f'{i.name}_{i.sp_index} = {i.value}')

    fmin = 0
    hmax = -np.inf
    for j in range(len(MDAO.subProblems)):
      print(f'SP_{MDAO.subProblems[j].index}: fmin= {MDAO.subProblems[j].MDA_process.getOutputs()}, hmin= {MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] , MDAO.subProblems[j].MDA_process.getOutputs())[1]}')
      fmin += sum(MDAO.subProblems[j].MDA_process.getOutputs())
      hmin= MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] , MDAO.subProblems[j].MDA_process.getOutputs())[1]
      if max(hmin) > hmax: 
        hmax = max(hmin) 
    print(f'P_main: fmin= {fmin}, hmax= {hmax}')
    print(f'Final obj value of the main problem: \n {fmin}')
  
  return MDAO

# def A1(x):
#   LAMBDA = 0.0
#   for i in range(len(x)):
#     if x[i] == 0.:
#       x[i] = 1e-12
#   return math.log(x[0]+LAMBDA) + math.log(x[1]+LAMBDA) + math.log(x[2]+LAMBDA)

# def A2(x):
#   LAMBDA = 0.0
#   for i in range(len(x)):
#     if x[i] == 0.:
#       x[i] = 1e-12
#   return np.divide(1., (x[0]+LAMBDA)) + np.divide(1., (x[1]+LAMBDA)) + np.divide(1., (x[2]+LAMBDA))

# def opt1(x, y):
#   return [sum(x)+y[0], [0.]]

# def opt2(x, y):
#   return [0., [x[1]+y[0]-10.]]

if __name__ == "__main__":
  #COMPLETED: Feature: Add more realistic analytical test problems
  #TODO: Feature: Add realistic multi-physics MDO problems that require using open-source physics-based simulation tools
  #COMPLETED: Feature: Move the MDO test functions and BM problems to the test folder and prepare the DMDO package to be published on PyPi
  #TODO: Feature: Develop a simple UI widget that facilitates simple MDO setup using the compact table or N2-chart
  #TODO: Feature: Import RAF and SML libraries once they are published on PYPI.com
  #FIXME: Bug: Add user and technical documentation 
  #FIXME: Bug: Enable the output report generation that summarizes the MDO history and final results
  p_file: str = ""

  """ Check if an input argument is provided"""
  if len(sys.argv) > 1:
    p_file = os.path.abspath(sys.argv[1])
    main(p_file, sys.argv[2], sys.argv[3:])

  if (p_file != "" and os.path.exists(p_file)):
    main(p_file, "build")

  if p_file == "":
    raise IOError("Undefined input args."
            " Please specify an appropriate DMDO input yamle file")
