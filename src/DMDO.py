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
from multiprocessing.dummy import Process
from multiprocessing.sharedctypes import Value

from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Protocol, Optional

import numpy as np
from numpy import cos, exp, pi, prod, sin, sqrt, subtract, inf
import math

import OMADS
from enum import Enum, auto
from scipy.optimize import minimize, Bounds

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
  index: int
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
    outs = self.blackbox(self.getInputsList())
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

# COMPLETE: ADMM needs to be customized for this code
@dataclass
class ADMM(ADMM_data):
  " Alternating directions method of multipliers "
  # Constructor
  def __init__(self, nsp, beta, budget, index_of_master_SP, display, scaling, mode, M_update_scheme):
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
      increase_w = self.q.shape * (self.q_stall and np.greater_equal(np.abs(self.q), np.max(np.abs(self.q))))
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
  realistic_objective: float = False
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

# TODO: MDO setup will be simplified when the N2 chart UI is implemented
def termTest():
  print("Termination criterria work!")

def A1(x):
  LAMBDA = 0.0
  for i in range(len(x)):
    if x[i] == 0.:
      x[i] = 1e-12
  return math.log(x[0]+LAMBDA) + math.log(x[1]+LAMBDA) + math.log(x[2]+LAMBDA)

def A2(x):
  LAMBDA = 0.0
  for i in range(len(x)):
    if x[i] == 0.:
      x[i] = 1e-12
  return np.divide(1., (x[0]+LAMBDA)) + np.divide(1., (x[1]+LAMBDA)) + np.divide(1., (x[2]+LAMBDA))

def opt1(x, y):
  return [sum(x)+y[0], [0.]]

def opt2(x, y):
  return [0., [x[1]+y[0]-10.]]

def SR_A1(x):
  """ Speed reducer A1 """
  return (0.7854*x[0]*x[1]**2*(3.3333*x[2]*x[2] + 14.9335*x[2] - 43.0934))

def SR_A2(x):
  """ Speed reducer A2 """
  return (-1.5079*x[0]*x[4]**2) + (7.477 * x[4]**3) + 0.7854 * x[3] * x[4]**2

def SR_A3(x):
  """ Speed reducer A3 """
  return (-1.5079*x[0]*x[4]**2 + 7.477*x[4]**3 + 0.7854*x[3]*x[4]**2)

def SR_opt1(x, y):
  g5 = 27/(x[0]*x[1]**2*x[2]) -1
  g6 = 397.5/(x[0]*x[1]**2*x[2]**2) -1
  g9 = x[1]*x[2]/40 -1
  g10 = 5*x[1]/x[0] -1
  g11 = x[0]/(12*x[1]) -1
  return [y, [g5,g6,g9,g10,g11]]


def SR_opt2(x, y):
  g1 = sqrt( ((745*x[3])/(x[1]*x[2]))**2 + 1.69e+7)/(110*x[4]**3) -1
  g3 = (1.5*x[4] + 1.9)/x[3] -1
  g7 = 1.93*x[3]**3/(x[1]*x[2]*x[4]**4) -1
  return [y, [g1,g3,g7]]

def SR_opt3(x, y):
  g2 = sqrt( ((745*x[3])/(x[1]*x[2]))**2 + 1.575e+8)/(85*x[4]**3) -1
  g4 = (1.1*x[4] + 1.9)/x[3] -1
  g8 = (1.93*x[3]**3)/(x[1]*x[2]*x[4]**4) -1
  return [y, [g2, g4, g8]]

def GP_A1(z):
  z1 = sqrt(z[0]**2 + z[1]**-2 + z[2]**2)
  z2 = sqrt(z[2]**2 + z[3]**2  + z[4]**2)
  return z1**2 + z2**2

def GP_opt1(z,y):
  if isinstance(y, list) and len(y) > 0:
    return [y[0], [z[0]**-2 + z[1]**2 - z[2]**2, z[2]**2 + z[3]**-2  - z[4]**2]]
  else:
    return [y, [z[0]**-2 + z[1]**2 - z[2]**2, z[2]**2 + z[3]**-2  - z[4]**2]]

def GP_A2(z):
  z3 = sqrt(z[0]**2 + z[1]**-2 + z[2]**-2 + z[3]**2)
  return z3

def GP_opt2(z,y):
  return [0, [z[0]**2 + z[1]**2 - z[3]**2, z[0]**-2 + z[2]**2 - z[3]**2]]

def GP_A3(z):
  z6 = sqrt(z[0]**2 + z[1]**2 + z[2]**2 +z[3] **2)
  return z6

def GP_opt3(z, y):
  return [0, [z[0]**2 + z[1]**-2 - z[2]**2, z[0]**2 +z[1]**2 - z[3]**2]]

def SBJ_A1(x):
  # SBJ_obj_range
  return x[1]+x[3]+x[4]

def SBJ_opt1(x, y):
  # SBJ_constraint_range
  if user.h <36089:
    theta = 1-0.000006875*user.h
  else:
    theta = 0.7519
  r = user.M * x[2] * 661 * np.sqrt(theta/x[0])*math.log(y[0]/(y[0]-x[4]))
  G = -r/2000 +1

  return [y[0], [G]]

def SBJ_A2(x):
  # SBJ_obj_power
  #  THIS SECTION COMPUTES SFC, ESF, AND ENGINE WEIGHT
  C=[500.0, 16000.0, 4.0 , 4360.0,  0.01375,  1.0]
  Thrust = x[0]
  Dim_Throttle = x[1]*16168
  s=[1.13238425638512, 1.53436586044561, -0.00003295564466, -0.00016378694115, -0.31623315541888, 0.00000410691343, -0.00005248000590, -0.00000000008574, 0.00000000190214, 0.00000001059951]
  SFC = s[0]+s[1]*user.M+s[2]*user.h+s[3]*Dim_Throttle+s[4]*user.M**2+2*user.h*user.M*s[5]+2*Dim_Throttle*user.M*s[6]+s[7]*user.h**2+2*Dim_Throttle*user.h*s[8]+s[9]*Dim_Throttle**2

  ESF = (Thrust/2)/Dim_Throttle

  We = C[3]*(ESF**1.05)*2

  return [We,SFC, ESF]

def SBJ_opt2(x, y):
  # SBJ_constraint_power
  # -----THIS SECTION COMPUTES POLYNOMIAL CONSTRAINT FUNCTIONS-----
  Dim_Throttle = x[1]*16168
  S_initial1=[user.M,user.h,x[0]]
  S1=[user.M,user.h,x[1]]
  flag1 = [2,4,2]
  bound1 = [.25,.25,.25]
  Temp_uA=1.02
  g1 = polyApprox(S_initial1, S1, flag1, bound1)
  g1 = g1 /Temp_uA-1
  p=[11483.7822254806, 10856.2163466548, -0.5080237941, 3200.157926969, -0.1466251679, 0.0000068572]
  Throttle_uA=p[0]+p[1]*user.M+p[2]*user.h+p[3]*user.M**2+2*p[4]*user.M*user.h+p[5]*user.h**2

  return[0, [g1, Dim_Throttle/Throttle_uA-1]]


def SBJ_A3(x):
  # SBJ_obj_dragpolar
  # %----Inputs----%
  Z = [x[3],user.h,user.M,x[4],x[5],x[6],x[7],x[8]]
  C = [500.0, 16000.0,  4.0,  4360.0,  0.01375,  1.0]
  Z = [1,	55000,	1.40000000000000,	1,	1,	1,	1,	1]
  ARht=Z[7]
  S_ht=Z[6]    
  Nh=C[5]

  # %-----Drag computations----%

  if Z[1]<36089:
     V = Z[2]*(1116.39*sqrt(1-(6.875e-06*Z[1])))
     rho = (2.377e-03)*(1-(6.875e-06*Z[1]))**4.2561
  else:
     V = Z[2]*968.1
     rho = (2.377e-03)*(.2971)*np.exp(-(Z[1]-36089)/20806.7)
  q=.5*rho*(V**2)

  # ### Modified by S. Tosserams:
  # # scale coefficients for proper conditioning of matrix A 
  a=q*Z[5]/1e5
  b=Nh*q*S_ht/1e5
  # -------------------------------
  c= x[10] #Lw
  d=(x[11])*Nh*(S_ht/Z[5])
  A= np.array([[a, b], [c, d]])
  # ---- Modified by S. Tosserams:
  # ---- scale coefficient Wt for proper conditioning of matrix A 
  B=np.array([x[1]/1e5, 0])
  # -----------------------
  try:
    CLo=np.linalg.solve(A, B)
  except:
    CLo = np.array([-np.inf, np.inf])
  delta_L=x[2]*q
  Lw1=CLo[0]*q*Z[5]-delta_L
  CLw1=Lw1/(q*Z[5])
  CLht1=-CLw1*c/d
  #  Modified by S. Tosserams:
  #  scale first coefficient of D for proper conditioning of matrix A 
  D=np.array([(x[1]-CLw1*a-CLht1*b)/1e5, -CLw1*c-CLht1*d])
  # -----------------
  try:
    DCL = np.linalg.solve(A, D)
  except:
    DCL = np.array([np.nan,np.nan])

  if Z[2] >= 1:
    kw = Z[3] * (Z[2]**2-1) * np.cos(Z[4]*np.pi/180)/(4*Z[3]*np.sqrt(Z[2]**2-1)-2)
    kht = ARht * (Z[2]**2-1)*np.cos(x[9]*pi/180)/(4*ARht*np.sqrt(Z[2]**2-1)-2)
  else:
    kw = 1/(np.pi*0.8*Z[3])
    kht= 1/(np.pi*0.8*ARht)
  
  S_initial1 = copy.deepcopy(x[0])
  S1 = copy.deepcopy(x[0])
  flag1 = 1
  bound1 = 0.25
  Fo1 = polyApprox(S_initial1 if isinstance(S_initial1, list) else [S_initial1], S1 if isinstance(S1, list) else [S1], flag1 if isinstance(flag1, list) else [flag1], bound1 if isinstance(bound1, list) else [bound1])


  CDmin = C[4]*Fo1 + 3.05*(Z[0]**(5/3))*((cos(Z[4]*np.pi/180))**(3/2))

  CDw=CDmin+kw*(CLo[0]**2)+kw*(DCL[0]**2)
  CDht=kht*(CLo[1]**2)+kht*(DCL[1]**2)
  CD=CDw+CDht
  CL=CLo[0]+CLo[1]
  L = x[1]
  D = q*CDw*Z[5]+q*CDht*Z[6]
  LD = CL/CD

  return [D, LD, L]

def SBJ_opt3(x, y):
  # SBJ_constraint_dragpolar
  # %----Inputs----%
  Z = [x[3],user.h,user.M,x[4],x[5],x[6],x[7],x[9]]
  C = [500.0, 16000.0,  4.0,  4360.0,  0.01375,  1.0]
  S_ht=Z[6]
  Nh=C[5]

  # %-----Drag computations----%

  if Z[1]<36089:
     V = Z[2]*(1116.39*sqrt(1-(6.875e-06*Z[1])))
     rho = (2.377e-03)*(1-(6.875e-06*Z[1]))**4.2561
  else:
     V = Z[2]*968.1
     rho = (2.377e-03)*(.2971)*exp(-(Z[1]-36089)/20806.7)
  q=.5*rho*(V**2)

  # ### Modified by S. Tosserams:
  # # scale coefficients for proper conditioning of matrix A 
  a=q*Z[5]/1e5
  b=Nh*q*S_ht/1e5
  # -------------------------------
  c= x[10] #Lw
  d=(x[11])*Nh*(S_ht/Z[5])
  A= np.array([[a, b], [c, d]])
  # ---- Modified by S. Tosserams:
  # ---- scale coefficient Wt for proper conditioning of matrix A 
  B=[x[1]/1e5, 0]
  # -----------------------
  try:
    CLo=np.linalg.solve(A, B)
  except:
    CLo = np.array([-np.inf, np.inf])
  
  S_initial2 = copy.deepcopy(x[3])
  S2 = copy.deepcopy(Z[0])
  flag1 = [1]
  bound1 = [0.25]
  
  g1 = polyApprox(S_initial2 if isinstance(S_initial2, list) else [S_initial2], S2 if isinstance(S2, list) else [S2], flag1 if isinstance(flag1, list) else [flag1], bound1 if isinstance(bound1, list) else [bound1])


  # %------Constraints------%

  Pg_uA=1.1
  g1=g1/Pg_uA-1
  if CLo[0] > 0:
      g2=(2*(CLo[1]))-(CLo[0])
      g3=(2*(-CLo[1]))-(CLo[0])
  else:
      g2=(2*(-CLo[1]))-(CLo[0])
      g3=(2*(CLo[1]))-(CLo[0])

  return [0, [g1, g2, g3]]

def SBJ_A4(x):
  # SBJ_obj_weight
  Z= [x[1], user.h, user.M, x[2], x[3], x[4], x[5], x[6]]
  L = x[0]
  # %----Inputs----%
  t= np.divide(x[7:16], 12.)#convert to feet
  ts= np.divide(x[16:], 12.);#convert to feet
  LAMBDA = x[10]
  C=[500.0, 16000.0,  4.0,  4360.0,  0.01375,  1.0]

  t1= [t[i] for i in range(3)] 
  t2= [t[i] for i in range(3, 6)]  
  t3= [t[i] for i in range(6, 9)]  
  ts1=[ts[i] for i in range(3)] 
  ts2=[ts[i] for i in range(3, 6)]  
  ts3=[ts[i] for i in range(6, 9)]  
  G=4000000*144
  E=10600000*144
  rho_alum=0.1*144
  rho_core=0.1*144/10
  rho_fuel=6.5*7.4805
  Fw_at_t=5

  # %----Inputs----%
  beta=.9
  c,c_box,Sweep_40,D_mx,b,_ = Wing_Mod(Z,LAMBDA)
  l=0.6*c_box
  h=(np.multiply([c[i] for i in range(3)], beta*float(Z[0])))-np.multiply((0.5),np.add(ts1,ts3))
  A_top=(np.multiply(t1, 0.5*l))+(np.multiply(t2, h/6))
  A_bottom=(np.multiply(t3, 0.5*l))+(np.multiply(t2, h/6))
  Y_bar=np.multiply(h, np.divide((2*A_top), (2*A_top+2*A_bottom)))
  Izz=np.multiply(2, np.multiply(A_top, np.power((h-Y_bar), 2)))+np.multiply(2, np.multiply(A_bottom,np.power((-Y_bar), 2)))
  P,Mz,Mx,bend_twist,Spanel=loads(b,c,Sweep_40,D_mx,L,Izz,Z,E)

  Phi=(Mx/(4*G*(l*h)**2))*(l/t1+2*h/t2+l/t3)
  aa=len(bend_twist)
  twist = np.array([0] * aa)
  twist[0:int(aa/3)]=bend_twist[0:int(aa/3)]+Phi[0]*180/pi
  twist[int(aa/3):int(aa*2/3)]=bend_twist[int(aa/3):int(aa*2/3)]+Phi[1]*180/pi
  twist[int(aa*2/3):aa]=bend_twist[int(aa*2/3):aa]+Phi[2]*180/pi
  deltaL_divby_q=sum(twist*Spanel*0.1*2)


  # %-----THIS SECTION COMPUTES THE TOTAL WEIGHT OF A/C-----%

  Wtop_alum=(b/4)*(c[0]+c[3])*np.mean(t1)*rho_alum
  Wbottom_alum=(b/4)*(c[0]+c[3])*np.mean(t3)*rho_alum
  Wside_alum=(b/2)*np.mean(h)*np.mean(t2)*rho_alum
  Wtop_core=(b/4)*(c[0]+c[3])*np.mean(np.subtract(ts1, t1))*rho_core
  Wbottom_core=(b/4)*(c[0]+c[3])*np.mean(np.subtract(ts3,t3))*rho_core
  Wside_core=(b/2)*np.mean(h)*np.mean(np.subtract(ts2,t2))*rho_core
  W_wingstruct=Wtop_alum+Wbottom_alum+Wside_alum+Wtop_core+Wbottom_core+Wside_core
  W_fuel_wing=np.mean(h*l)*(b/3)*(2)*rho_fuel
  Bh=sqrt(Z[7]*Z[6])
  W_ht=3.316*((1+(Fw_at_t/Bh))**-2.0)*((L*C[2]/1000)**0.260)*(Z[6]**0.806)
  Wf = C[0] + W_fuel_wing
  Ws = C[1] + W_ht + 2*W_wingstruct
  theta = deltaL_divby_q

  return [Ws, Wf, theta]


def Wing_Mod(Z, LAMBDA):
  c = [0, 0, 0, 0]
  x = [0]*8
  y = [0] * 8
  b=max(2,np.real(sqrt(Z[3]*Z[5])))
  c[0]=2*Z[5]/((1+LAMBDA)*b)
  c[3]=LAMBDA*c[0]
  x[0]=0
  y[0]=0
  x[1]=c[0]
  y[1]=0
  x[6]=(b/2)*np.tan(Z[4]*np.pi/180)
  y[6]=b/2
  x[7]=x[6]+c[3]
  y[7]=b/2
  y[2]=b/6
  x[2]=(x[6]/y[6])*y[2]
  y[4]=b/3
  x[4]=(x[6]/y[6])*y[4]
  x[5]=x[7]+((x[1]-x[7])/y[7])*(y[7]-y[4])
  y[5]=y[4]
  x[3]=x[7]+((x[1]-x[7])/y[7])*(y[7]-y[2])
  y[3]=y[2]
  c[1]=x[3]-x[2]
  c[2]=x[5]-x[4]
  TE_sweep=(np.arctan((x[7]-x[1])/y[7]))*180/np.pi
  Sweep_40=(np.arctan(((x[7]-0.6*(x[7]-x[6]))-0.4*x[1])/y[7]))*180/np.pi

  l=np.multiply([c[i] for i in range(3)], .4*np.cos(Z[4]*np.pi/180))
  k=np.multiply([c[i] for i in range(3)], .6*np.sin((90-TE_sweep)*np.pi/180)/sin((90+TE_sweep-Z[4])*np.pi/180))
  c_box=np.add(l, k)
  D_mx=np.subtract(l, np.multiply(0.407, c_box))


  return c,c_box,Sweep_40,D_mx,b,l

def loads(b,c,Sweep_40,D_mx,L,Izz,Z,E):
  NP=9 #   %----number of panels per halfspan
  n=90
  rn = int(n/NP)

  h=(b/2)/n
  x=np.linspace(0,b/2-h, n)
  x1=np.linspace(h, b/2, n)

  #%----Calculate Mx, Mz, and P----%

  l=np.linspace(0, (b/2)-(b/2)/NP, NP)


  c1mc4 = c[0]-c[3]
  f_all  =np.multiply((3*b/10), np.sqrt(np.subtract(1, ( np.power(x, 2))/(np.power(np.divide(b,2),2)))))
  f1_all =np.multiply((3*b/10), np.sqrt(np.subtract(1, ( np.power(x1, 2))/(np.power(np.divide(b,2),2)))))
  C= c[3] + 2*( (b/2- x)/b )*c1mc4
  C1=c[3] + 2*( (b/2-x1)/b )*c1mc4
  A_Tot: np.ndarray =np.multiply((h/4)*(C+C1), (np.add(f_all, f1_all)))
  Area = np.sum(A_Tot.reshape((NP,rn)), axis=1)
  Spanel = np.multiply((h*(rn)/2), (np.add([C[int(i)] for i in np.linspace(0, n-10, 9)], [C[int(i)] for i in np.linspace(9, n-1, 9)])))

    # % cos, tan, and cos**-1 of Sweep
  cosSweep = np.cos(Sweep_40*pi/180)
  cosInvSweep = 1/cosSweep
  tanCos2Sweep = np.tan(Sweep_40*pi/180)*cosSweep*cosSweep



  p=np.divide(L*Area, sum(Area))
  


  # % Replace T by:
  Tcsp = np.cumsum(p)
  Tsp = Tcsp[-1]
  temp = [0]
  temp = temp +[Tcsp[i] for i in range(len(Tcsp)-1)]
  T = np.subtract(Tsp, temp)
  pl = np.multiply(p, l)
  Tcspl = np.cumsum(pl)
  Tspl = Tcspl[-1]
  Mb = np.multiply(np.subtract( np.subtract(Tspl, Tcspl), np.multiply(l, np.subtract(Tsp, Tcsp)) ),cosInvSweep)


  P=[T[int(i)] for i in np.arange(0,NP-1, int(NP/3))]
  Mx=np.multiply(P, D_mx)
  Mz=[Mb[int(i)] for i in np.arange(0,NP-1, int(NP/3))]

  # %----Calculate Wing Twist due to Bending----%
  I = np.zeros((NP))
  chord=c[3]+ (np.divide(2*(b/2-l), b))*c1mc4
  y = np.zeros((2,9))
  y[0,:]=(l-.4*chord*tanCos2Sweep)*cosInvSweep
  y[1,:]=(l+.6*chord*tanCos2Sweep)*cosInvSweep
  y[1,0]=0
  I[0:int(NP/3)]=sqrt((Izz[0]**2+Izz[1]**2)/2)
  I[int(NP/3):int(2*NP/3)]=sqrt((Izz[1]**2+Izz[2]**2)/2)
  I[int(2*NP/3):int(NP)]=sqrt((Izz[2]**2)/2)

  La=y[0,1:NP]-y[0,0:NP-1]
  La = np.append(0, La)
  Lb=y[1,1:NP]-y[1,0:NP-1]
  Lb=np.append(0, Lb)
  A=T*La**3/(3*E*I)+Mb*La**2./(2*E*I)
  B=T*Lb**3/(3*E*I)+Mb*Lb**2./(2*E*I)
  Slope_A=T*La**2./(2*E*I)+Mb*La/(E*I)
  Slope_B=T*Lb**2./(2*E*I)+Mb*Lb/(E*I)
  for i in range(NP-1):
    Slope_A[i+1]=Slope_A[i]+Slope_A[i+1]
    Slope_B[i+1]=Slope_B[i]+Slope_B[i+1]
    A[i+1]=A[i]+Slope_A[i]*La[i+1]+A[i+1]
    B[i+1]=B[i]+Slope_B[i]*Lb[i+1]+B[i+1]

  bend_twist=((B-A)/chord)*180/pi
  for i in range(1, len(bend_twist)):
    if bend_twist[i]<bend_twist[i-1]:
        bend_twist[i]=bend_twist[i-1]

  return P,Mz,Mx,bend_twist,Spanel


def SBJ_opt4(x, y):
  # SBJ_constraint_weight
  # Z = [tc,h,M,ARw,LAMBDAw,Sref,Sht,ARht]
  Z= [x[1], user.h, user.M, x[2], x[3], x[4], x[5], x[6]]
  t= np.divide(x[7:16], 12.)#convert to feet
  ts= np.divide(x[16:], 12.);#convert to feet
  LAMBDA= x[10]
  L=x[0]

  # %----Inputs----%
  # if isinstance(t, np.ndarray):
  #   t=t/12 #convert to feet
  #   ts=ts/12 #convert to feet
  # else:
  #   t=np.array(t)/12 #convert to feet
  #   ts=np.array(ts)/12 #convert to feet

  t1=t[0:3] 
  t2=t[3:6]
  t3=t[6:9]
  ts1=ts[0:3]
  ts2=ts[3:6]
  ts3=ts[6:9]
  G=4000000*144
  E=10600000*144
  nu=0.3

  # %----Inputs----%

  beta=.9
  [c,c_box,Sweep_40,D_mx,b,a]=Wing_Mod(Z,LAMBDA)
  teq1=((t1**3)/4+(3*t1)*(ts1-t1/2)**2)**(1/3)
  teq2=((t2**3)/4+(3*t2)*(ts2-t2/2)**2)**(1/3)
  teq3=((t3**3)/4+(3*t3)*(ts3-t3/2)**2)**(1/3)
  l=0.6*c_box
  h=Z[0]*(beta*np.array(c[0:3]))-(0.5)*(ts1+ts3)
  A_top=(0.5)*(t1*l)+(1/6)*(t2*h)
  A_bottom=(0.5)*(t3*l)+(1/6)*(t2*h)
  Y_bar=h*(2*A_top)/(2*A_top+2*A_bottom)
  Izz=2*A_top*(h-Y_bar)**2+2*A_bottom*(-Y_bar)**2
  P,Mz,Mx,_,_ =loads(b,c,Sweep_40,D_mx,L,Izz,Z,E)

  sig_1=Mz*(0.95*h-Y_bar)/Izz
  sig_2=Mz*(h-Y_bar)/Izz
  sig_3=sig_1
  sig_4=Mz*(0.05*h-Y_bar)/Izz
  sig_5=Mz*(-Y_bar)/Izz
  sig_6=sig_4
  q=Mx/(2*l*h)

  # %-----THIS SECTION COMPUTES THE TOTAL WEIGHT OF A/C-----%

  k=6.09375


  # %----Point 1----%
  T1=P*(l-a)/l
  tau1_T=T1/(h*t2)
  tau1=q/t2+tau1_T
  sig_eq1=sqrt(sig_1**2+3*tau1**2)
  sig_cr1=((pi**2)*E*4/(12*(1-nu**2)))*(teq2/(0.95*h))**2
  tau_cr1=((pi**2)*E*5.5/(12*(1-nu**2)))*(teq2/(0.95*h))**2
  G = np.zeros((72))
  G[0:3]=k*sig_eq1
  G[3:6]=k*(((sig_1)/sig_cr1)+(tau1/tau_cr1)**2)
  G[6:9]=k*((-(sig_1)/sig_cr1)+(tau1/tau_cr1)**2)


  # %----Point 2----%

  tau2=q/t1
  sig_eq2=sqrt(sig_2**2+3*tau2**2)
  sig_cr2=((pi**2)*E*4/(12*(1-nu**2)))*(teq1/l)**2
  tau_cr2=((pi**2)*E*5.5/(12*(1-nu**2)))*(teq1/l)**2
  G[9:12]=k*sig_eq2
  G[12:15]=k*(((sig_2)/sig_cr2)+(tau2/tau_cr2)**2)
  G[15:18]=k*((-(sig_2)/sig_cr2)+(tau2/tau_cr2)**2)

  # %----Point 3----%

  T2=P*a/l
  tau3_T=-T2/(h*t2)
  tau3=q/t2+tau3_T
  sig_eq3=sqrt(sig_3**2+3*tau3**2)
  sig_cr3=sig_cr1
  tau_cr3=tau_cr1
  G[18:21]=k*sig_eq3
  G[21:24]=k*(((sig_3)/sig_cr3)+(tau3/tau_cr3)**2)
  G[24:27]=k*(((-sig_3)/sig_cr3)+(tau3/tau_cr3)**2)

  # %----Point 4----%

  tau4=-q/t2+tau1_T
  sig_eq4=sqrt(sig_4**2+3*tau4**2)
  G[27:30]=k*sig_eq4

  # %----Point 5----%

  tau5=q/t3
  sig_eq5=sqrt(sig_5**2+3*tau5**2)
  sig_cr5=((pi**2)*E*4/(12*(1-nu**2)))*(teq3/l)**2
  tau_cr5=((pi**2)*E*5.5/(12*(1-nu**2)))*(teq3/l)**2
  G[30:33]=k*sig_eq5
  G[33:36]=k*(((sig_5)/sig_cr5)+(tau5/tau_cr5)**2)
  G[36:39]=k*(((-sig_5)/sig_cr5)+(tau5/tau_cr5)**2)

  # %----Point 6----%

  tau6=-q/t2+tau3_T
  sig_eq6=sqrt(sig_6**2+3*tau6**2)
  G[39:42]=k*sig_eq6

  # %-----Constraints-----%

  Sig_C=65000*144
  Sig_T=65000*144


  G1 = np.zeros((72))

  G1[0:3]=((G[0:3])/Sig_C)-1
  G1[54:57]=-(G[0:3])/Sig_C-1
  G1[3:9]=G[3:9]-1

  G1[9:12]=(G[9:12])/Sig_C-1
  G1[57:60]=-(G[9:12])/Sig_C-1
  G1[12:18]=G[12:18]-1

  G1[18:21]=(G[18:21])/Sig_C-1
  G1[60:63]=-(G[18:21])/Sig_C-1
  G1[21:27]=G[21:27]-1

  G1[27:30]=(G[27:30])/Sig_T-1
  G1[63:66]=-(G[27:30])/Sig_T-1

  G1[30:33]=(G[30:33])/Sig_T-1
  G1[66:69]=-(G[30:33])/Sig_T-1
  G1[33:39]=G[33:39]-1

  G1[39:42]=(G[39:42])/Sig_T-1
  G1[69:72]=-(G[39:42])/Sig_T-1

  G1[42:45]=(1/2)*(ts1+ts3)/h-1
  G1[45:48]=t1/(ts1-.1*t1)-1
  G1[48:51]=t2/(ts2-.1*t2)-1
  G1[51:54]=t3/(ts3-.1*t3)-1


  return[0, [xx for xx in G1]]

def polyApprox(S, S_new, flag, S_bound):
  S_norm = []
  S_shifted = []
  Ai = []
  Aij = np.zeros((len(S),len(S)))
  for i in range(len(S)):
    S_norm.append(S_new[int(i/S[i])])
    if S_norm[i]>1.25:
      S_norm[i]=1.25
    elif S_norm[i]<0.75:
        S_norm[i]=0.75
    S_shifted.append(S_norm[i] - 1)
    a = 0.1
    b = a


    if flag[i]==5:
      # CALCULATE POLYNOMIAL COEFFICIENTS (S-ABOUT ORIGIN)
      So=0
      Sl=So-S_bound[i]
      Su=So+S_bound[i]
      Mtx_shifted = np.array([[1, Sl, Sl**2], [1, So, So**2], [1, Su, Su**2]])

      F_bound = np.array([1+(.5*a)**2, 1, 1+(.5*b)**2])
      A = np.linalg.solve(Mtx_shifted, F_bound)
      Ao = A[0]
      Ai.append(A[1])
      Aij[i,i] = A[2]

      # CALCULATE POLYNOMIAL COEFFICIENTS
    else:
      if flag[i] == 0:
        S_shifted.append(0)
      elif flag[i]==3:
        a *= -1.
        b=copy.deepcopy(a)
      elif flag[i]==2:
        b = 2 * a
      elif flag[i] == 4:
        a *= -1
        b = 2*a
      # DETERMINE BOUNDS ON FF DEPENDING ON SLOPE-SHAPE
      #  CALCULATE POLYNOMIAL COEFFICIENTS (S-ABOUT ORIGIN)
      So=0
      Sl=So-S_bound[i]
      Su=So+S_bound[i]
      Mtx_shifted = np.array([[1, Sl, Sl**2], [1, So, So**2], [1, Su, Su**2]])
      F_bound = np.array([1-.5*a, 1, 1+.5*b])
      A = np.linalg.solve(Mtx_shifted, F_bound)
      Ao = A[0]
      Ai.append(A[1])
      Aij[i,i] = A[2]
    
      #  CALCULATE POLYNOMIAL COEFFICIENTS
  R = np.array([[0.2736,    0.3970,    0.8152,    0.9230,    0.1108], 
                [0.4252,    0.4415,    0.6357,    0.7435,    0.1138],
                [0.0329,    0.8856,    0.8390,    0.3657,    0.0019],
                [0.0878,    0.7248,    0.1978,    0.0200,    0.0169],
                [0.8955,    0.4568,    0.8075,    0.9239,    0.2525]])
  
  for i in range(len(S)):
    for j in range(i+1, len(S)):
      Aij[i, j] = Aij[i,i] * R[i,j]
      Aij[j, i] = Aij[i, j]

  S_shifted = np.array(S_shifted)
  
  FF = Ao + np.dot(Ai, (np.transpose(S_shifted))) + (1/2)*np.dot(np.dot(S_shifted, Aij), np.transpose(S_shifted))

  return FF



def Basic_MDO():
  #  Variables setup
  v = {}
  V: List[variableData] = []
  names = ["u", "v", "a", "b", "u", "w", "a", "b"]
  spi = [1,1,1,1,2,2,2,2]
  links = [2, None, 2, 2, 1, None, 1, 1]
  coupling_t = [COUPLING_TYPE.SHARED, COUPLING_TYPE.UNCOUPLED, COUPLING_TYPE.FEEDFORWARD,
  COUPLING_TYPE.FEEDBACK, COUPLING_TYPE.SHARED, COUPLING_TYPE.UNCOUPLED,
   COUPLING_TYPE.FEEDBACK, COUPLING_TYPE.FEEDFORWARD]
  lb = [0.]*8
  ub = [10.]*8
  bl = [1.]*8
  scaling = np.subtract(ub,lb)

  # Variables dictionary with subproblems link
  for i in range(8):
    v[f"var{i+1}"] = {"index": i+1,
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

  for i in range(8):
    V.append(variableData(**v[f"var{i+1}"]))

  # Analyses setup; construct disciplinary analyses
  DA1: process = DA(inputs=[V[0], V[1], V[3]],
  outputs=[V[2]],
  blackbox=A1,
  links=2,
  coupling_type=COUPLING_TYPE.FEEDFORWARD
  )

  DA2: process = DA(inputs=[V[4], V[5], V[6]],
  outputs=[V[7]],
  blackbox=A2,
  links=1,
  coupling_type=COUPLING_TYPE.FEEDFORWARD
  )

  # MDA setup; construct subproblems MDA
  sp1_MDA: process = MDA(nAnalyses=1, analyses = [DA1], variables=[V[0], V[1], V[3]], responses=[V[2]])
  sp2_MDA: process = MDA(nAnalyses=1, analyses = [DA2], variables=[V[4], V[5], V[6]], responses=[V[7]])

  # Construct the coordinator
  coord = ADMM(beta = 1.3,
  nsp=2,
  budget = 50,
  index_of_master_SP=1,
  display = True,
  scaling = 0.1,
  mode = "serial",
  M_update_scheme= w_scheme.MEDIAN
  )

  # Construct subproblems
  sp1 = SubProblem(nv = 3,
  index = 1,
  vars = [V[0], V[1], V[3]],
  resps = [V[2]],
  is_main=1,
  analysis= sp1_MDA,
  coordination=coord,
  opt=opt1,
  fmin_nop=np.inf,
  budget=20,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST)

  sp2 = SubProblem(nv = 3,
  index = 2,
  vars = [V[4], V[5], V[6]],
  resps = [V[7]],
  is_main=0,
  analysis= sp2_MDA,
  coordination=coord,
  opt=opt2,
  fmin_nop=np.inf,
  budget=20,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST
  )

  # Construct MDO workflow
  MDAO: MDO = MDO(
  Architecture = MDO_ARCHITECTURE.IDF,
  Coordinator = coord,
  subProblems = [sp1, sp2],
  variables = V,
  responses = [V[2], V[7]],
  fmin = np.inf,
  hmin = np.inf,
  display = True,
  inc_stop = 1E-9,
  stop = "Iteration budget exhausted",
  tab_inc = [],
  noprogress_stop = 100
  )

  # Run the MDO problem
  out = MDAO.run()

  print(f'------Run_Summary------')
  print(MDAO.stop)
  print(f'q = {MDAO.Coordinator.q}')
  for i in MDAO.Coordinator.master_vars:
    print(f'{i.name}_{i.sp_index} = {i.value}')

  fmin = 0
  hmax = -inf
  for j in range(len(MDAO.subProblems)):
    print(f'SP_{MDAO.subProblems[j].index}: fmin= {MDAO.subProblems[j].MDA_process.getOutputs()}, hmin= {MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] , MDAO.subProblems[j].MDA_process.getOutputs())[1]}')
    fmin += sum(MDAO.subProblems[j].MDA_process.getOutputs())
    hmin= MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] , MDAO.subProblems[j].MDA_process.getOutputs())[1]
    if max(hmin) > hmax: 
      hmax = max(hmin) 
  print(f'P_main: fmin= {fmin}, hmax= {hmax}')
  print(f'Final obj value of the main problem: \n {fmin}')

def speedReducer():
  #  Variables setup
  f1min = 722
  f1max = 5408
  f2min = 184
  f2max = 506
  f3min = 942
  f3max = 1369
  #
  v = {}
  V: List[variableData] = []
  s  = COUPLING_TYPE.SHARED
  ff = COUPLING_TYPE.FEEDFORWARD
  fb = COUPLING_TYPE.FEEDBACK
  un = COUPLING_TYPE.UNCOUPLED
  dum = COUPLING_TYPE.DUMMY


  names = ["x1", "x2", "x3", "f1",   "x1", "x2", "x3", "x4", "x6", "f2",   "x1", "x2", "x3", "x5", "x7", "f3"]
  spi =   [   1,    1,    1,		1,		  2,		2,		2,		2,		2,		2,      3,    3,		3,		3,    3,	  3]
  links = [[2,3],[2,3],[2,3],   None,  [1,3],[1,3],[1,3], None, None,    None,  [1,2],[1,2],[1,2], None, None,    None]
  lb =    [2.6 ,  0.7 ,  17., 722.,  2.6 ,  0.7,  17.,  7.3,  2.9, 184.,   2.6 ,  0.7,  17.,  7.3,   5.,942.]
  ub =    [3.6 ,  0.8 ,  28.,5408.,  3.6 ,  0.8,  28.,  8.3,  3.9, 506.,   3.6 ,  0.8 , 28.,  8.3,  5.5,1369.]
  bl =    np.divide(np.subtract(ub, lb), 2.)
  
  coupling_t = \
          [ s,      s,		s,		un,		s,		s,		s,		un,		un,	 un,   s,    s,    s,   un,    un,    un]
 
  scaling = [10.] * 16

  # Variables dictionary with subproblems link
  for i in range(16):
    v[f"var{i+1}"] = {"index": i+1,
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

  for i in range(16):
    V.append(variableData(**v[f"var{i+1}"]))

  # Analyses setup; construct disciplinary analyses
  DA1: process = DA(inputs=[V[0], V[1], V[2]],
  outputs=[V[3]],
  blackbox=SR_A1,
  links=[4],
  coupling_type=COUPLING_TYPE.FEEDFORWARD)
  
  DA2: process = DA(inputs=[V[4], V[5], V[6], V[7], V[8]],
  outputs=[V[9]],
  blackbox=SR_A2,
  links=[4],
  coupling_type=COUPLING_TYPE.FEEDFORWARD
  )

  DA3: process = DA(inputs=[V[10], V[11], V[12], V[13], V[14]],
  outputs=[V[15]],
  blackbox=SR_A3,
  links=[4],
  coupling_type=COUPLING_TYPE.FEEDFORWARD
  )

  # MDA setup; construct subproblems MDA
  sp1_MDA: process = MDA(nAnalyses=1, analyses = [DA1], variables=[V[0], V[1], V[2]], responses=[V[3]])
  sp2_MDA: process = MDA(nAnalyses=1, analyses = [DA2], variables=[V[4], V[5], V[6], V[7], V[8]], responses=[V[9]])
  sp3_MDA: process = MDA(nAnalyses=1, analyses = [DA3], variables=[V[10], V[11], V[12], V[13], V[14]], responses=[V[15]])

  # Construct the coordinator
  coord = ADMM(beta = 1.3,
  nsp=4,
  budget = 50,
  index_of_master_SP=1,
  display = True,
  scaling = 0.1,
  mode = "serial",
  M_update_scheme= w_scheme.MEDIAN
  )

  # Construct subproblems
  sp1 = SubProblem(nv = 3,
  index = 1,
  vars = [V[0], V[1], V[2]],
  resps = [V[3]],
  is_main=1,
  analysis= sp1_MDA,
  coordination=coord,
  opt=SR_opt1,
  fmin_nop=np.inf,
  budget=20,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST)

  sp2 = SubProblem(nv = 5,
  index = 2,
  vars = [V[4], V[5], V[6], V[7], V[8]],
  resps = [V[9]],
  is_main=0,
  analysis= sp2_MDA,
  coordination=coord,
  opt=SR_opt2,
  fmin_nop=np.inf,
  budget=20,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST
  )

  sp3 = SubProblem(nv = 5,
  index = 3,
  vars = [V[10], V[11], V[12], V[13], V[14]],
  resps = [V[15]],
  is_main=0,
  analysis= sp3_MDA,
  coordination=coord,
  opt=SR_opt3,
  fmin_nop=np.inf,
  budget=20,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST)

# Construct MDO workflow
  MDAO: MDO = MDO(
  Architecture = MDO_ARCHITECTURE.IDF,
  Coordinator = coord,
  subProblems = [sp1, sp2, sp3],
  variables = V,
  responses = [V[3], V[9], V[15]],
  fmin = np.inf,
  hmin = np.inf,
  display = True,
  inc_stop = 1E-9,
  stop = "Iteration budget exhausted",
  tab_inc = [],
  noprogress_stop = 100
  )


# Run the MDO problem
  out = MDAO.run()

  print(f'------Run_Summary------')
  print(MDAO.stop)
  print(f'q = {MDAO.Coordinator.q}')
  for i in MDAO.Coordinator.master_vars:
    print(f'{i.name}_{i.sp_index} = {i.value}')
  fmin = 0
  hmax = -inf
  for j in range(len(MDAO.subProblems)):
    print(f'SP_{MDAO.subProblems[j].index}: fmin= {MDAO.subProblems[j].MDA_process.getOutputs()}, hmin= {MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] , [])[1]}')
    fmin += sum(MDAO.subProblems[j].MDA_process.getOutputs())
    hmin= MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] , [])[1]
    if max(hmin) > hmax: 
      hmax = max(hmin) 
  print(f'P_main: fmin= {fmin}, hmax= {hmax}')
  print(f'Final obj value of the main problem: \n {fmin}')

def geometric_programming():
  v = {}
  V: List[variableData] = []
  s  = COUPLING_TYPE.SHARED
  ff = COUPLING_TYPE.FEEDFORWARD
  fb = COUPLING_TYPE.FEEDBACK
  un = COUPLING_TYPE.UNCOUPLED
  dum = COUPLING_TYPE.DUMMY


  names = ["z3", "z4", "z5", "z6",   "z7", "z3", "z8", "z9", "z10", "z11",   "z6", "z11", "z12", "z13", "z14", "dd"]
  spi =   [   1,    1,    1,		1,		  1,		2,		2,		2,		2,		  2,      3,     3,		  3,		 3,     3, 1]
  links = [   2, None, None,    3,   None,    1, None, None, None,      3,      1,     2,  None,  None,  None, None]
  coupling_t = \
          [ fb,    un,	 un,	 fb,		 un,	 ff,	 un,	 un,	 un,	    s,     ff,     s,    un,    un,    un, ff]
 
  lb =    [1e-6]*15
  ub =    [1e6]*15
  bl =    [1.]*15
  scaling = [9.e5] * 15

  lb.append(15)
  ub.append(18)
  bl.append(GP_A1([1]*5))
  scaling.append(1)

  # Variables dictionary with subproblems link
  for i in range(16):
    v[f"var{i+1}"] = {"index": i+1,
    "sp_index": spi[i],
    f"name": names[i],
    "dim": 1,
    "coupling_type": coupling_t[i],
    "link": links[i],
    "baseline": bl[i],
    "scaling": scaling[i],
    "lb": lb[i],
    "value": bl[i],
    "ub": ub[i]}

  for i in range(16):
    V.append(variableData(**v[f"var{i+1}"]))
  
  # Analyses setup; construct disciplinary analyses
  DA1: process = DA(inputs=[V[0], V[1], V[2], V[3], V[4]],
  outputs=[V[15]],
  blackbox=GP_A1,
  links=[2, 3],
  coupling_type=COUPLING_TYPE.FEEDBACK)
  
  DA2: process = DA(inputs=[V[6], V[7], V[8], V[9]],
  outputs=[V[5]],
  blackbox=GP_A2,
  links=[1, 3],
  coupling_type=COUPLING_TYPE.FEEDFORWARD
  )

  DA3: process = DA(inputs=[V[11], V[12], V[13], V[14]],
  outputs=[V[10]],
  blackbox=GP_A3,
  links=[1, 2],
  coupling_type=COUPLING_TYPE.FEEDFORWARD
  )

  # MDA setup; construct subproblems MDA
  sp1_MDA: process = MDA(nAnalyses=1, analyses = [DA1], variables=[V[0], V[1], V[2], V[3], V[4]], responses=[V[15]])
  sp2_MDA: process = MDA(nAnalyses=1, analyses = [DA2], variables=[V[6], V[7], V[8], V[9]], responses=[V[5]])
  sp3_MDA: process = MDA(nAnalyses=1, analyses = [DA3], variables=[V[11], V[12], V[13], V[14]], responses=[V[10]])

  # Construct the coordinator
  coord = ADMM(beta = 1.3,
  nsp=3,
  budget = 200,
  index_of_master_SP=1,
  display = True,
  scaling = 1.,
  mode = "serial",
  M_update_scheme= w_scheme.MEDIAN
  )

  # Construct subproblems
  sp1 = SubProblem(nv = 5,
  index = 1,
  vars = [V[0], V[1], V[2], V[3], V[4]],
  resps = [V[15]],
  is_main=1,
  analysis= sp1_MDA,
  coordination=coord,
  opt=GP_opt1,
  fmin_nop=np.inf,
  budget=5,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST,
  freal=15.)

  sp2 = SubProblem(nv = 4,
  index = 2,
  vars = [V[6], V[7], V[8], V[9]],
  resps = [V[5]],
  is_main=0,
  analysis= sp2_MDA,
  coordination=coord,
  opt=GP_opt2,
  fmin_nop=np.inf,
  budget=20,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST
  )

  sp3 = SubProblem(nv = 4,
  index = 3,
  vars = [V[11], V[12], V[13], V[14]],
  resps = [V[10]],
  is_main=0,
  analysis= sp3_MDA,
  coordination=coord,
  opt=GP_opt3,
  fmin_nop=np.inf,
  budget=20,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST)

# Construct MDO workflow
  MDAO: MDO = MDO(
  Architecture = MDO_ARCHITECTURE.IDF,
  Coordinator = coord,
  subProblems = [sp1, sp2, sp3],
  variables = V,
  responses = [V[5], V[10], V[15]],
  fmin = np.inf,
  hmin = np.inf,
  display = True,
  inc_stop = 1E-9,
  stop = "Iteration budget exhausted",
  tab_inc = [],
  noprogress_stop = 100
  )


# Run the MDO problem
  out = MDAO.run()
  print(f'------Run_Summary------')
  print(MDAO.stop)
  print(f'q = {MDAO.Coordinator.q}')
  xsp2 = []
  xsp3 = []
  index = np.argmax(MDAO.Coordinator.q)
  for i in MDAO.Coordinator.master_vars:
    if i.sp_index == MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[1,index]-1].link:
      xsp2.append(i.value)
    if i.sp_index == MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[0,index]-1].link:
      xsp3.append(i.value)
    print(f'{i.name}_{i.sp_index} = {i.value}')
  fmin = 0
  hmax = -inf
  for j in range(len(MDAO.subProblems)):
    print(f'SP_{MDAO.subProblems[j].index}: fmin= {MDAO.subProblems[j].MDA_process.getOutputs()}, hmin= {MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] , [])[1]}')
    if MDAO.subProblems[j].is_main:
      fmin = MDAO.subProblems[j].MDA_process.getOutputs()
      hmin= MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] , [])[1]
      if max(hmin) > hmax: 
        hmax = max(hmin) 
  print(f'P_main: fmin= {fmin}, hmax= {hmax}')
  print(f'Final obj value of the main problem: \n {fmin}')

  # Checking the impact of swapping z11_2 and z11_3 on the feasibility of SP2 and SP3, respectively
  
  temp = copy.deepcopy(xsp3[1])
  xsp3[1] = xsp2[-1]
  xsp2[-1] = temp
  print(f'For {MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[0,index]-1].name}_'
    f'{MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[0,index]-1].link} to '
      f'{MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[1,index]-1].name}_'
    f'{MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[1,index]-1].link}, using {MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[0,index]-1].name}_'
    f'{MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[0,index]-1].link}' 
    f' will make h_sp{MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[1,index]-1].link} = {GP_opt2(xsp2, 0)[1]}')
  print(f'For {MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[0,index]-1].name}_'
    f'{MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[0,index]-1].link} to '
      f'{MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[1,index]-1].name}_'
    f'{MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[1,index]-1].link}, using {MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[1,index]-1].name}_'
    f'{MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[1,index]-1].link}' 
    f' will make h_sp{MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[0,index]-1].link} = {GP_opt3(xsp3, 0)[1]}')

def SBJ():
  """ Supersonic Business Jet aircraft conceptual design"""
  # Variables definition
  v = {}
  V: List[variableData] = []
  s  = COUPLING_TYPE.SHARED
  ff = COUPLING_TYPE.FEEDFORWARD
  fb = COUPLING_TYPE.FEEDBACK
  un = COUPLING_TYPE.UNCOUPLED
  dum = COUPLING_TYPE.DUMMY


  names = ["SFC", "We", "LD", "Ws",   "Wf", "Wt",   "D", "T", "We", "SFC",   "ESF", "ESF", "Wt", "theta", "tc", "ARw", "LAMBDAw", "Sref", "Sht", "ARht", "LAMBDAht", "Lw", "Lht", "D", "LD", "L",   "L", "tc", "ARw", "LAMBDAw", "Sref", "Sht", "ARht", "Ws", "Wf", "theta", "lambda", "t", "ts"]
  spi =   [   1,    1,    1,		1,		  1,		1,		  2,	 2,		 2,		  2,       2,     3,		3,		   3,    3,     3,         3,      3,     3,      3,          3,    3,     3,   3,    3,   3,     4,    4,     4,         4,      4,     4,      4,    4,    4,       4,        4,   4,    4]
  links = [   2,    2,    3,    4,      4,    3,      3, None,   1,     1,       3,     2,    1,       4,    4,     4,         4,      4,     4,      4,       None, None,  None,   2,    1,   4,     3,    3,     3,         3,      3,     3,      3,    1,    1,       3,     None,None, None]
  coupling_t = \
          [ fb,    fb,	 fb,	 fb,		 fb,	 ff,	   fb,	 un,	 ff,	 ff,      ff,    fb,   fb,      fb,    s,     s,         s,      s,     s,      s,         un,   un,    un,  ff,   ff,  ff,    fb,   s,     s,         s,      s,     s,      s,   ff,   ff,      ff,       un,  un,   un]
  lb =    [1  ,   100,  0.1, 5000,   5000, 5000,   1000,  0.1,  100,    1,     0.5,   0.5, 5000,     0.2, 0.01,   2.5,        40.,    200.,    50,    2.5,         40, 0.01,     1,1000,  0.1,5000,  5000, 0.01,   2.5,        40,    200,    50,    2.5, 5000, 5000,     0.2,      0.1, 0.1,  0.1] 
  ub =    [4  , 30000,   10,100000,100000,100000, 70000,   1.,30000,    4,     1.5,   1.5,100000,     50,  0.1,    8.,        70.,    800., 148.9,    8.5,         70,  0.2,   3.5,70000,  10,100000,100000,0.1,     8,        70,    800, 148.9,    8.5,100000,100000,    50,      0.4, 4.0,  9.0]
  dim = [1]*39
  dim[37] = 9
  dim[38] = 9

  fstar = 33600
  frealistic = 30000
  altitude = 55000
  Mach = 1.4
  bl = [1.]*55
  scaling = np.subtract(ub,lb)

  Qscaling = []

  # Variables dictionary with subproblems link
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
        Qscaling.append(.1/scaling[i] if .1/scaling[i] != np.inf and .1/scaling[i] != np.nan else 1.)
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
      Qscaling.append(.1/scaling[i] if .1/scaling[i] != np.inf and .1/scaling[i] != np.nan else 1.)

  for i in range(len(v)):
    V.append(variableData(**v[f"var{i+1}"]))

  # Analyses setup; construct disciplinary analyses
  V1 = copy.deepcopy([V[0], V[1], V[2], V[3], V[4]])
  Y1 = copy.deepcopy([V[5]])
  DA1: process = DA(inputs=V1,
  outputs=Y1,
  blackbox=SBJ_A1,
  links=[2,3,4],
  coupling_type=COUPLING_TYPE.FEEDFORWARD)
  
  V2 = copy.deepcopy([V[6], V[7]])
  Y2 = copy.deepcopy([V[8], V[9], V[10]])
  DA2: process = DA(inputs=V2,
  outputs=Y2,
  blackbox=SBJ_A2,
  links=[1,3],
  coupling_type=[COUPLING_TYPE.FEEDFORWARD, COUPLING_TYPE.FEEDFORWARD, COUPLING_TYPE.FEEDFORWARD]
  )
                    # "ESF", "Wt",  "theta", "tc", "ARw", "LAMBDAw", "Sref", "Sht", "ARht", "LAMBDAht", "Lw", "Lht", 
  V3 = copy.deepcopy([V[11], V[12], V[13], V[14], V[15], V[16],     V[17],  V[18], V[19],       V[20], V[21], V[22]])
  Y3 = copy.deepcopy([V[23], V[24], V[25]])
  DA3: process = DA(inputs=V3,
  outputs=Y3,
  blackbox=SBJ_A3,
  links=[1,2,4],
  coupling_type=[COUPLING_TYPE.FEEDFORWARD, COUPLING_TYPE.FEEDFORWARD, COUPLING_TYPE.FEEDFORWARD]
  )

  V4 = copy.deepcopy([V[26], V[27], V[28], V[29], V[30], V[31], V[32], V[36]]+V[37:len(V)])
  Y4 = copy.deepcopy([V[33], V[34], V[35]])
  DA4: process = DA(inputs=V4,
  outputs=Y4,
  blackbox=SBJ_A4,
  links=[1,3],
  coupling_type=[COUPLING_TYPE.FEEDFORWARD, COUPLING_TYPE.FEEDFORWARD, COUPLING_TYPE.FEEDFORWARD]
  )

# MDA setup; construct subproblems MDA
  sp1_MDA: process = MDA(nAnalyses=1, analyses = [DA1], variables=V1, responses=Y1)
  sp2_MDA: process = MDA(nAnalyses=1, analyses = [DA2], variables=V2, responses=Y2)
  sp3_MDA: process = MDA(nAnalyses=1, analyses = [DA3], variables=V3, responses=Y3)
  sp4_MDA: process = MDA(nAnalyses=1, analyses = [DA4], variables=V4, responses=Y4)

  # Construct the coordinator
  coord = ADMM(beta = 1.3,
  nsp=4,
  budget = 500,
  index_of_master_SP=1,
  display = True,
  scaling = Qscaling,
  mode = "serial",
  M_update_scheme= w_scheme.MEDIAN
  )

  # Construct subproblems
  sp1 = SubProblem(nv = len(V1),
  index = 1,
  vars = V1,
  resps = Y1,
  is_main=1,
  analysis= sp1_MDA,
  coordination=coord,
  opt=SBJ_opt1,
  fmin_nop=np.inf,
  budget=20,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST,
  freal=30000)

  sp2 = SubProblem(nv = len(V2),
  index = 2,
  vars = V2,
  resps = Y2,
  is_main=0,
  analysis= sp2_MDA,
  coordination=coord,
  opt=SBJ_opt2,
  fmin_nop=np.inf,
  budget=20,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST
  )

  sp3 = SubProblem(nv = len(V3),
  index = 3,
  vars = V3,
  resps = Y3,
  is_main=0,
  analysis= sp3_MDA,
  coordination=coord,
  opt=SBJ_opt3,
  fmin_nop=np.inf,
  budget=20,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST)

  sp4 = SubProblem(nv = len(V4),
  index = 4,
  vars = V4,
  resps = Y4,
  is_main=0,
  analysis= sp4_MDA,
  coordination=coord,
  opt=SBJ_opt4,
  fmin_nop=np.inf,
  budget=20,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST)

  # Construct MDO workflow
  MDAO: MDO = MDO(
  Architecture = MDO_ARCHITECTURE.IDF,
  Coordinator = coord,
  subProblems = [sp1, sp2, sp3, sp4],
  variables = V,
  responses = Y1+Y2+Y3+Y4,
  fmin = np.inf,
  hmin = np.inf,
  display = True,
  inc_stop = 1E-9,
  stop = "Iteration budget exhausted",
  tab_inc = [],
  noprogress_stop = 500
  )

  user = USER
  # Altitude
  user.h = 55000
  # Mach number
  user.M = 1.4
  
  out = MDAO.run()
  



if __name__ == "__main__":
  #TODO: Feature: Add more realistic analytical test problems
  #TODO: Feature: Add realistic multi-physics MDO problems that require using open-source physics-based simulation tools
  #TODO: Feature: Move the MDO test functions and BM problems to NOBM package and prepare the DMDO package to be published on PyPi
  #TODO: Feature: Develop a simple UI widget that facilitates simple MDO setup using the compact table or N2-chart
  #TODO: Feature: Import RAF library once the latter is published on PYPI.com
  #FIXME: Bug: Add user and technical documentation 
  #FIXME: Bug: Enable the output report generation that summarizes the MDO history and final results
  Basic_MDO()
  # speedReducer()
  # geometric_programming()
  # SBJ()















