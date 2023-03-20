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
import csv
import shutil

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
  CONSTANT = auto()

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
  set: str = None
  cond_on: str = None


  def __sub__(self, other):
    if type(other)!=variableData:
      raise IOError(f'The variables data dunder subtraction from {self.name} expects a variable data object as an input but {type(other)} is invoked!')
    if self.dim > 1:
      if other.dim == 1:
        raise IOError(f'The variables data dunder subtraction from {self.name} expects a vector of variables but a scalar is invoked!')
      if self.dim> other.dim:
        dif = self.dim-other.dim
        other.value += [0]*dif
        other.dim = self.dim
      elif self.dim<other.dim:
        dif = other.dim-self.dim
        val: list = copy.deepcopy(self.value)
        val += [0]*dif
        return np.subtract(val, other.value)


    return np.subtract(self.value, other.value)

  def __add__(self, other):
    return np.add(self.value, other.value)

  def __mul__(self, other):
    return np.multiply(self.value, other.value)

  def __truediv__(self, other):
    if isinstance(other, variableData):
      return np.divide(self.value, other.value)
    else:
      return np.divide(self.value, other)
  
  def __update__(self, other):
    if type(other)!=variableData and type(self.value) != type(other):
      raise IOError(f'The variables data dunder equality method of {self.name} expects a variable data object as an input or variable values with the same type of {self.name}!')
    if isinstance(other, variableData):
      self =  copy.deepcopy(other)
    elif isinstance(other, list) or isinstance(other, np.ndarray):
      l = self.dim
      if len(other) != l and self.cond_on is None:
        raise IOError(f'The feedback of {self.name} does not have the same size. That variable size is not conditional though!')
      if len(other) > l:
        dif = len(other)-l
        for io in range(l, len(other)):
          self.value.append(other[io])
        self.baseline += [self.baseline[l-1]]*dif
        self.lb += [self.lb[l-1]]*dif
        self.ub += [self.ub[l-1]]*dif
        self.scaling += [self.scaling[l-1]]*dif
        self.type += [self.type[l-1]]*dif
      elif len(other) < l:
        dif = l - len(other)
        self.value = self.value[:-dif]
        self.baseline = self.baseline[:-dif]
        self.lb = self.lb[:-dif]
        self.ub = self.ub[:-dif]
        self.scaling = self.scaling[:-dif]
        self.type = self.type[:-dif]
      else:
        self.value = other
      self.dim = len(other)
    elif isinstance(other, int) or isinstance(other, float) or isinstance(other, str):
      self.value = other
    else:
      raise IOError(f'The variables data dunder equality method expects an object with the same type a list of values or a scalar numerical/textual value!')


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
    if len(self.outputs) > 1 and (isinstance(values, list) or isinstance(values, np.ndarray)):
      o = 0
      for i in range(len(self.outputs)):
        self.outputs[i].__update__(values[o])
        o += 1
    elif len(self.outputs) == 1 and (isinstance(values, list) or isinstance(values, np.ndarray)):
      if self.outputs[0].dim>1:
        if self.outputs[0].dim != len(values):
          raise IOError(f'The size of the analysis outputs does not match the subproblem #{self.index}!')
        self.outputs[0].value = copy.deepcopy(values)
      else:
        self.outputs[0].value = values[0]
    else:
      raise RuntimeError(f'The number of expected response outputs of DA{self.index} associated with {self.blackbox} is {len(self.outputs)} however the analysis returned only a single value!')


  def getInputsList(self):
    v = []
    for i in range(len(self.inputs)):
      if isinstance(self.inputs[i].value, list):
        for j in range(len(self.inputs[i].value)):
          v.append(self.inputs[i].value[j])
      else:
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
      raise IOError("The size of the introduced batched variables inconsistency vector doesn't match the total number of links available!")
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
          l = len(vt)
          dif = dim - dt
          vt = vt + [value[l+ii] for ii in range(dif)]
          tt += [tt[l-1]]*dif
          lbt += [lbt[l-1]]*dif
          ubt += [ubt[l-1]]*dif
          blt += [blt[l-1]]*dif
          st += [st[l-1]]*dif
          # TODO: Enhance the set logic
          if isinstance(sett, list):
            sett += [sett[l-1]]*dif
        elif dim < dt:
          l = len(value)
          dif = dt - dim
          vt = vt[:len(vt)-dif]
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
          raise Exception(IOError, f'The variable {self.master_vars[self._linker[0, i]-1].name} linked between subproblems index #{self._linker[0, i]} and #{self._linker[1, i]} does not have the same dimension!')
      else:
        if self.master_vars[self._linker[0, i]-1].coupling_type == COUPLING_TYPE.FEEDFORWARD:
          self.update_all_cond_linked(self._linker[0, i]-1)

  def calc_inconsistency(self):
    if self.save_q_in_out:
      global eps_qio
    q_temp : np.ndarray = np.zeros([0,0])
    tl0: List = []
    tl1: List = []
    self.update_conditional_vars()
    for i in range(self._linker.shape[1]):
      if self.master_vars:
        # TODO: Add a sanity check early on to ensure that linked parameters has the same type and linked to the same set if they were of discrete type and may be add a dunder methed to handle all the necessary equality checks
        #  Check if linked parameters have same type
        type1: Any = self.master_vars[self._linker[0, i]-1].type
        type2: Any = self.master_vars[self._linker[1, i]-1].type
        val1: Any = self.master_vars[self._linker[0, i]-1].value
        val2: Any = self.master_vars[self._linker[1, i]-1].value
        dim1: int = self.master_vars[self._linker[0, i]-1].dim
        dim2: int = self.master_vars[self._linker[1, i]-1].dim
        set1_name: Any = self.master_vars[self._linker[1, i]-1].set
        set2_name: Any = self.master_vars[self._linker[1, i]-1].set

        #COMPLETED: The variables coupling relationships and their size dependencies need a review
        qtest: List = []
        for ik in range(dim1):
          t1 = type1[ik] if isinstance(type1, list) else type1
          t2 = type2[ik] if isinstance(type2, list) else type2
          sn1 = set1_name[ik] if isinstance(set1_name, list) else set1_name
          sn2 = set2_name[ik] if isinstance(set2_name, list) else set2_name
          tl0.append(self._linker[0, i])
          tl1.append(self._linker[1, i])
          if t1[0] != t2[0]:
            raise Exception(IOError, f'The variable {self.master_vars[self._linker[0, i]-1].name} linked between subproblems index #{self._linker[0, i]} and #{self._linker[1, i]} does not have the same type(s)!')
          if t1[0].lower() != "r" and t1[0].lower() != "i" and t1[0].lower() != "d" and t1[0].lower() != "c":
            raise Exception(IOError, f'The variable {self.master_vars[self._linker[0, i]-1].name} linked between subproblems index #{self._linker[0, i]} and #{self._linker[1, i]} has unknown type(s)!')
          if t1[0].lower() == "c":
            v1 = val1[ik] if isinstance(val1, list) else val1
            v2 = val2[ik] if isinstance(val2, list) else val2
            if isinstance(val1, list) and isinstance(val2, list):
              q_temp = np.append(q_temp, sum([not x for x in np.equal(val1,val2)]))
            else:
              q_temp = np.append(q_temp, 0 if val1 == val2 else 1)
            continue
          else:
            v1 = val1[ik] if isinstance(val1, list) else val1
            v2 = val2[ik] if isinstance(val2, list) else val2
          if t1[0].lower() == "d":
            i1 = self.sets[sn1].index(v1)
            i2 = self.sets[sn2].index(v2)
          if  (isinstance(self.scaling, list) and len(self.scaling) == len(self.master_vars)):
            qscale = np.multiply(np.add(self.scaling[self._linker[0, i]-1], self.scaling[self._linker[1, i]-1]), 0.5)
            if t1[0].lower() == "d":
              q_temp = np.append(q_temp, np.multiply(subtract(i1, i2), qscale))
            else:
              q_temp = np.append(q_temp, np.multiply(subtract(v1, v2), qscale))
          elif isinstance(self.scaling, float) or  isinstance(self.scaling, int):
            if t1[0].lower() == "d":
              q_temp = np.append(q_temp, np.multiply(subtract(i1, i2), self.scaling))
            else:
              q_temp = np.append(q_temp, np.multiply(subtract(v1, v2), self.scaling))
          else:
            if t1[0].lower() == "d":
              q_temp = np.append(q_temp, np.multiply(subtract(i1, i2), min(self.scaling)))
            else:
              q_temp = np.append(q_temp, np.multiply(subtract(v1, v2), min(self.scaling)))
            warning("The inconsistency scaling factors are defined in a list which has a different size from the master variables vector! The minimum value of the provided scaling list will be used to scale the inconsistency vector.")
        
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
          #   warning("The inconsistency scaling factors are defined in a list which has a different size from the master variables vector! The minimum value of the provided scaling list will be used to scale the inconsistency vector.")
      else:
        raise Exception(IOError, "Master variables vector have to be non-empty to calculate inconsistencies!")
    # qb: list = self.batch_q(q_temp)
    # self.modify_multipliers(qb, wb, vb)
    self.q = copy.deepcopy(q_temp)
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
        # TODO: Add a sanity check early on to ensure that linked parameters has the same type and linked to the same set if they were of discrete type and may be add a dunder methed to handle all the necessary equality checks
        #  Check if linked parameters have same type
        type1: Any = self.master_vars_old[self._linker[0, i]-1].type
        type2: Any = self.master_vars_old[self._linker[1, i]-1].type
        val1: Any = self.master_vars_old[self._linker[0, i]-1].value
        val2: Any = self.master_vars_old[self._linker[1, i]-1].value
        dim1: int = self.master_vars_old[self._linker[0, i]-1].dim
        dim2: int = self.master_vars_old[self._linker[1, i]-1].dim
        set1_name: Any = self.master_vars_old[self._linker[1, i]-1].set
        set2_name: Any = self.master_vars_old[self._linker[1, i]-1].set


        
        # Check whether linked variables have same dimension
        qtest: List = []
        for ik in range(dim1):
          t1 = type1[ik] if isinstance(type1, list) else type1
          t2 = type2[ik] if isinstance(type2, list) else type2
          sn1 = set1_name[ik] if isinstance(set1_name, list) else set1_name
          sn2 = set2_name[ik] if isinstance(set2_name, list) else set2_name
          tl0.append(self._linker[0, i])
          tl1.append(self._linker[1, i])
          if t1 != t2:
            raise Exception(IOError, f'The variable {self.master_vars_old[self._linker[0, i]-1].name} linked between subproblems index #{self._linker[0, i]} and #{self._linker[1, i]} does not have the same type(s)!')
          if t1[0].lower() != "r" and t1[0].lower() != "i" and t1[0].lower() != "d" and t1[0].lower() != "c":
            raise Exception(IOError, f'The variable {self.master_vars_old[self._linker[0, i]-1].name} linked between subproblems index #{self._linker[0, i]} and #{self._linker[1, i]} has unknown type(s)!')
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
              q_temp = np.append(q_temp, np.multiply(subtract(i1, i2), qscale))
            else:
              q_temp = np.append(q_temp, np.multiply(subtract(v1, v2), qscale))
          elif isinstance(self.scaling, float) or  isinstance(self.scaling, int):
            if t1[0].lower() == "d":
              q_temp = np.append(q_temp, np.multiply(subtract(i1, i2), self.scaling))
            else:
              q_temp = np.append(q_temp, np.multiply(subtract(v1, v2), self.scaling))
          else:
            if t1[0].lower() == "d":
              q_temp = np.append(q_temp, np.multiply(subtract(i1, i2), min(self.scaling)))
            else:
              q_temp = np.append(q_temp, np.multiply(subtract(v1, v2), min(self.scaling)))
            warning("The inconsistency scaling factors are defined in a list which has a different size from the master variables vector! The minimum value of the provided scaling list will be used to scale the inconsistency vector.")
      else:
        raise Exception(IOError, "Master variables vector have to be non-empty to calculate inconsistencies!")
    self.qold = copy.deepcopy(q_temp)

  def update_master_vector_val(self, vars: List[variableData], resps: List[variableData], sets):
    if self.master_vars:
      for i in range(len(vars)):
        self.master_vars[vars[i].index-1] = copy.deepcopy(vars[i])
        typ = vars[i].type[0].lower() if vars[i].dim == 1 else vars[i].type[0][0].lower()
        self.master_vars[vars[i].index-1].value = vars[i].value #if (typ != "c" and typ != "d") else sets[vars[i].set].index(vars[i].value)
      for i in range(len(resps)):
        self.master_vars[resps[i].index-1] = copy.deepcopy(resps[i])
        typ = resps[i].type[0].lower() if resps[i].dim == 1 else resps[i].type[0][0].lower()
        self.master_vars[resps[i].index-1].value = resps[i].value #if (typ != "c" and typ != "d") else sets[resps[i].set].index(resps[i].value)
    else:
      raise Exception(IOError, "Master variables vector have to be non-empty to calculate inconsistencies!")

  def update_master_vector(self, vars: List[variableData], resps: List[variableData]):
    if self.master_vars:
      for i in range(len(vars)):
        self.master_vars[vars[i].index-1] = copy.deepcopy(vars[i])
      for i in range(len(resps)):
        self.master_vars[resps[i].index-1] = copy.deepcopy(resps[i])
    else:
      raise Exception(IOError, "Master variables vector have to be non-empty to calculate inconsistencies!")


  def calc_penalty(self, q_indices):
    if len(self.w) != len(self.v):
      raise RuntimeError('The multipliers vectors w and v have inconsistent size!')
    if len(self.q) != len(self.w):
      raise RuntimeError("The variables inconsistency vector has different size from the multipliers vectoe w and v!")
    phi = np.add(np.multiply(self.v, self.q), np.multiply(np.multiply(self.w, self.w), np.multiply(self.q, self.q)))
    phib = self.batch_q(phi)
    #COMPLETE: Sum relevant components of q to accelerate the convergence of variables consistency
    s = 0
    for i in q_indices:
      if isinstance(i, list):
        s += sum(phi[i])
      else:
        s += phi[i]
    self.phi = s

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

  def update_multipliers(self, iter):
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

    self.q_stall = np.greater(np.abs(self.q), self.gamma*np.abs(self.qold))
    increase_w = []
    wold = copy.deepcopy(self.w)
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

    if False and iter > 0:
      dq = (np.abs(self.q) - np.abs(self.qold))
      # Forget gate
      f =self.sigmoid(dq+np.abs(np.subtract(self.w,wold)))
      # Input gate for the next iteration
      for i in range(len(self.w)):
        self.w[i] *= f[i]
        self.w[i] += np.tanh(dq[i]+np.abs(np.subtract(self.w[i],wold[i])))*f[i] 

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

@dataclass
class SubProblem(partitionedProblemData):
  # Constructor
  def __init__(self, nv, index, vars, resps, is_main, analysis, coordination, opt, fmin_nop, budget, display, psize, pupdate, scipy=None, freal=None, tol=1E-12, solver='OMADS', sets=None):
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
    self.sets = sets
    

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
  
  def modify_cond_vars(self, V: variableData):
     # TODO: Add a check to support conditionality on other variable properties
    for i in range(len(self.vars)):
      if self.vars[i].cond_on != None:
        for v in V:
          if self.index == v.sp_index and v.name == self.vars[i].name and v.cond_on != None:
            if self.vars[i].dim > v.dim:
              temp = self.vars[i].value[:v.dim]
              self.vars[i].__update__(temp)
            elif self.vars[i].dim < v.dim:
              temp = copy.deepcopy(self.vars[i].value)
              temp += [0]*(v.dim-self.vars[i].dim)
              self.vars[i].__update__(temp)

  def set_variables_value(self, vlist: List, clist: List=None):
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
      elif self.vars[i].coupling_type == COUPLING_TYPE.CONSTANT:
        if self.vars[i].dim > 1:
          for ik in range(self.vars[i].dim):
            self.vars[i].value[ik] = clist[kc]
            self.vars[i].baseline[ik] = clist[kc]
            kc += 1
        else:
          self.vars[i].value = clist[kc]
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
    

  def initialize_IL_res_file(self, file):
    if not os.path.exists(file):
      mode = 'x'
    else:
      mode = 'w'
    with open(file, mode=mode) as csv_file:
      keys = ['Iter#', 'qmax', 'Obj', 'x', 'y', 'const']
      writer = csv.DictWriter(csv_file, fieldnames=keys)
      writer.writeheader()    # add column names in the CSV file
  
  def prepare_post(self, file):
    ht = os.path.split(file)
    name = file.split('.')[0]
    ext = file.split('.')[1]
    pd = os.path.join(ht[0], f'{name}_post')
    pdsb = os.path.join(pd, f'SP_{self.index}')
    pfo = os.path.join(pdsb, f'SB_{self.index}_{self.iter}.out')
    if not os.path.exists(pdsb):
      os.mkdir(pdsb)
    self.postDir = pdsb
    self.Il_file = pfo
    self.initialize_IL_res_file(pfo)
    
  def Add_IL_res_row(self, r: Dict):
    with open(self.Il_file, mode='a') as csv_file:
      keys = ['Iter#', 'qmax', 'Obj', 'x', 'y', 'const']
      writer = csv.DictWriter(csv_file, fieldnames=keys)
      writer.writerow(r)    # add column names in the CSV file
  
  def evaluate(self, vlist: List[float]=None, clist: List[float]=None):
    if self.coord.save_q_in_out:
      global eps_fio
    # If no variables were provided use existing value of the variables of the current subproblem (might happen during initialization)
    if vlist is None:
      v: List = self.get_design_vars()
      vlist = self.get_list_vars(v)
    
    self.set_variables_value(vlist, clist)
    self.MDA_process.setInputs(self.vars)
    wb, vb = self.coord.batch_multipliers()
    self.MDA_process.run()
    y = self.MDA_process.getOutputs()

    fun = self.opt(vlist, y, clist)
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
    qb = self.coord.batch_q(self.coord.q)
    qtemp = []
    for i in range(len(qb)):
      if isinstance(qb[i], list):
        qtemp.append(max(qb[i]))
      else:
        qtemp.append(qb[i])
    self.coord.modify_multipliers(qb, wb, vb)
    if self.Il_file is not None:
      self.Add_IL_res_row({'Iter#': self.iter, 'qmax':max(qtemp), 'Obj': fun[0], 'x': vlist, 'y': y, 'const': clist})
    
    # TODO: Fix the local indices to select from a batched q list
    q_indices: List = self.getLocalIndices()
    self.coord.calc_penalty(q_indices)
    # TODO: change the name of this routine
    self.coord.update_master_vector_val(self.vars, self.MDA_process.responses, self.sets)

    if self.realistic_objective:
      con.append(y-self.frealistic)
    if self.solver == "OMADS":
      return [fun[0]+self.coord.phi, con]
    else:
      return fun[0]+self.coord.phi+max(max(con),0)**2

  def solve(self, v, w, file: str = None, iter: int = None):
    if file is not None:
      self.iter = iter
      self.prepare_post(file + f'_{iter}')
    self.set_dependent_baseline(self.coord.master_vars)
    self.coord.v = copy.deepcopy(v)
    self.coord.w = copy.deepcopy(w)
    bl = self.get_list_vars(self.get_design_vars())
    if self.solver == 'OMADS' or self.solver != 'scipy':
      eval = {"blackbox": self.evaluate}
      if self.sets is None:
        self.sets = {}
      param = {"baseline": bl,
                  "lb": self.get_list_vars_lb(self.get_design_vars()),
                  "ub": self.get_list_vars_ub(self.get_design_vars()),
                  "var_names": self.get_list_vars_names(self.get_design_vars()),
                  "var_type": self.get_vars_types(self.get_design_vars()),
                  "var_sets": self.sets,
                  "scaling": self.get_design_vars_scaling(self.get_design_vars()),
                  "post_dir": "./post",
                  "constants": self.get_list_constant_updates(self.get_design_vars()),
                  "constants_name": self.get_list_const_names(self.get_design_vars())}
      pinit = min(max(self.tol, self.psize), 1)
      options = {
        "seed": 0,
        "budget": 2*self.budget*len(self.vars),
        "tol": max(pinit/1000, self.tol),
        "psize_init": pinit,
        "display": self.display,
        "opportunistic": False,
        "check_cache": True,
        "store_cache": True,
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

      self.evaluate(out["xmin"], self.get_list_constant_updates(self.get_design_vars()))
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
      if isinstance(vc[i].value, list):
        for j in range(len(vc[i].value)):
          con.append(float(vc[i].value[j]-vc[i].ub[j]))
      else:
        con.append(float(vc[i].value-vc[i].ub))

    for i in range(len(vc)):
      if isinstance(vc[i].value, list):
        for j in range(len(vc[i].value)):
          con.append(float(vc[i].lb[j]-vc[i].value[j]))
      else:
        con.append(float(vc[i].lb-vc[i].value))
    return con

  def set_dependent_baseline(self, vars: List[variableData]):
    for i in range(len(vars)):
      if vars[i].sp_index == self.index:
        if vars[i].coupling_type == COUPLING_TYPE.FEEDFORWARD or vars[i].coupling_type == COUPLING_TYPE.SHARED or vars[i].coupling_type == COUPLING_TYPE.CONSTANT:
          for j in range(len(self.vars)):
            if self.vars[j].name == vars[i].name:
              self.vars[j].baseline = vars[i].value

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
        if isinstance(vars[i].baseline, list):
          for j in range(len(vars[i].baseline)):
            v.append(vars[i].baseline[j] if (vars[i].type[j].lower() == 'r' or vars[i].type[j].lower() == 'i') else vars[i].baseline[j])
        else:
          v.append(vars[i].baseline if (vars[i].type[0].lower() == 'r' or vars[i].type[0].lower() == 'i') else vars[i].baseline)
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
            if vars[i].type[j] == VAR_TYPE.CONTINUOUS:
              v.append("R")
            elif vars[i].type[j] == VAR_TYPE.INTEGER:
              v.append("I")
            elif vars[i].type[j] == VAR_TYPE.CATEGORICAL:
              v.append("C")
            else:
              v.append(vars[i].type[j])
        else:
          if vars[i].type == VAR_TYPE.CONTINUOUS:
            v.append("R")
          elif vars[i].type == VAR_TYPE.INTEGER:
            v.append("I")
          elif vars[i].type == VAR_TYPE.CATEGORICAL:
            v.append("C")
          else:
            v.append(vars[i].type)
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
      mv_clone = copy.deepcopy(self.Coordinator.master_vars[i])
      mvold_clone = copy.deepcopy(self.Coordinator.master_vars_old[i])
      mv_type = mv_clone.type[0][0].lower() if isinstance(mv_clone.type, list) else mv_clone.type[0].lower()

      if mv_clone.dim > 1:
        for k in range(mv_clone.dim):
          if mv_type == 'c' or mv_type == 'd':
            mv_clone.value[k] = self.sets[mv_clone.set].index(self.Coordinator.master_vars[i].value[k])
            mvold_clone.value[k] = self.sets[mvold_clone.set].index(self.Coordinator.master_vars_old[i].value[k])
      else:
        if mv_type == 'c' or mv_type == 'd':
            mv_clone.value = self.sets[mv_clone.set].index(self.Coordinator.master_vars[i].value)
            mvold_clone.value = self.sets[mvold_clone.set].index(self.Coordinator.master_vars_old[i].value)
      if mv_type =='c':
        if mv_clone.value == mvold_clone.value:
          dx.append(0)
        else:
          dx.append(1)
      else:
        dx.append(mv_clone - mvold_clone)
      x.append(self.Coordinator.master_vars[i].value)
      xold.append(self.Coordinator.master_vars_old[i].value)
    DX: list = []
    for i in range(len(dx)):
      if isinstance(dx[i], list) or isinstance(dx[i], np.ndarray):
        for j in range(len(dx[i])):
          DX.append(dx[i][j])
      else:
        DX.append(dx[i])
    return np.linalg.norm(DX, 2)

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

  def get_weights_size(self, sp_i: int) -> int:
    link1 = self.subProblems[sp_i].coord._linker[0]
    out = 0
    for i in range(len(link1)):
      out += self.subProblems[sp_i].coord.master_vars[link1[i]-1].dim
    
    return out
  
  def initialize_OL_res_file(self, file):
    if not os.path.exists(file):
      mode = 'x'
    else:
      mode = 'w'
    with open(file, mode=mode) as csv_file:
      keys = ['Iter#', 'qmax', 'Obj', 'dx', 'max(w)', 'Coupling_with_qmax', 'xmin']
      writer = csv.DictWriter(csv_file, fieldnames=keys)
      writer.writeheader()    # add column names in the CSV file
  
  def prepare_post(self, file):
    ht = os.path.split(file)
    name = file.split('.')[0]
    ext = file.split('.')[1]
    pd = os.path.join(ht[0], f'{name}_post')
    pfo = os.path.join(pd, 'OL_hist.out')
    if os.path.exists(pd):
      shutil.rmtree(pd)
    os.mkdir(pd)
    self.postDir = pd
    self.Ol_file = pfo
    self.initialize_OL_res_file(pfo)
    
  def Add_OL_res_row(self, r: Dict):
    with open(self.Ol_file, mode='a') as csv_file:
      keys = ['Iter#', 'qmax', 'Obj', 'dx', 'max(w)', 'Coupling_with_qmax', 'xmin']
      writer = csv.DictWriter(csv_file, fieldnames=keys)
      writer.writerow(r)    # add column names in the CSV file
  
  def run(self, file):
    global eps_fio, eps_qio
    # Note: once you run MDAO, the data stored in eps_fio and eps_qio shall be deleted. It is recommended to store such data to a different variable before running another MDAO
    eps_fio = []
    eps_qio = []
    self.prepare_post(file)


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
          self.Coordinator.v = [0.] * self.get_weights_size(s)
          self.Coordinator.w = [1.] * self.get_weights_size(s)
        else:
          self.subProblems[s].coord.master_vars = copy.deepcopy(self.Coordinator.master_vars)
        self.subProblems[s].modify_cond_vars(self.Coordinator.master_vars)
      
        out_sp = self.subProblems[s].solve(self.Coordinator.v, self.Coordinator.w, file=file, iter=iter)
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
        qb = self.Coordinator.batch_q(self.Coordinator.q)
        ql: list = []
        for i in range(len(qb)):
          if isinstance(qb[i], list):
            ql.append(max(np.abs(qb[i])))
          else:
            ql.append(abs(qb[i]))

        index = np.argmax(ql)
        print(f'Highest inconsistency : {self.Coordinator.master_vars[self.Coordinator.extended_linker[0,index]-1].name}_'
          f'{self.Coordinator.master_vars[self.Coordinator.extended_linker[0,index]-1].link} to '
            f'{self.Coordinator.master_vars[self.Coordinator.extended_linker[1,index]-1].name}_'
          f'{self.Coordinator.master_vars[self.Coordinator.extended_linker[1,index]-1].link}')
      """ Write OL results to the file"""
      self.Add_OL_res_row({'Iter#': f'{iter}', 'qmax': f'{np.max(np.abs(self.Coordinator.q))}', 'Obj': f'{self.fmin}', 'dx': f'{dx}', 'max(w)': f'{np.max(self.Coordinator.w)}', 'Coupling_with_qmax': f'{self.Coordinator.master_vars[self.Coordinator.extended_linker[0,index]-1].name}_'
          f'{self.Coordinator.master_vars[self.Coordinator.extended_linker[0,index]-1].link} to '
            f'{self.Coordinator.master_vars[self.Coordinator.extended_linker[1,index]-1].name}_'
          f'{self.Coordinator.master_vars[self.Coordinator.extended_linker[1,index]-1].link}', 'xmin': f'{[self.Coordinator.master_vars[i].value for i in range(len(self.Coordinator.master_vars))]}'})
      """ Update LM and PM """
      self.Coordinator.update_multipliers(iter)

      """ Stopping criteria """
      self.tab_inc.append(np.max(np.abs(self.Coordinator.q)))
      stop: bool = self.check_termination_critt(iter)
      

      if stop:
        break
      self.eps_qio = copy.deepcopy(eps_qio)
      self.eps_fio = copy.deepcopy(eps_fio)
    print(f'xmin={[self.Coordinator.master_vars[i].value for i in range(len(self.Coordinator.master_vars))]}')
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
  Sets: Dict = None

  def get_varSets(self):
    if "sets" in self.data or "Sets" in self.data:
      self.Sets = self.data["Sets"]

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
      elif vin[i][3] == 'CP':
        ct.append(COUPLING_TYPE.CONSTANT)
      else:
        raise IOError(f'Unrecognized coupling type {vin[i][3]} is introduced to the variables dictionary at this key {i}')
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
    
    ub: List = [vin[i][6] for i in vin]
    dim: List = [vin[i][7] for i in vin]
    vtype: List = [vin[i][8]  if len(vin[i])>8 else "R" for i in vin]
    vsets: List = [(vin[i][8].split('_')[1:][0] if len(vin[i][8])>1 else None)  if len(vin[i])>8 else None for i in vin]

    bl: List = [self.data["Sets"][vsets[list(vin.keys()).index(i)]].index(vin[i][5]) if (vtype[list(vin.keys()).index(i)] == 'c' or vtype[list(vin.keys()).index(i)] == 'i') else vin[i][5] for i in vin]

    scaling = np.subtract(ub,lb)
    self.Qscaling = []
    for i in range(len(names)):
      cond_on = None
      if isinstance(dim[i], str):
        conp = dim[i].split("_")[1]
        for k in range(len(names)):
          if spi[i] == spi[k] and names[k] == vin[conp][0]:
            dim[i] = bl[k]
            cond_on = vin[conp][0]
      if dim[i] > 1:
        v[f"var{i+1}"] = {"index": i+1,
        "sp_index": spi[i],
        f"name": names[i],
        "dim": dim[i],
        "value": [bl[i]]*dim[i],
        "coupling_type": coupling_t[i],
        "link": links[i],
        "baseline": [bl[i]]*dim[i],
        "scaling": [scaling[i]]*dim[i],
        "lb": [lb[i]]*dim[i],
        "value": [bl[i]]*dim[i],
        "ub": [ub[i]]*dim[i],
        "type": [vtype[i]]*dim[i],
        "set": vsets[i],
        "cond_on": cond_on}
        self.Qscaling.append(.1/scaling[i] if .1/scaling[i] != np.inf and .1/scaling[i] != np.nan else 1.)
      else:
        v[f"var{i+1}"] = {"index": i+1,
        "sp_index": spi[i],
        f"name": names[i],
        "dim": 1,
        "value": bl[i],
        "coupling_type": coupling_t[i],
        "link": links[i],
        "baseline": bl[i],
        "scaling": scaling[i],
        "lb": lb[i],
        "value": bl[i],
        "ub": ub[i],
        "type": vtype[i],
        "set": vsets[i],
        "cond_on": cond_on}
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
        scipy= SP[i]["scipy"] if "scipy" in SP[i] else None,
        sets=self.Sets
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
      noprogress_stop= MDAO["noprogress_stop"] if "noprogress_stop" in MDAO and isinstance(MDAO["noprogress_stop"], int) else 100
    )
    self.MDAO.sets = self.data["Sets"]
  
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
    self.get_varSets()
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

  MDAO.run(file)

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
  #COMPLETED: Bug: Add user and technical documentation 
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
