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
from typing import Any, List, Callable, Optional, Dict

import numpy as np

from .variables import variableData
from dataclasses import dataclass, field

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
      raise IOError("Callables are the only evaluator type currently allowed."
                    " Enabling evaluating BB executables is still in progress!")
      # evalerr = False
      # try:
      #   p = subprocess.run(self.blackbox, shell=True, timeout=self.timeout)
      #   if p.returncode != 0:
      #     evalerr = True
      #     logging.error("Evaluation # {self.blackbox} is errored!")
      # except subprocess.TimeoutExpired:
      #   timouterr = True 
      #   logging.error(f'Timeout for {self.blackbox} ({self.timeout}s) expired!')
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
    elif len(self.outputs) == 1 and self.outputs[0] is not None and (isinstance(values, list) \
                                                                     or isinstance(values, np.ndarray)):
      if self.outputs[0].dim>1:
        if self.outputs[0].dim != len(values):
          raise IOError(f'The size of the analysis outputs does not match the subproblem #{self.index}!')
        self.outputs[0].value = copy.deepcopy(values)
      else:
        self.outputs[0].value = values[0]
    else:
      raise RuntimeError(f'The number of expected response outputs of DA{self.index} '
                         f'associated with {self.blackbox} is {len(self.outputs)} '
                         'however the analysis returned only a single value!')


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