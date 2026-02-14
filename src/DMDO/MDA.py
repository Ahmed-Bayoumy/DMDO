# ------------------------------------------------------------------------------------#
#  Distributed Multidisciplinary Design Optimization - DMDO                           #
#                                                                                     #
#  Author: Ahmed H. Bayoumy                                                           #
#  email: ahmed.bayoumy@mail.mcgill.ca                                                #
#                                                                                     #
#  This program is free software: you can redistribute it and/or modify it under the  #
#  terms of the GNU General Public License as published by the Free Software   #
#  Foundation, either version 3 of the License, or (at your option) any later         #
#  version.                                                                           #
#                                                                                     #
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY    #
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A    #
#  PARTICULAR PURPOSE.  See the GNU General Public License for more details.   #
#                                                                                     #
#  You should have received a copy of the GNU General Public License along     #
#  with this program. If not, see <http://www.gnu.org/licenses/>.                     #
#                                                                                     #
#  You can find information on simple_mads at                                         #
#  https://github.com/Ahmed-Bayoumy/DMDO                                              #
# ------------------------------------------------------------------------------------#



import copy
from dataclasses import dataclass
from genericpath import isfile
import json
from typing import Any, Dict, List

from .DA import DA
from .MDO import Process_data


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
    if self.responses is None:
      return [None]
    for i in range(len(self.responses)):
      out.append(self.responses[i].value)
    return out


