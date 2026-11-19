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
from genericpath import isfile
import json
from typing import Any, Dict, List

from .da import DA
from ..core.protocols import Process_data


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
    # Each analysis input gets an independent copy of the variable with the same index (the last one wins, as in
    # the former all-pairs loop).
    by_index = {}
    for v in self.variables:
      by_index[v.index] = v
    for i in range(self.nAnalyses):
      inputs = self.analyses[i].inputs
      for k in range(len(inputs)):
        src = by_index.get(inputs[k].index)
        if src is not None:
          inputs[k] = src.clone() if hasattr(src, "clone") else copy.deepcopy(src)
      self.analyses[i].run()

  def validation(self, vType: int):
    self.term_status = []
    for i in range(len(self.term_critteria)):
      if self.term_type[i] == vType:
          self.term_status.append(self.term_critteria[i])

  def setInputs(self, values: List[Any]):
    # run() copies every variable it hands to an analysis, so a shallow list is enough here.
    self.variables = list(values)

  def getOutputs(self):
    out = []
    if self.responses is None:
      return [None]
    for i in range(len(self.responses)):
      out.append(self.responses[i].value)
    return out


