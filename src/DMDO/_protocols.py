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

from typing import List, Callable, Protocol
from dataclasses import dataclass, field
from .variables import variableData

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
