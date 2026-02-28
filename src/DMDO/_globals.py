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

from enum import Enum, auto
from dataclasses import dataclass
import numpy as np

class USER:
  """ Custom class """

class PSIZE_UPDATE(Enum):
  DEFAULT = auto()
  SUCCESS = auto()
  MAX = auto()
  LAST = auto()

@dataclass
class double_precision:
    decimals: int
    """ Decimal precision control """

    def truncate(self, value: float) -> float:
        if abs(value) != np.inf:
            multiplier = 10 ** self.decimals
            return int(value * multiplier) / multiplier if value != np.inf else np.inf
        else:
            return value

class MSG_TYPE(Enum):
  DEBUG = auto()
  WARNING = auto()
  ERROR = auto()
  INFO = auto()
  CRITICAL = auto()

class VAR_TYPE(Enum):
  REAL = auto()
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

eps_qio = []
eps_fio = []

user = USER
VERSION_NUMBER = "2601"
def check_space(string: str) -> int:
  if string.isspace():
    return string.count(" ")
  return 0

def replace_space_with_comma(string: str) -> str:
  if check_space(string=string) > 1:
    return ",".join(string.split())
  else:
    return string