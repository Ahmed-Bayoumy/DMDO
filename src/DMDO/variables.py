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

from ._globals import *

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

  def extend_noncat(self, other):
    """ TODO: Extend the other values vector size to the self values vector size"""
    return other
  
  def shrink_noncat(self, other):
    """ TODO: Shrink the other values vector size to the self values vector size"""
    return other
  
  def extend_cat(self, other):
    """ TODO: Extend the other values vector size to the self values vector size"""
    return other
  
  def shrink_cat(self, other):
    """ TODO: Shrink the other values vector size to the self values vector size"""
    return other


  def __sub__(self, other):
    if type(other)!=variableData:
      raise IOError(f'The variables data dunder subtraction from {self.name} expects a variable data object as an input but {type(other)} is invoked!')
    if self.dim > 1:
      valo: list = copy.deepcopy(other.value)
      val: list = copy.deepcopy(self.value)

      if other.dim == 1:
        raise IOError(f'The variables data dunder subtraction from {self.name} expects a vector of variables but a scalar is invoked!')
      if self.dim> other.dim:
        dif = self.dim-other.dim
        valo += [0]*dif
        return np.subtract(val, valo)
      elif self.dim<other.dim:
        dif = other.dim-self.dim
        val += [0]*dif
        return np.subtract(val, valo)
      
      return np.subtract(val, valo)

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
    # if type(other)!=variableData and type(self.value) != type(other):
    #   raise IOError(f'The variables data dunder equality method of {self.name} expects a variable data object as an input or variable values with the same type of {self.name}!')
    if isinstance(other, variableData):
      self =  copy.deepcopy(other)
    elif isinstance(other, list) or isinstance(other, np.ndarray):
      l = self.dim
      if len(other) != l and self.cond_on is None:
        raise IOError(f'The feedback of {self.name} does not have the same size. That variable size is not conditional though!')
      if len(other) > l:
        dif = len(other)-l
        # for io in range(l, len(other)):
        self.value = copy.deepcopy(other)
        self.baseline += [self.baseline[l-1]]*dif
        self.lb += [self.lb[l-1]]*dif
        self.ub += [self.ub[l-1]]*dif
        self.scaling += [self.scaling[l-1]]*dif
        self.type += [self.type[l-1]]*dif
      elif len(other) < l:
        dif = l - len(other)
        self.value = copy.deepcopy(other)
        self.baseline = self.baseline[:-dif]
        self.lb = self.lb[:-dif]
        self.ub = self.ub[:-dif]
        self.scaling = self.scaling[:-dif]
        self.type = self.type[:-dif]
      else:
        self.value = copy.deepcopy(other)
      self.dim = len(other)
    elif isinstance(other, int) or isinstance(other, float) or isinstance(other, str):
      self.value = other
    else:
      raise IOError(f'The variables data dunder equality method expects an object with the same type a list of values or a scalar numerical/textual value!')

