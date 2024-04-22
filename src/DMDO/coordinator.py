
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
from .variables import *

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
          l = len(value)
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
          raise Exception(IOError, f'The variable {self.master_vars[self._linker[0, i]-1].name} linked between subproblems index #{self._linker[0, i]} and #{self._linker[1, i]} does not have the same dimension!')
      else:
        if self.master_vars[self._linker[0, i]-1].coupling_type == COUPLING_TYPE.FEEDFORWARD:
          self.update_all_cond_linked(self._linker[0, i]-1)
    for i in range(self._linker.shape[1]):
      dim1: int = self.master_vars[self._linker[0, i]-1].dim
      dim2: int = self.master_vars[self._linker[1, i]-1].dim

      # Check whether linked variables have same dimension
      if dim1 != dim2 and self.master_vars[self._linker[0, i]-1].cond_on is None:
          raise Exception(IOError, f'The variable {self.master_vars[self._linker[0, i]-1].name} linked between subproblems index #{self._linker[0, i]} and #{self._linker[1, i]} does not have the same dimension!')
      else:
        if self.master_vars[self._linker[1, i]-1].coupling_type == COUPLING_TYPE.FEEDFORWARD:
          self.update_all_cond_linked(self._linker[1, i]-1)

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
            warning("The inconsistency scaling factors are defined in a list which has a different size from the master variables vector! The minimum value of the provided scaling list will be used to scale the inconsistency vector.")
      else:
        raise Exception(IOError, "Master variables vector have to be non-empty to calculate inconsistencies!")
    self.qold = copy.deepcopy(q_temp)

  def update_master_vector_val(self, vars: List[variableData], resps: List[variableData], sets):
    if self.master_vars:
      for i in range(len(vars)):
        self.master_vars[vars[i].index-1] = copy.deepcopy(vars[i])
        typ = vars[i].type[0].lower() if vars[i].dim == 1 else vars[i].type[0][0].lower()
        # self.master_vars[vars[i].index-1].value = vars[i].value #if (typ != "c" and typ != "d") else sets[vars[i].set].index(vars[i].value)
      if resps is not None:
        for i in range(len(resps)):
          self.master_vars[resps[i].index-1] = copy.deepcopy(resps[i])
          typ = resps[i].type[0].lower() if resps[i].dim == 1 else resps[i].type[0][0].lower()
          # self.master_vars[resps[i].index-1].value = resps[i].value #if (typ != "c" and typ != "d") else sets[resps[i].set].index(resps[i].value)
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
        increase_w.append(len(self.q) * (self.q_stall[i] and tc[i]))
    elif self.M_update_scheme == w_scheme.NORMAL:
      increase_w = self.q_stall
    elif self.M_update_scheme == w_scheme.RANK:
      temp = np.argsort(self.q)
      rank = np.empty_like(temp).tolist()
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
