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
import csv
from dataclasses import dataclass
from logging import warning
import os
import platform
import time
from typing import Any, Callable, Dict, List
import OMADS
import numpy as np
# import pandas as pd
from scipy.optimize import minimize, Bounds, NonlinearConstraint, BFGS

from ._common import MSG_TYPE, logger
from ._globals import COUPLING_TYPE, PSIZE_UPDATE, VAR_TYPE, eps_fio
from ._protocols import process
from .coordinator import ADMM
from .variables import variableData

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
  conf: Dict = None
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
  fmin: float
  cache: List
  hmin: float
  log: logger = None
  RANDOM_SEED: float = 1234
  update_bl: bool = True
  job: Any = None
  allConf: Dict = None 
  bl: List[float] = None

@dataclass
class SubProblem(partitionedProblemData):
  # Constructor
  def __init__(self, nv, index, vars, resps, is_main, analysis, coordination, opt, fmin_nop, budget, display, scipy=None, \
               psize=1, pupdate=PSIZE_UPDATE.LAST, freal=None, tol=1E-12, solver='OMADS', sets=None, conf = None, \
                realistic_objective=False, log: logger=None):
    self.nv = nv
    self.index = index
    self.vars = copy.deepcopy(vars)
    self.resps = copy.deepcopy(resps)
    self.is_main = is_main
    self.MDA_process = analysis
    self.coord = coordination
    self.opt = opt
    self.optimizer = OMADS.mads.main 
    self.fmin_nop = fmin_nop
    self.budget=budget
    self.display = display
    self.psize = psize
    self.psize_init = pupdate
    self.frealistic = freal
    self.tol = tol
    self.solver = solver
    self.realistic_objective = realistic_objective
    self.fmin = np.inf
    self.hmin = np.inf
    self.cache = []
    self.eval_ID = 0
    if log is not None:
      self.log = log
    if solver == 'mads':
      self.scipy = None
      self.optimizer = OMADS.mads.main
    elif solver == 'poll':
      self.scipy = None
      self.optimizer = OMADS.poll.main
    elif solver == 'search':
      self.scipy = None
      self.optimizer = OMADS.search.main
    elif solver == 'scipy':
      self.scipy = scipy
    else:
      msg = f'Inappropriate solver method definition for subproblem # {self.index}! OMADS will be used.'
      if self.log is not None and self.log.log is not None:
        self.log.log_msg(msg=msg, msg_type=MSG_TYPE.WARNING.value)
      warning(msg)
      self.solver = 'mads'
    self.sets = sets
    self.conf = conf

    if (self.solver == 'scipy' and (scipy is None or "options" not in self.scipy or "method" not in self.scipy)):
      msg = f'Inappropriate definition of the scipy settings for subproblem # {self.index}! '
      'scipy default settings shall be used!'
      if self.log is not None:
        self.log.log_msg(msg=msg, msg_type=MSG_TYPE.WARNING.value)
      warning(msg)
      self.scipy: Dict = {}
      self.scipy["options"] = {'disp': False, 'verbose': 0}
      self.scipy["method"] = 'SLSQP'
      self.scipy["is_con"] = False
      self.scipy["tol"] = 1E-4

  def __getstate__(self):
    # Remove or replace non-picklable attributes
    state = self.__dict__.copy()
    # Example: remove or replace RLock
    if '_lock' in state:
        del state['_lock']
    # Or replace with a placeholder
    # state['_lock'] = None
    return state

  def __setstate__(self, state):
      self.__dict__.update(state)
      # Reconstruct non-picklable objects if needed
      # self._lock = threading.RLock()
        

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
      if self.vars[i].cond_on is not None:
        for v in V:
          if self.index == v.sp_index and v.name == self.vars[i].name and v.cond_on is not None:
            if self.vars[i].dim > v.dim:
              temp = self.vars[i].value[:v.dim]
              self.vars[i].__update__(temp)
            elif self.vars[i].dim < v.dim:
              temp = copy.deepcopy(self.vars[i].value)
              temp += [0]*(v.dim-self.vars[i].dim)
              self.vars[i].__update__(temp)

  def set_variables_value(self, vlist: List, clist: List=None):
    # FIXME: scipy doesn't consider constants list, if defined the routine crashes
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
      elif self.vars[i].coupling_type == COUPLING_TYPE.CONSTANT and clist is not None:
        if self.vars[i].dim > 1:
          for ik in range(self.vars[i].dim):
            self.vars[i].value[ik] = clist[kc]
            self.vars[i].baseline[ik] = clist[kc]
            kc += 1
        else:
          self.vars[i].value = clist[kc]
          self.vars[i].baseline = clist[kc]
          kc += 1

  def set_variables_baseline(self, vlist: List, clist: List=None):
    # FIXME: scipy doesn't consider constants list, if defined the routine crashes
    kv = 0
    kc = 0
    for i in range(len(self.vars)):
      if self.vars[i].coupling_type != COUPLING_TYPE.FEEDFORWARD and self.vars[i].coupling_type != COUPLING_TYPE.CONSTANT:
        if self.vars[i].dim > 1:
          for ik in range(self.vars[i].dim):
            self.vars[i].baseline[ik] = vlist[kv]
            kv += 1
        else:
          self.vars[i].baseline = vlist[kv]
          kv += 1
      elif self.vars[i].coupling_type == COUPLING_TYPE.CONSTANT and clist is not None:
        if self.vars[i].dim > 1:
          for ik in range(self.vars[i].dim):
            self.vars[i].baseline[ik] = clist[kc]
            kc += 1
        else:
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
      if (self.index == self.coord.master_vars[i].sp_index \
          or self.coord.master_vars[i].coupling_type != COUPLING_TYPE.UNCOUPLED) \
            and check and self.coord.master_vars[i].index not in indices2:
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
      keys = [f'{"Time"}', f'{"Iteration #".rjust(30)}', f'{"Evaluation #".rjust(30)}', f'{"Max. inconsistency".rjust(30)}', \
              f'{"Penalty".rjust(30)}', f'{"Objective".rjust(30)}', f'{"Coupling Status".rjust(30)}'] + \
                [f'{f"{x.name}".rjust(30)}' for x in self.vars] + \
                  [f'{f"{y.name}".rjust(30)}' for y in self.resps] 
      writer = csv.DictWriter(csv_file, fieldnames=keys)
      writer.writeheader()    # add column names in the CSV file
  
  def prepare_post(self, file):
    ht = os.path.split(file)
    name = file.split('.')[0]
    pd = os.path.join(ht[0], f'{name}_post')
    pdsb = os.path.join(pd, f'SP_{self.index}')
    pfo = os.path.join(pdsb, f'SP_{self.index}_{self.iter}.out')
    if not os.path.exists(pd):
      os.mkdir(pd)
    if not os.path.exists(pdsb):
      os.mkdir(pdsb)
    self.postDir = pdsb
    self.Il_file = pfo
    self.initialize_IL_res_file(pfo)
    
  def Add_IL_res_row(self, r: Dict):
    with open(self.Il_file, mode='a') as csv_file:
      keys = [f'{"Time"}', f'{"Iteration #".rjust(30)}', f'{"Evaluation #".rjust(30)}', f'{"Max. inconsistency".rjust(30)}', \
              f'{"Penalty".rjust(30)}', f'{"Objective".rjust(30)}', f'{"Coupling Status".rjust(30)}'] + \
                [f'{f"{x.name}".rjust(30)}' for x in self.vars] + [f'{f"{y.name}".rjust(30)}' 
                                                                   for y in self.MDA_process.responses] 
      writer = csv.DictWriter(csv_file, fieldnames=keys)
      writer.writerow(r)    # add column names in the CSV file

  def get_con(self, x):
    return self.evaluate(x, None, "con_only")
  
  def evaluate(self, vlist: List[float]=None, clist: List[float]=None, *argv):  # noqa: C901
    is_conOnly = False
    if argv is not None and len(argv)>0 and argv[0] == "con_only":
      is_conOnly = True
    if self.coord.save_q_in_out:
      global eps_fio
    self.eval_ID += 1
    # If no variables were provided use existing value of the 
    # variables of the current subproblem (might happen during initialization)
    if vlist is None:
      v: List = self.get_design_vars()
      vlist = self.get_list_vars(v)
    
    self.set_variables_value(vlist, clist)
    self.MDA_process.setInputs(self.vars)
    if argv is not None and len(argv) > 0 and argv[0] == "evalOnly":
      self.MDA_process.run()
      y = self.MDA_process.getOutputs()
      fun = self.opt(vlist, y, clist)
      return fun
    wb, vb = self.coord.batch_multipliers()
    if self.MDA_process.responses is not None:
      self.MDA_process.run()
      y = self.MDA_process.getOutputs()
    else:
      y = None

    fun = self.opt(vlist, y, clist)
    
    self.fmin_nop = fun[0]
    con = fun[1]
    if self.MDA_process.responses is not None:
      con = self.get_coupling_vars_diff(con)
    con_coupling = self.get_coupling_vars_diff([0])

    if self.is_main:
      if self.frealistic is not None and self.frealistic != 0.:
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
      curr_time = time.strftime("%H:%M:%S", time.localtime()) 
      hstatus = "Feasible" if max(con_coupling) <= 0 else "Infeasible"
      status = copy.deepcopy(hstatus) if fun[0] != np.inf else "Error"
      keys = [f'{"Time"}', f'{"Iteration #".rjust(30)}', f'{"Evaluation #".rjust(30)}', f'{"Max. inconsistency".rjust(30)}', \
              f'{"Penalty".rjust(30)}', f'{"Objective".rjust(30)}', \
                f'{"Coupling Status".rjust(30)}'] + \
                  [f'{f"{x.name}".rjust(30)}' for x in self.vars] + \
                    [f'{f"{y.name}".rjust(30)}' for y in self.MDA_process.responses] 

      row = {keys[0]: f'{f"{curr_time}"}', 
             keys[1]: f'{f"{self.iter}".rjust(30)}', 
             keys[2]: f'{f"{self.eval_ID}".rjust(30)}', 
             keys[3]: f'{f"{max(np.abs(self.coord.q))}".rjust(30)}', 
             keys[4]: f'{f"{self.coord.phi}".rjust(30)}', 
             keys[5]: f'{f"{fun[0]}".rjust(30)}', 
             keys[6]: f'{f"{status}".rjust(30)}'}
      cv = 0
      lr = len(row)
      for v in vlist:
        row[keys[lr+cv]] = f'{f"{v}".rjust(30)}'
        cv += 1
      cr = 0
      lr = len(row)
      for rs in self.MDA_process.responses:
        row[keys[lr+cr]] = f'{f"{rs.value}".rjust(30)}'
        cr += 1
      self.Add_IL_res_row(row)
    
    # TODO: Fix the local indices to select from a batched q list
    q_indices: List = self.getLocalIndices()
    self.coord.calc_inconsistency()
    self.coord.calc_penalty(q_indices)
    # TODO: change the name of this routine
    self.coord.update_master_vector_val(self.vars, self.MDA_process.responses, self.sets)
    mvs = self.coord.master_vars
    for i in range(len(mvs)):
      for j in range(len(self.vars)):
        if mvs[i].index == self.vars[j].index:
          self.vars[j] = copy.deepcopy(mvs[i])

    self.cache.append([fun[0]+self.coord.phi, vlist, self.MDA_process.getOutputs()])
    # if self.realistic_objective:
    #   fun[0] =(y[0]-self.frealistic)
    if self.solver == "scipy":
      if self.scipy["is_con"]:
        if is_conOnly:
          return con
        else:
          return fun[0] + self.coord.phi
      else:
        hmax = 0.001
        h = max(max(con),hmax)
        return fun[0]+self.coord.phi + (max(max(con),0)**2 if h <= hmax else np.inf) 
    else:
      return [fun[0]+self.coord.phi, con]

  def solve(self, v, w, file: str = None, iter: int = None):  # noqa: C901
    if file is not None:
      self.iter = iter
      self.prepare_post(file)
    if self.solver == "poll" and self.is_main:
      self.set_dependent_baseline(self.coord.master_vars)
    self.cache = []
    self.coord.v = copy.deepcopy(v)
    self.coord.w = copy.deepcopy(w)
    bl = self.get_list_vars(self.get_design_vars())
    res = None
    self.log.log_msg(msg=f"Coordination iteration # {iter}: running subproblem {self.index}", msg_type=MSG_TYPE.INFO.value)
    if self.solver == 'OMADS' or self.solver == 'mads' or self.solver == 'omads' or self.solver != 'scipy':
      eval = {"blackbox": self.evaluate}
      # if self.sets is None:
      #   self.sets = {}
      # param = {"baseline": bl,
      #             "lb": self.get_list_vars_lb(self.get_design_vars()),
      #             "ub": self.get_list_vars_ub(self.get_design_vars()),
      #             "var_names": self.get_list_vars_names(self.get_design_vars()),
      #             "var_type": self.get_vars_types(self.get_design_vars()),
      #             "var_sets": self.sets,
      #             "scaling": self.get_design_vars_scaling(self.get_design_vars()),
      #             "mesh_type": "gmesh",
      #             # "post_dir": "./post",
      #             "constants": self.get_list_constant_updates(self.get_design_vars()),
      #             "constants_name": self.get_list_const_names(self.get_design_vars()),
      #             "name": f"SP_{self.index}",
      #             "post_dir": self.postDir,
      #             "constraints_type": ["PB"]*100}
      is_mac = platform.platform().split('-')[0] == 'macOS'
      param = {"baseline": bl,
                  "lb": self.get_list_vars_lb(self.get_design_vars()),
                  "ub": self.get_list_vars_ub(self.get_design_vars()),
                  "var_names": self.get_list_vars_names(self.get_design_vars()),
                  "var_type": self.get_vars_types(self.get_design_vars()),
                  "var_sets": self.sets,
                  "scaling": [1]*self.nv,
                  # "post_dir": "./post",
                  "constants": self.get_list_constant_updates(self.get_design_vars()),
                  "constants_name": self.get_list_const_names(self.get_design_vars()),
                  "name": f"SP_{self.index}",
                  "mesh_type": self.conf["mesh_type"] \
                  if self.conf is not None \
                  and isinstance(self.conf, dict) \
                  and "mesh_type" in self.conf \
                  and self.conf["mesh_type"] in ["GMESH", "OMESH"] \
                  else "GMESH",
                  "post_dir": self.postDir,
           }
      pinit = min(max(self.tol, max(self.psize) if isinstance(self.psize, list) else self.psize), 1)
      if self.conf is not None and "options" in self.conf and self.conf["options"] is not None:
        options = self.conf["options"]
        options["seed"] = self.conf["options"]["seed"] + iter
      else:
        options = {
          "seed": 10000,
          "budget": self.budget,
          "tol": self.tol,
          "psize_init": pinit,
          "display": self.display,
          "opportunistic": False,
          "check_cache": True,
          "store_cache": True,
          "collect_y": False,
          "rich_direction": True,
          "precision": "high" if is_mac else "medium",
          "save_results": False,
          "save_coordinates": False,
          "save_all_best": False,
          "parallel_mode": False

        }
      # isWin = platform.platform().split('-')[0] == 'Windows'
      # options["precision"] = "high" if isWin else "medium"
      if self.conf is not None and "search" in self.conf and self.conf["search"] is not None:
        search = self.conf["search"]
      else:
        search = {
                    "type": "VNS",
                    "s_method": "LH",
                    "ns": 10,
                    "visualize": False,
                    "criterion": None
                  }

      if self.conf is not None and "constraintsHandling" in self.conf and self.conf["constraintsHandling"] is not None:
        if "Barriers" in self.conf["constraintsHandling"] and self.conf["constraintsHandling"]["Barriers"] is not None:
          param["constraints_type"] = copy.deepcopy(self.conf["constraintsHandling"]["Barriers"])
        if "rho" in self.conf["constraintsHandling"] and self.conf["constraintsHandling"]["rho"] is not None:
          param["rho"] = self.conf["constraintsHandling"]["rho"]
        if "h_max" in self.conf["constraintsHandling"] and self.conf["constraintsHandling"]["h_max"] is not None:
          param["h_max"] = self.conf["constraintsHandling"]["h_max"]
        if "lambda_multipliers" in self.conf["constraintsHandling"] and \
          self.conf["constraintsHandling"]["lambda_multipliers"] is not None:
          param["lambda_multipliers"] = self.conf["constraintsHandling"]["lambda_multipliers"]
        if "barriers" in self.conf["constraintsHandling"] and \
          self.conf["constraintsHandling"]["barriers"] is not None:
          param["constraints_type"] = self.conf["constraintsHandling"]["barriers"]
      
      data = {"evaluator": eval, "param": param, "options":options, "search": search}

      out = {}
      pinit = self.psize
      log_copy = self.log
      # self.log.switch_handler(f"SP{self.index}")
      if self.solver == 'OMADS' or self.solver == 'MADS' or self.solver == 'mads' or self.solver == 'omads':
        out, _, _ = self.optimizer(data)
      else:
        out, _ = self.optimizer(data)
      self.log = log_copy
      # self.log.switch_handler("DMDO")
      self.fmin = out["fmin"]
      self.hmin = out["hmin"]
      if self.conf is not None and "constraintsHandling" in self.conf:
        self.conf["constraintsHandling"]["h_max"] = copy.deepcopy(out["hmin"])
      else:
        self.conf = {}
        self.conf["constraintsHandling"] = {}
        self.conf["constraintsHandling"]["h_max"] = copy.deepcopy(out["hmin"])
      if self.psize_init == PSIZE_UPDATE.DEFAULT:
        self.psize = 1.
      elif self.psize_init == PSIZE_UPDATE.SUCCESS:
        self.psize = out["psuccess"]
      elif self.psize_init == PSIZE_UPDATE.MAX:
        maxpsize = max(out["psize"]) if isinstance(out["psize"],list) else out["psize"]
        psuccess = max(out["psuccess"]) if isinstance(out["psuccess"],list) else out["psuccess"]
        self.psize = max(maxpsize, psuccess)
      elif self.psize_init == PSIZE_UPDATE.LAST:
        self.psize = out["psize"]
      else:
        self.psize = 1.
      
      # COMPLETE: Coordinator forgets q after calling the optimizer, possible remedy is to 
      # update the subproblem variables from the optimizer output and the master variables too
      # then at the end of each outer loop iteration we can calculate q of that subproblem before updating penalty parameters

      #  We need this extra evaluation step to update inconsistincies and the master_variables vector

      self.evaluate(out["xmin"], self.get_list_constant_updates(self.get_design_vars()))
    elif self.solver == 'scipy':
      if self.scipy is not None and isinstance(self.scipy, dict):
        opts = self.scipy["options"]
        bnds = Bounds(lb=self.get_list_vars_lb(self.get_design_vars()), ub=self.get_list_vars_ub(self.get_design_vars()))
        
        if self.scipy["is_con"]:
          non_linear_constraints = NonlinearConstraint(fun=self.get_con, lb=-np.inf, \
            ub= 0.0, keep_feasible=False, jac='2-point', hess=BFGS())
          res = minimize(self.evaluate, method=self.scipy["method"], x0=bl, options=opts, tol=self.scipy["tol"], bounds=bnds,\
                          constraints=[non_linear_constraints])
          self.fmin = res.fun
          self.hmin = 0 if (max(self.get_con(x=res.x)) <= 0.) else max(self.get_con(x=res.x))
        else:
          res = minimize(self.evaluate, method=self.scipy["method"], x0=bl, options=opts, tol=self.scipy["tol"], bounds=bnds)
          self.fmin = res.fun
          self.hmin = 0
        out = copy.deepcopy(res.x)
        self.evaluate(out)
        
      else:
        msg = 'Scipy solver is selected but its dictionary settings is inappropriately defined!'
        self.log.log_msg(msg=msg, msg_type=MSG_TYPE.ERROR.value)
        raise IOError(msg)
      out = copy.deepcopy(res.x)
      self.evaluate(out)
    else:
      self.log.log_msg(msg=f"Unknown solver is assigned to subproblem #{self.index}", msg_type=MSG_TYPE.ERROR.value)
    self.log.log_msg(msg=f"Coordination iteration # {iter}: completed subproblem {self.index}", msg_type=MSG_TYPE.INFO.value)

    

    return out

  def get_coupling_vars_diff(self, con):
    # TODO: Add categorical coupling difference
    vc: List[variableData] = self.get_coupling_vars()
    for i in range(len(vc)):
      if isinstance(vc[i].value, list):
        for j in range(len(vc[i].value)):
          if vc[i].type[j][0].lower() == "c":
            con.append(float(self.sets[vc[i].set].index(vc[i].value[j])-vc[i].ub[j]))
          else:
            con.append(float(vc[i].value[j]-vc[i].ub[j]))
      else:
        if vc[i].type[0].lower() == "c":
          con.append(float(self.sets[vc[i].set].index(vc[i].value)-vc[i].ub))
        else:
          con.append(float(vc[i].value-vc[i].ub))

    for i in range(len(vc)):
      if isinstance(vc[i].value, list):
        for j in range(len(vc[i].value)):
          if vc[i].type[j][0].lower() == "c":
            con.append(float(vc[i].lb[j]-self.sets[vc[i].set].index(vc[i].value[j])))
          else:
            con.append(float(vc[i].lb[j]-vc[i].value[j]))
      else:
        if vc[i].type[0].lower() == "c":
          con.append(float(vc[i].lb-self.sets[vc[i].set].index(vc[i].value)))
        else:
          con.append(float(vc[i].lb-vc[i].value))
    return con

  def set_dependent_baseline(self, vars: List[variableData]):
    for j in range(len(self.vars)):
      if self.vars[j].coupling_type == COUPLING_TYPE.FEEDBACK or self.vars[j].coupling_type == COUPLING_TYPE.SHARED \
        or self.vars[j].coupling_type == COUPLING_TYPE.UNCOUPLED or self.vars[j].coupling_type == COUPLING_TYPE.CONSTANT:
        for i in range(len(vars)):
          if isinstance(self.vars[j].link , list):
            found = False
            for inx in self.vars[j].link:
              if inx == vars[i].sp_index:
                found = True
          else:
            found = self.vars[j].link == vars[i].sp_index
          if self.vars[j].name == vars[i].name and found:
            self.vars[j].baseline = vars[i].value
            self.vars[j].value = vars[i].value
    
    # for j in range(len(self.resps)):
    #   if self.resps[j].coupling_type == COUPLING_TYPE.FEEDFORWARD 
    # or self.resps[j].coupling_type == COUPLING_TYPE.SHARED or self.resps[j].coupling_type == COUPLING_TYPE.CONSTANT:
    #     for i in range(len(vars)):
    #       if isinstance(self.resps[j].link , list):
    #         found = False
    #         for inx in self.resps[j].link:
    #           if inx == vars[i].sp_index:
    #             found = True
    #       else:
    #         found = self.resps[j].link == vars[i].sp_index
    #       if self.resps[j].name == vars[i].name and found:
    #         self.resps[j].baseline = vars[i].value

    # for i in range(len(vars)):
    #   if vars[i].sp_index == self.index:
    #     if vars[i].coupling_type == COUPLING_TYPE.FEEDFORWARD 
    # or vars[i].coupling_type == COUPLING_TYPE.SHARED or vars[i].coupling_type == COUPLING_TYPE.CONSTANT:
    #       for j in range(len(self.vars)):
    #         if isinstance(self.vars[j].link , list):
    #           found = False
    #           for inx in self.vars[j].link:
    #             if inx == vars[i].sp_index:
    #               found = True
    #         else:
    #           found = self.vars[j].link == vars[i].sp_index
    #         if self.vars[j].name == vars[i].name and found:
    #           self.vars[j].baseline = vars[i].value

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
        if isinstance(vars[i].value, list):
          for j in range(len(vars[i].value)):
            v.append(vars[i].value[j] if (vars[i].type[j].lower() == 'r' \
                                          or vars[i].type[j].lower() == 'i') else vars[i].value[j])
        else:
          v.append(vars[i].value if (vars[i].type[0].lower() == 'r' or vars[i].type[0].lower() == 'i') else vars[i].value)
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
            if vars[i].type[j] == VAR_TYPE.REAL.name:
              v.append("REAL")
            elif vars[i].type[j] == VAR_TYPE.INTEGER.name:
              v.append("INTEGER")
            elif vars[i].type[j] == VAR_TYPE.CATEGORICAL.name:
              v.append("CATEGORICAL")
            else:
              v.append(vars[i].type[j])
        else:
          if vars[i].type == VAR_TYPE.REAL.name:
            v.append("REAL")
          elif vars[i].type == VAR_TYPE.INTEGER.name:
            v.append("INTEGER")
          elif vars[i].type == VAR_TYPE.CATEGORICAL.name:
            v.append("CATEGORICAL")
          else:
            v.append(vars[i].type)
    return v
