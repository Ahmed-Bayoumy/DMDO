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
from ._protocols import *
from .coordinator import *
from .SP import *
import pickle


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
  log: logger = None
  iter: int = 0
  file: str = None


@dataclass
class MDO(MDO_data):
  def setup(self, input):
    data: Dict = {}
    if isfile(input):
      data = json.load(input)

  def get_list_of_var_values(self, x: List[variableData]):
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
        for k in range(mvold_clone.dim):
          if mv_type == 'c' or mv_type == 'd':
            mvold_clone.value[k] = self.sets[mvold_clone.set].index(self.Coordinator.master_vars_old[i].value[k])
      else:
        if mv_type == 'c' or mv_type == 'd':
            mv_clone.value = self.sets[mv_clone.set].index(self.Coordinator.master_vars[i].value)
            mvold_clone.value = self.sets[mvold_clone.set].index(self.Coordinator.master_vars_old[i].value)
      # if mv_type =='c':
      #   if mv_cloneuntil.value == mvold_clone.value:
      #     dx.append(0)
      #   else:
      #     dx.append(1)
      # else:
      dx.append(mv_clone - mvold_clone)
      # x.append(self.Coordinator.master_vars[i].value)
      # xold.append(self.Coordinator.master_vars_old[i].value)
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
      msg = f'Stop: qmax = {np.max(np.abs(self.Coordinator.q))} < {self.inc_stop}'
      self.log.log_msg(msg=msg, msg_type=MSG_TYPE.INFO.value)
      print(msg)
      msg = "Max inconsitency is below stopping threshold"
      self.log.log_msg(msg=msg, msg_type=MSG_TYPE.INFO.value)
      self.stop = msg
      return True

    i = self.noprogress_stop

    if iter > i + 2 and np.log(np.min(self.Coordinator.w) > 6.) and np.less_equal(self.tab_inc[iter-i], np.min(self.tab_inc[iter-i+1:iter])):
      msg = f'Stop: no progress after {i} iterations.'
      self.log.log_msg(msg=msg, msg_type=MSG_TYPE.INFO.value)
      print(msg)
      msg = "No progress after several iterations"
      self.log.log_msg(msg=msg, msg_type=MSG_TYPE.INFO.value)
      print(msg)
      self.stop = msg
      return True

    return False

  def get_weights_size(self, sp_i: int) -> int:
    link1 = self.subProblems[sp_i].coord._linker[0]
    out = 0
    for i in range(len(link1)):
      out += self.subProblems[sp_i].coord.master_vars[link1[i]-1].dim
    
    return out
  
  def initialize_OL_res_file(self, file):
    if not hasattr(self.Coordinator, "master_vars"):
      return
    if not os.path.exists(file):
      mode = 'x'
    else:
      mode = 'w'
    with open(file, mode=mode) as csv_file:
      keys = [f'{"Time"}', f'{"Iteration #".rjust(30)}', f'{"Max. inconsistency".rjust(30)}', f'{"Objective".rjust(30)}', f'{"Status".rjust(30)}', f'{"Variables change".rjust(30)}', f'{"Maximum penalty".rjust(30)}', f'{"Coupling_with_qmax".rjust(30)}'] + [f'{f"{x.name}_{x.sp_index}".rjust(30)}' for x in self.Coordinator.master_vars]
      writer = csv.DictWriter(csv_file, fieldnames=keys)
      writer.writeheader()    # add column names in the CSV file
  
  def prepare_post(self, file=None):
    if file is None:
      name = "unknown"
      pd = os.path.join(os.getcwd(), f'{name}_post')
    else:
      res_dir = os.path.dirname(file)
      ht = os.path.split(file)
      name = file.split('.')[0]
      ext = file.split('.')[1]
      pd = os.path.join(ht[0], f'{name}_post')
    
    
    pfo = os.path.join(pd, 'Coordination_history.out')
    if os.path.exists(pd):
      shutil.rmtree(pd)
    os.mkdir(pd)
    self.postDir = pd
    self.Ol_file = pfo
    self.log.relocate_logger(source_file="DMDO.log", Dest_file=pd)
    self.initialize_OL_res_file(pfo)
    
  def Add_OL_res_row(self, r: Dict):
    with open(self.Ol_file, mode='a') as csv_file:
      keys = [f'{"Time"}', f'{"Iteration #".rjust(30)}', f'{"Max. inconsistency".rjust(30)}', f'{"Objective".rjust(30)}', f'{"Status".rjust(30)}', f'{"Variables change".rjust(30)}', f'{"Maximum penalty".rjust(30)}', f'{"Coupling_with_qmax".rjust(30)}'] + [f'{f"{x.name}_{x.sp_index}".rjust(30)}' for x in self.Coordinator.master_vars]
      writer = csv.DictWriter(csv_file, fieldnames=keys)
      writer.writerow(r)    # add column names in the CSV file

  def propose_best_candidates(self, s):
    temp_obj = []
    sp_vnames = [vs.name for vs in self.subProblems[s].vars]
    for nsp in range(len(self.subProblems)):
      ctemp = copy.deepcopy(self.subProblems[nsp].cache)
      for ct in ctemp:
        if len(temp_obj) == 0 or ct[0] < min(temp_obj):
          temp_obj.append(ct[0])
          if s == nsp:
            copy.deepcopy(ct[1])
          else:
            svIndex = -1
            for v in self.subProblems[nsp].vars:
              svIndex += 1
              try:
                vindex = sp_vnames.index(v.name)
                if v.link == self.subProblems[s].index:
                  self.subProblems[s].vars[vindex].baseline = ct[1][svIndex]
              except:
                continue
            srIndex = -1
            for v in self.subProblems[nsp].resps:
              srIndex += 1
              try:
                vindex = sp_vnames.index(v.name)
                if v.link == self.subProblems[s].index:
                  self.subProblems[s].vars[vindex].baseline = ct[2][srIndex]
              except:
                continue
    
    return


  def run(self, file=None, resume= False, mode="Serial"):
    global eps_fio, eps_qio
    if self.log == None:
      self.log: logger = logger()
      self.log.initialize("DMDO.log")
    if not resume:
      self.log.log_msg(msg="Running MDO ... ", msg_type=MSG_TYPE.INFO.value)
      # Note: once you run MDAO, the data stored in eps_fio and eps_qio shall be deleted. It is recommended to store such data to a different variable before running another MDAO
      eps_fio = []
      eps_qio = []
      self.file = file
    else:
      self.log.log_msg(msg=f"Resuming MDO coordination at iteration #{self.iter} ... ", msg_type=MSG_TYPE.INFO.value)
      file = self.file


    """ Run the MDO process """
    #  COMPLETE: fix the setup of the local (associated with SP) and global (associated with MDO) coordinators
    self.fmin = np.inf
    for iter in range(self.Coordinator.budget):
      if iter > 0:
        coord_inst = self
        pickle.dump(coord_inst, open(self.postDir + '/coord_chk.pkl', "wb"))
        self.Coordinator.master_vars_old = copy.deepcopy(self.Coordinator.master_vars)
      else:
        self.Coordinator.master_vars_old = copy.deepcopy(self.variables)
        self.Coordinator.master_vars = copy.deepcopy(self.variables)

      """ ADMM inner loop """
      if mode == "Serial" or mode == "serial":
        for s in range(len(self.subProblems)):
          if iter == 0:
            self.subProblems[s].coord = copy.deepcopy(self.Coordinator)
            self.subProblems[s].set_pair()
            self.xavg = self.subProblems[s].coord._linker
            self.Coordinator.v = [0.] * self.get_weights_size(s)
            self.Coordinator.w = [1.] * self.get_weights_size(s)
            if s == 0:
              self.prepare_post(file)
            self.subProblems[s].log = self.log
          else:
            self.subProblems[s].coord.master_vars = copy.deepcopy(self.Coordinator.master_vars)
            
          self.subProblems[s].modify_cond_vars(self.Coordinator.master_vars)
          # if iter > 1:
          #   self.propose_best_candidates(s)
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
          self.log.log_msg(msg=f'{iter} || qmax: {np.max(np.abs(self.Coordinator.q))} || Obj: {self.fmin} || dx: {dx} || max(w): {np.max(self.Coordinator.w)}', msg_type=MSG_TYPE.INFO.value)
          print(f'{iter} || qmax: {np.max(np.abs(self.Coordinator.q))} || Obj: {self.fmin} || dx: {dx} || max(w): {np.max(self.Coordinator.w)}')
          qb = self.Coordinator.batch_q(self.Coordinator.q)
          ql: list = []
          for i in range(len(qb)):
            if isinstance(qb[i], list):
              ql.append(max(np.abs(qb[i])))
            else:
              ql.append(abs(qb[i]))

          index = np.argmax(ql)
          self.log.log_msg(msg=f'Highest inconsistency : {self.Coordinator.master_vars[self.Coordinator._linker[0,index]-1].name}_'
          f'{self.Coordinator.master_vars[self.Coordinator._linker[0,index]-1].sp_index} to '
            f'{self.Coordinator.master_vars[self.Coordinator._linker[1,index]-1].name}_'
          f'{self.Coordinator.master_vars[self.Coordinator._linker[1,index]-1].link}', msg_type=MSG_TYPE.INFO.value)
        print(f'Highest inconsistency : {self.Coordinator.master_vars[self.Coordinator._linker[0,index]-1].name}_'
          f'{self.Coordinator.master_vars[self.Coordinator._linker[0,index]-1].link} to '
            f'{self.Coordinator.master_vars[self.Coordinator._linker[1,index]-1].name}_'
          f'{self.Coordinator.master_vars[self.Coordinator._linker[1,index]-1].link}')
        """ Write OL results to the file"""
      keys = [f'{"Time"}', f'{"Iteration #".rjust(30)}', f'{"Max. inconsistency".rjust(30)}', f'{"Objective".rjust(30)}', f'{"Status".rjust(30)}', f'{"Variables change".rjust(30)}', f'{"Maximum penalty".rjust(30)}', f'{"Coupling_with_qmax".rjust(30)}'] + [f'{f"{x.name}_{x.sp_index}".rjust(30)}' for x in self.Coordinator.master_vars]
      curr_time = time.strftime("%H:%M:%S", time.localtime()) 
      hmin = max(self.hmin) if isinstance(self.hmin, list) and self.hmin is not None and len(self.hmin) > 0 else self.hmin
      hstatus = "Feasible" if hmin <= 0 else "Infeasible"
      status = copy.deepcopy(hstatus) if self.fmin != np.inf else "Error"
      cmax = f'{self.Coordinator.master_vars[self.Coordinator.extended_linker[0,index]-1].name}_{self.Coordinator.master_vars[self.Coordinator.extended_linker[0,index]-1].link} to {self.Coordinator.master_vars[self.Coordinator.extended_linker[1,index]-1].name}_{self.Coordinator.master_vars[self.Coordinator.extended_linker[1,index]-1].link}'
      row = {keys[0]: f'{f"{curr_time}"}', 
             keys[1]: f'{f"{iter}".rjust(30)}', 
             keys[2]: f'{f"{np.max(np.abs(self.Coordinator.q))}".rjust(30)}', 
             keys[3]: f'{f"{self.fmin}".rjust(30)}', 
             keys[4]: f'{f"{status}".rjust(30)}',
             keys[5]: f'{f"{dx}".rjust(30)}',
             keys[6]: f'{f"{np.max(self.Coordinator.w)}".rjust(30)}',
             keys[7]: f'{f"{cmax}".rjust(30)}'}
      
      cv = 0
      lr = len(row)
      for v in self.Coordinator.master_vars:
        row[keys[lr+cv]] = f'{f"{v.value}".rjust(30)}'
        cv += 1
      
      self.Add_OL_res_row(row)
      """ Update LM and PM """
      self.Coordinator.update_multipliers(iter)

      """ Stopping criteria """
      self.tab_inc.append(np.max(np.abs(self.Coordinator.q)))
      stop: bool = self.check_termination_critt(iter)
      

      if stop:
        break
      self.eps_qio = copy.deepcopy(eps_qio)
      self.eps_fio = copy.deepcopy(eps_fio)
    else:
      # TODO: Add parallel sunbproblems execution
      """ """
    if self.display == True:
      print(f'------Run_Summary------')
      self.log.log_msg(msg=f'------Run_Summary------', msg_type=MSG_TYPE.INFO.value)
      print(self.stop)
      self.log.log_msg(msg=self.stop, msg_type=MSG_TYPE.INFO.value)
      print(f'q = {self.Coordinator.q}')
      self.log.log_msg(msg=f'q = {self.Coordinator.q}', msg_type=MSG_TYPE.INFO.value)
      for i in self.Coordinator.master_vars:
        print(f'{i.name}_{i.sp_index} = {i.value}')
        self.log.log_msg(msg=f'{i.name}_{i.sp_index} = {i.value}', msg_type=MSG_TYPE.INFO.value)

      fmin = 0
      hmax = np.inf
      for j in range(len(self.subProblems)):
        print(f'Study_ID_{self.subProblems[j].index}: fmin= {self.subProblems[j].fmin_nop}, hmin= {self.subProblems[j].hmin}')
        self.log.log_msg(msg=f'Study_ID_{self.subProblems[j].index}: fmin= {self.subProblems[j].fmin_nop}, hmin= {self.subProblems[j].hmin}', msg_type=MSG_TYPE.INFO.value)
        fmin += self.subProblems[j].fmin_nop
        hmin= self.subProblems[j].hmin
        if isinstance(hmin, list): 
          hmax = max(hmin) 
        else:
          hmax = hmin
      print(f'P_main: fmin= {fmin}, hmin= {hmax}')
      self.log.log_msg(msg=f'P_main: fmin= {fmin}, hmin= {hmax}', msg_type=MSG_TYPE.INFO.value)
      print(f'Final obj value of the main problem: \n {fmin}')
      self.log.log_msg(msg=f'Final obj value of the main problem: \n {fmin}', msg_type=MSG_TYPE.INFO.value)
    self.log.log_msg(msg="MDO run has been successfully completed.", msg_type=MSG_TYPE.INFO.value)
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
