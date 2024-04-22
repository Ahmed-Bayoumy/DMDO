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
from ._common import *
from ._protocols import *
from .SP import *
from .MDO import *
from .MDA import *
import platform

@dataclass
class optimizationData:
  objectives: List[Callable]
  constraints: List[Callable]
  objectiveWeights: List[Any]
  constraintsHandling: List[int]
  solver: Callable



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
  log: logger = None

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
        msg = f'Unrecognized coupling type {vin[i][4]} is introduced to the variables dictionary at this key {i}'
        self.log.log_msg(msg=msg, msg_type=MSG_TYPE.ERROR.value)
        raise IOError(f'Unrecognized coupling type {vin[i][4]} is introduced to the variables dictionary at this key {i}')
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
  
  def raiser(self, blerr): 
    raise ImportError(blerr)
  
  def variablesSetup(self):
    """ Setup the list of global variables struct """
    self.log.log_msg(msg="Starting variables setup ...", msg_type=MSG_TYPE.INFO.value)
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
    # blError = raise IOError()
    for i in range(len(names)):
      cond_on = None
      if isinstance(dim[i], str):
        conp = dim[i].split("_")[1:]
        for k in range(len(names)):
          if spi[i] == spi[k] and names[k] == vin[conp[0]][0]:
            if len(conp)>1:
              dim[i] = int(bl[k] + float(conp[1]))
            else:
              dim[i] = bl[k]
            cond_on = vin[conp[0]][0]
      if dim[i] > 1:
        v[f"var{i+1}"] = {"index": i+1,
        "sp_index": spi[i],
        f"name": names[i],
        "dim": dim[i],
        "value": [bl[i]]*dim[i] if not isinstance(bl[i], list) else bl[i] if len(bl[i]) == dim[i] else self.raiser(f'The baseline vector of {names[i]} has {len(bl[i])} elements which is different from the variable initial #dimensions which is {dim[i]}'),
        "coupling_type": coupling_t[i],
        "link": links[i],
        "baseline": [bl[i]]*dim[i] if not isinstance(bl[i], list) else bl[i] if len(bl[i]) == dim[i] else self.raiser(f'The baseline vector of {names[i]} has {len(bl[i])} elements which is different from the variable initial #dimensions which is {dim[i]}'),
        "scaling": [scaling[i]]*dim[i],
        "lb": [lb[i]]*dim[i],
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
    
    self.log.log_msg(msg="Completed variables setup.\n", msg_type=MSG_TYPE.INFO.value)
  
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
        msg = f"The blackbox file {str(bb)} is not an executable! Please provide a valid executable file."
        self.log.log_msg(msg, MSG_TYPE.ERROR.value)
        raise IOError(msg)
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
    self.log.log_msg(msg="Starting coordinator setup ...", msg_type=MSG_TYPE.INFO.value)
    c = self.data["coord"]
    self.Coords = []
    for i in c:
      if c[i]["type"] != "ADMM":
        msg = f'Currently DMDO only supports ADMM coordinator. Please change the coordination type under {i} to ADMM.'
        self.log.log_msg(msg=msg, msg_type=MSG_TYPE.ERROR.value)
        raise IOError(msg)
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
      self.log.log_msg(msg=f"Coordinator configurations: {c}", msg_type=MSG_TYPE.INFO.value)
    self.log.log_msg(msg="Completed coordinator setup ...\n", msg_type=MSG_TYPE.INFO.value)

  def SPSetup(self):
    """ Setup subproblems definition """
    self.log.log_msg(msg="Starting subproblems setup ...", msg_type=MSG_TYPE.INFO.value)
    SP = self.data["subproblem"]
    is_conf =  "CONFIG" in self.data
    if is_conf:
      CONF = self.data["CONFIG"]
    


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
        sets=self.Sets,
        conf= CONF[SP[i]["configurations"]] if ("configurations" in SP[i]) and is_conf and (SP[i]["configurations"] in CONF) else None

      ))
    self.log.log_msg(msg="Completed subproblems setup.\n", msg_type=MSG_TYPE.INFO.value)

  def MDOSetup(self):
    """ Setup MDO process """
    self.log.log_msg("Starting MDO architecture setup ...", msg_type=MSG_TYPE.INFO.value)
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
    if "Sets" in self.data:
      self.MDAO.sets = self.data["Sets"]
    else:
      self.MDAO.sets = None
    if self.MDAO.Coordinator == None:
      msg = f'Could not find the coordinator with index {MDAO["coordinator"]} assigned to the MDO coordinator key.'
      self.log.log_msg(msg=msg, msg_type=MSG_TYPE.ERROR.value)
      raise IOError(msg)
    self.log.log_msg(f'Configurations of MDO: {self.data["MDO"]}', msg_type=MSG_TYPE.INFO.value)
    self.log.log_msg("Completed MDO architecture setup.\n", msg_type=MSG_TYPE.INFO.value)
  
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
    self.log.log_msg(msg = "MDO problem setup...\n", msg_type=MSG_TYPE.INFO.value)
    self.variablesSetup()
    self.DASetup()
    self.MDASetup()
    self.COORDSetup()
    self.get_varSets()
    self.SPSetup()
    self.MDOSetup()
    self.UserData()
    self.log.log_msg(msg = "Successfully completed the MDO problem setup.\n", msg_type=MSG_TYPE.INFO.value)
    self.log.log_msg(msg="--------------------------------------------------\n", msg_type=MSG_TYPE.INFO.value)
    return self.MDAO

