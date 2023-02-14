import os
from DMDO import *
import math
import numpy as np
from numpy import cos, exp, pi, prod, sin, sqrt, subtract, inf
import copy
from typing import List, Dict, Any, Callable, Protocol, Optional

user =USER

def termTest():
  print("Termination criterria work!")

def A1(x):
  LAMBDA = 0.0
  for i in range(len(x)):
    if x[i] == 0.:
      x[i] = 1e-12
  return math.log(x[0]+LAMBDA) + math.log(x[1]+LAMBDA) + math.log(x[2]+LAMBDA)

def A2(x):
  LAMBDA = 0.0
  for i in range(len(x)):
    if x[i] == 0.:
      x[i] = 1e-12
  return np.divide(1., (x[0]+LAMBDA)) + np.divide(1., (x[1]+LAMBDA)) + np.divide(1., (x[2]+LAMBDA))

def opt1(x, y):
  return [sum(x)+y[0], [0.]]

def opt2(x, y):
  return [0., [x[1]+y[0]-10.]]

def SR_A1(x):
  """ Speed reducer A1 """
  return (0.7854*x[0]*x[1]**2*(3.3333*x[2]*x[2] + 14.9335*x[2] - 43.0934))

def SR_A2(x):
  """ Speed reducer A2 """
  return (-1.5079*x[0]*x[4]**2) + (7.477 * x[4]**3) + 0.7854 * x[3] * x[4]**2

def SR_A3(x):
  """ Speed reducer A3 """
  return (-1.5079*x[0]*x[4]**2 + 7.477*x[4]**3 + 0.7854*x[3]*x[4]**2)

def SR_opt1(x, y):
  g5 = 27/(x[0]*x[1]**2*x[2]) -1
  g6 = 397.5/(x[0]*x[1]**2*x[2]**2) -1
  g9 = x[1]*x[2]/40 -1
  g10 = 5*x[1]/x[0] -1
  g11 = x[0]/(12*x[1]) -1
  return [y[0], [g5,g6,g9,g10,g11]]

def SR_opt2(x, y):
  g1 = sqrt( ((745*x[3])/(x[1]*x[2]))**2 + 1.69e+7)/(110*x[4]**3) -1
  g3 = (1.5*x[4] + 1.9)/x[3] -1
  g7 = 1.93*x[3]**3/(x[1]*x[2]*x[4]**4) -1
  return [y, [g1,g3,g7]]

def SR_opt3(x, y):
  g2 = sqrt( ((745*x[3])/(x[1]*x[2]))**2 + 1.575e+8)/(85*x[4]**3) -1
  g4 = (1.1*x[4] + 1.9)/x[3] -1
  g8 = (1.93*x[3]**3)/(x[1]*x[2]*x[4]**4) -1
  return [y, [g2, g4, g8]]

def GP_A1(z):
  z1 = sqrt(z[0]**2 + z[1]**-2 + z[2]**2)
  z2 = sqrt(z[2]**2 + z[3]**2  + z[4]**2)
  return z1**2 + z2**2

def GP_opt1(z,y):
  if isinstance(y, list) and len(y) > 0:
    return [y[0], [z[0]**-2 + z[1]**2 - z[2]**2, z[2]**2 + z[3]**-2  - z[4]**2]]
  else:
    return [y, [z[0]**-2 + z[1]**2 - z[2]**2, z[2]**2 + z[3]**-2  - z[4]**2]]

def GP_A2(z):
  z3 = sqrt(z[0]**2 + z[1]**-2 + z[2]**-2 + z[3]**2)
  return z3

def GP_opt2(z,y):
  return [0, [z[0]**2 + z[1]**2 - z[3]**2, z[0]**-2 + z[2]**2 - z[3]**2]]

def GP_A3(z):
  z6 = sqrt(z[0]**2 + z[1]**2 + z[2]**2 +z[3] **2)
  return z6

def GP_opt3(z, y):
  return [0, [z[0]**2 + z[1]**-2 - z[2]**2, z[0]**2 +z[1]**2 - z[3]**2]]


def Basic_MDO():
  #  Variables setup
  v = {}
  V: List[variableData] = []
  names = ["u", "v", "a", "b", "u", "w", "a", "b"]
  spi = [1,1,1,1,2,2,2,2]
  links = [2, None, 2, 2, 1, None, 1, 1]
  coupling_t = [COUPLING_TYPE.SHARED, COUPLING_TYPE.UNCOUPLED, COUPLING_TYPE.FEEDFORWARD,
  COUPLING_TYPE.FEEDBACK, COUPLING_TYPE.SHARED, COUPLING_TYPE.UNCOUPLED,
   COUPLING_TYPE.FEEDBACK, COUPLING_TYPE.FEEDFORWARD]
  lb = [0.]*8
  ub = [10.]*8
  bl = [1.]*8
  scaling = np.subtract(ub,lb)
  Qscaling = []
  # Variables dictionary with subproblems link
  for i in range(8):
    v[f"var{i+1}"] = {"index": i+1,
    "sp_index": spi[i],
    f"name": names[i],
    "dim": 1,
    "value": 0.,
    "coupling_type": coupling_t[i],
    "link": links[i],
    "baseline": bl[i],
    "scaling": scaling[i],
    "lb": lb[i],
    "value": bl[i],
    "ub": ub[i]}
    Qscaling.append(1/scaling[i] if 1/scaling[i] != np.inf and 1/scaling[i] != np.nan else 1.)

  for i in range(8):
    V.append(variableData(**v[f"var{i+1}"]))

  # Analyses setup; construct disciplinary analyses
  DA1: process = DA(inputs=[V[0], V[1], V[3]],
  outputs=[V[2]],
  blackbox=A1,
  links=2,
  coupling_type=COUPLING_TYPE.FEEDFORWARD
  )

  DA2: process = DA(inputs=[V[4], V[5], V[6]],
  outputs=[V[7]],
  blackbox=A2,
  links=1,
  coupling_type=COUPLING_TYPE.FEEDFORWARD
  )

  # MDA setup; construct subproblems MDA
  sp1_MDA: process = MDA(nAnalyses=1, analyses = [DA1], variables=[V[0], V[1], V[3]], responses=[V[2]])
  sp2_MDA: process = MDA(nAnalyses=1, analyses = [DA2], variables=[V[4], V[5], V[6]], responses=[V[7]])

  # Construct the coordinator
  coord = ADMM(beta = 1.3,
  nsp=2,
  budget = 50,
  index_of_master_SP=1,
  display = True,
  scaling = Qscaling,
  mode = "serial",
  M_update_scheme= w_scheme.MEDIAN,
  store_q_io=True)

  

  # Construct subproblems
  sp1 = SubProblem(nv = 3,
  index = 1,
  vars = [V[0], V[1], V[3]],
  resps = [V[2]],
  is_main=1,
  analysis= sp1_MDA,
  coordination=coord,
  opt=opt1,
  fmin_nop=np.inf,
  budget=20,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST,
  freal=2.625)

  sp2 = SubProblem(nv = 3,
  index = 2,
  vars = [V[4], V[5], V[6]],
  resps = [V[7]],
  is_main=0,
  analysis= sp2_MDA,
  coordination=coord,
  opt=opt2,
  fmin_nop=np.inf,
  budget=20,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST
  )

  # Construct MDO workflow
  MDAO: MDO = MDO(
  Architecture = MDO_ARCHITECTURE.IDF,
  Coordinator = coord,
  subProblems = [sp1, sp2],
  variables = V,
  responses = [V[2], V[7]],
  fmin = np.inf,
  hmin = np.inf,
  display = True,
  inc_stop = 1E-9,
  stop = "Iteration budget exhausted",
  tab_inc = [],
  noprogress_stop = 100
  )

  # Run the MDO problem
  out = MDAO.run()

  print(f'------Run_Summary------')
  print(MDAO.stop)
  print(f'q = {MDAO.Coordinator.q}')
  for i in MDAO.Coordinator.master_vars:
    print(f'{i.name}_{i.sp_index} = {i.value}')

  fmin = 0
  hmax = -inf
  for j in range(len(MDAO.subProblems)):
    print(f'SP_{MDAO.subProblems[j].index}: fmin= {MDAO.subProblems[j].MDA_process.getOutputs()}, hmin= {MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] , MDAO.subProblems[j].MDA_process.getOutputs())[1]}')
    fmin += sum(MDAO.subProblems[j].MDA_process.getOutputs())
    hmin= MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] , MDAO.subProblems[j].MDA_process.getOutputs())[1]
    if max(hmin) > hmax: 
      hmax = max(hmin) 
  print(f'P_main: fmin= {fmin}, hmax= {hmax}')
  print(f'Final obj value of the main problem: \n {fmin}')

def speedReducer():
  #  Variables setup
  f1min = 722
  f1max = 5408
  f2min = 184
  f2max = 506
  f3min = 942
  f3max = 1369
  #
  v = {}
  V: List[variableData] = []
  s  = COUPLING_TYPE.SHARED
  ff = COUPLING_TYPE.FEEDFORWARD
  fb = COUPLING_TYPE.FEEDBACK
  un = COUPLING_TYPE.UNCOUPLED
  dum = COUPLING_TYPE.DUMMY


  names = ["x1", "x2", "x3", "f1",   "x1", "x2", "x3", "x4", "x6", "f2",   "x1", "x2", "x3", "x5", "x7", "f3"]
  spi =   [   1,    1,    1,		1,		  2,		2,		2,		2,		2,		2,      3,    3,		3,		3,    3,	  3]
  links = [[2,3],[2,3],[2,3],   None,  [1,3],[1,3],[1,3], None, None,    None,  [1,2],[1,2],[1,2], None, None,    None]
  lb =    [2.6 ,  0.7 ,  17., 722.,  2.6 ,  0.7,  17.,  7.3,  2.9, 184.,   2.6 ,  0.7,  17.,  7.3,   5.,942.]
  ub =    [3.6 ,  0.8 ,  28.,5408.,  3.6 ,  0.8,  28.,  8.3,  3.9, 506.,   3.6 ,  0.8 , 28.,  8.3,  5.5,1369.]
  bl =    np.divide(np.subtract(ub, lb), 2.)
  
  coupling_t = \
          [ s,      s,		s,		un,		s,		s,		s,		un,		un,	 un,   s,    s,    s,   un,    un,    un]
 
  scaling = [10.] * 16

  # Variables dictionary with subproblems link
  for i in range(16):
    v[f"var{i+1}"] = {"index": i+1,
    "sp_index": spi[i],
    f"name": names[i],
    "dim": 1,
    "value": 0.,
    "coupling_type": coupling_t[i],
    "link": links[i],
    "baseline": bl[i],
    "scaling": scaling[i],
    "lb": lb[i],
    "value": bl[i],
    "ub": ub[i]}

  for i in range(16):
    V.append(variableData(**v[f"var{i+1}"]))

  # Analyses setup; construct disciplinary analyses
  DA1: process = DA(inputs=[V[0], V[1], V[2]],
  outputs=[V[3]],
  blackbox=SR_A1,
  links=[4],
  coupling_type=COUPLING_TYPE.FEEDFORWARD)
  
  DA2: process = DA(inputs=[V[4], V[5], V[6], V[7], V[8]],
  outputs=[V[9]],
  blackbox=SR_A2,
  links=[4],
  coupling_type=COUPLING_TYPE.FEEDFORWARD
  )

  DA3: process = DA(inputs=[V[10], V[11], V[12], V[13], V[14]],
  outputs=[V[15]],
  blackbox=SR_A3,
  links=[4],
  coupling_type=COUPLING_TYPE.FEEDFORWARD
  )

  # MDA setup; construct subproblems MDA
  sp1_MDA: process = MDA(nAnalyses=1, analyses = [DA1], variables=[V[0], V[1], V[2]], responses=[V[3]])
  sp2_MDA: process = MDA(nAnalyses=1, analyses = [DA2], variables=[V[4], V[5], V[6], V[7], V[8]], responses=[V[9]])
  sp3_MDA: process = MDA(nAnalyses=1, analyses = [DA3], variables=[V[10], V[11], V[12], V[13], V[14]], responses=[V[15]])

  # Construct the coordinator
  coord = ADMM(beta = 1.3,
  nsp=4,
  budget = 50,
  index_of_master_SP=1,
  display = True,
  scaling = 0.1,
  mode = "serial",
  M_update_scheme= w_scheme.MEDIAN,
  store_q_io=True
  )

  # Construct subproblems
  sp1 = SubProblem(nv = 3,
  index = 1,
  vars = [V[0], V[1], V[2]],
  resps = [V[3]],
  is_main=1,
  analysis= sp1_MDA,
  coordination=coord,
  opt=SR_opt1,
  fmin_nop=np.inf,
  budget=20,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST,
  freal=2994.47)

  sp2 = SubProblem(nv = 5,
  index = 2,
  vars = [V[4], V[5], V[6], V[7], V[8]],
  resps = [V[9]],
  is_main=0,
  analysis= sp2_MDA,
  coordination=coord,
  opt=SR_opt2,
  fmin_nop=np.inf,
  budget=20,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST
  )

  sp3 = SubProblem(nv = 5,
  index = 3,
  vars = [V[10], V[11], V[12], V[13], V[14]],
  resps = [V[15]],
  is_main=0,
  analysis= sp3_MDA,
  coordination=coord,
  opt=SR_opt3,
  fmin_nop=np.inf,
  budget=20,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST)

# Construct MDO workflow
  MDAO: MDO = MDO(
  Architecture = MDO_ARCHITECTURE.IDF,
  Coordinator = coord,
  subProblems = [sp1, sp2, sp3],
  variables = V,
  responses = [V[3], V[9], V[15]],
  fmin = np.inf,
  hmin = np.inf,
  display = True,
  inc_stop = 1E-9,
  stop = "Iteration budget exhausted",
  tab_inc = [],
  noprogress_stop = 100
  )


# Run the MDO problem
  out = MDAO.run()

  print(f'------Run_Summary------')
  print(MDAO.stop)
  print(f'q = {MDAO.Coordinator.q}')
  for i in MDAO.Coordinator.master_vars:
    print(f'{i.name}_{i.sp_index} = {i.value}')
  fmin = 0
  hmax = -inf
  for j in range(len(MDAO.subProblems)):
    print(f'SP_{MDAO.subProblems[j].index}: fmin= {MDAO.subProblems[j].MDA_process.getOutputs()}, hmin= {MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] , MDAO.subProblems[j].MDA_process.getOutputs())[1]}')
    fmin += sum(MDAO.subProblems[j].MDA_process.getOutputs())
    hmin= MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] , MDAO.subProblems[j].MDA_process.getOutputs())[1]
    if max(hmin) > hmax: 
      hmax = max(hmin) 
  print(f'P_main: fmin= {fmin}, hmax= {hmax}')
  print(f'Final obj value of the main problem: \n {fmin}')

def geometric_programming():
  v = {}
  V: List[variableData] = []
  s  = COUPLING_TYPE.SHARED
  ff = COUPLING_TYPE.FEEDFORWARD
  fb = COUPLING_TYPE.FEEDBACK
  un = COUPLING_TYPE.UNCOUPLED
  dum = COUPLING_TYPE.DUMMY


  names = ["z3", "z4", "z5", "z6",   "z7", "z3", "z8", "z9", "z10", "z11",   "z6", "z11", "z12", "z13", "z14", "dd"]
  spi =   [   1,    1,    1,		1,		  1,		2,		2,		2,		2,		  2,      3,     3,		  3,		 3,     3, 1]
  links = [   2, None, None,    3,   None,    1, None, None, None,      3,      1,     2,  None,  None,  None, None]
  coupling_t = \
          [ fb,    un,	 un,	 fb,		 un,	 ff,	 un,	 un,	 un,	    s,     ff,     s,    un,    un,    un, ff]
 
  lb =    [1e-6]*15
  ub =    [1e6]*15
  bl =    [1.]*15
  scaling = [9.e5] * 15

  lb.append(15)
  ub.append(18)
  bl.append(GP_A1([1]*5))
  scaling.append(1)

  # Variables dictionary with subproblems link
  for i in range(16):
    v[f"var{i+1}"] = {"index": i+1,
    "sp_index": spi[i],
    f"name": names[i],
    "dim": 1,
    "coupling_type": coupling_t[i],
    "link": links[i],
    "baseline": bl[i],
    "scaling": scaling[i],
    "lb": lb[i],
    "value": bl[i],
    "ub": ub[i]}

  for i in range(16):
    V.append(variableData(**v[f"var{i+1}"]))
  
  # Analyses setup; construct disciplinary analyses
  DA1: process = DA(inputs=[V[0], V[1], V[2], V[3], V[4]],
  outputs=[V[15]],
  blackbox=GP_A1,
  links=[2, 3],
  coupling_type=COUPLING_TYPE.FEEDBACK)
  
  DA2: process = DA(inputs=[V[6], V[7], V[8], V[9]],
  outputs=[V[5]],
  blackbox=GP_A2,
  links=[1, 3],
  coupling_type=COUPLING_TYPE.FEEDFORWARD
  )

  DA3: process = DA(inputs=[V[11], V[12], V[13], V[14]],
  outputs=[V[10]],
  blackbox=GP_A3,
  links=[1, 2],
  coupling_type=COUPLING_TYPE.FEEDFORWARD
  )

  # MDA setup; construct subproblems MDA
  sp1_MDA: process = MDA(nAnalyses=1, analyses = [DA1], variables=[V[0], V[1], V[2], V[3], V[4]], responses=[V[15]])
  sp2_MDA: process = MDA(nAnalyses=1, analyses = [DA2], variables=[V[6], V[7], V[8], V[9]], responses=[V[5]])
  sp3_MDA: process = MDA(nAnalyses=1, analyses = [DA3], variables=[V[11], V[12], V[13], V[14]], responses=[V[10]])

  # Construct the coordinator
  coord = ADMM(beta = 1.3,
  nsp=3,
  budget = 200,
  index_of_master_SP=1,
  display = True,
  scaling = 1.,
  mode = "serial",
  M_update_scheme= w_scheme.MEDIAN,
  store_q_io=True
  )

  # Construct subproblems
  sp1 = SubProblem(nv = 5,
  index = 1,
  vars = [V[0], V[1], V[2], V[3], V[4]],
  resps = [V[15]],
  is_main=1,
  analysis= sp1_MDA,
  coordination=coord,
  opt=GP_opt1,
  fmin_nop=np.inf,
  budget=5,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST,
  freal=15.)

  sp2 = SubProblem(nv = 4,
  index = 2,
  vars = [V[6], V[7], V[8], V[9]],
  resps = [V[5]],
  is_main=0,
  analysis= sp2_MDA,
  coordination=coord,
  opt=GP_opt2,
  fmin_nop=np.inf,
  budget=20,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST
  )

  sp3 = SubProblem(nv = 4,
  index = 3,
  vars = [V[11], V[12], V[13], V[14]],
  resps = [V[10]],
  is_main=0,
  analysis= sp3_MDA,
  coordination=coord,
  opt=GP_opt3,
  fmin_nop=np.inf,
  budget=20,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST)

# Construct MDO workflow
  MDAO: MDO = MDO(
  Architecture = MDO_ARCHITECTURE.IDF,
  Coordinator = coord,
  subProblems = [sp1, sp2, sp3],
  variables = V,
  responses = [V[5], V[10], V[15]],
  fmin = np.inf,
  hmin = np.inf,
  display = True,
  inc_stop = 1E-9,
  stop = "Iteration budget exhausted",
  tab_inc = [],
  noprogress_stop = 100
  )


# Run the MDO problem
  out = MDAO.run()
  print(f'------Run_Summary------')
  print(MDAO.stop)
  print(f'q = {MDAO.Coordinator.q}')
  xsp2 = []
  xsp3 = []
  index = np.argmax(MDAO.Coordinator.q)
  for i in MDAO.Coordinator.master_vars:
    if i.sp_index == MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[1,index]-1].link:
      xsp2.append(i.value)
    if i.sp_index == MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[0,index]-1].link:
      xsp3.append(i.value)
    print(f'{i.name}_{i.sp_index} = {i.value}')
  fmin = 0
  hmax = -inf
  for j in range(len(MDAO.subProblems)):
    print(f'SP_{MDAO.subProblems[j].index}: fmin= {MDAO.subProblems[j].MDA_process.getOutputs()}, hmin= {MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] , [])[1]}')
    if MDAO.subProblems[j].is_main:
      fmin = MDAO.subProblems[j].MDA_process.getOutputs()
      hmin= MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] , [])[1]
      if max(hmin) > hmax: 
        hmax = max(hmin) 
  print(f'P_main: fmin= {fmin}, hmax= {hmax}')
  print(f'Final obj value of the main problem: \n {fmin}')

  # Checking the impact of swapping z11_2 and z11_3 on the feasibility of SP2 and SP3, respectively
  
  temp = copy.deepcopy(xsp3[1])
  xsp3[1] = xsp2[-1]
  xsp2[-1] = temp
  print(f'For {MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[0,index]-1].name}_'
    f'{MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[0,index]-1].link} to '
      f'{MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[1,index]-1].name}_'
    f'{MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[1,index]-1].link}, using {MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[0,index]-1].name}_'
    f'{MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[0,index]-1].link}' 
    f' will make h_sp{MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[1,index]-1].link} = {GP_opt2(xsp2, 0)[1]}')
  print(f'For {MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[0,index]-1].name}_'
    f'{MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[0,index]-1].link} to '
      f'{MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[1,index]-1].name}_'
    f'{MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[1,index]-1].link}, using {MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[1,index]-1].name}_'
    f'{MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[1,index]-1].link}' 
    f' will make h_sp{MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[0,index]-1].link} = {GP_opt3(xsp3, 0)[1]}')


def Sellar_A1(x):
  return x[0] + x[1]**2 + x[2] - 0.2*x[3]

def Sellar_A2(x):
  return x[0] + x[1] + np.sqrt(x[2]) 

def Sellar_opt1(x, y):
  return [x[0]**2 + x[2] + y[0] + np.exp(-x[3]), [3.16-y[0]]]

def Sellar_opt2(x, y):
  return [0., [y[0]-24.]]

def Sellar():
  #  Sellar - Two discipline problem with IDF

  #  Variables grouping and problem setup
  x = {}
  X: List[variableData] = []
  # Define variable names
  N = ["x", "z1", "z2", "y1", "y2", "z1", "z2", "y1", "y2"]
  nx: int = len(N)
  # Subproblem indices: Indices should be non-zero
  J = [1,1,1,1,1,2,2,2,2]
  # Subproblems links
  L = [None, 2, 2, 2, 2, 1, 1, 1, 1]
  # Coupling types
  Ct = [COUPLING_TYPE.UNCOUPLED, 
        COUPLING_TYPE.SHARED, 
        COUPLING_TYPE.SHARED,
        COUPLING_TYPE.FEEDFORWARD, 
        COUPLING_TYPE.FEEDBACK, 
        COUPLING_TYPE.SHARED,
        COUPLING_TYPE.SHARED, 
        COUPLING_TYPE.FEEDBACK,
        COUPLING_TYPE.FEEDFORWARD]
  # Realistic lower bounds
  lb = [0, -10, 0, 3.16, 1.77763888346, -10, 0, 3.16, 1.77763888346]
  # Realistic upper bounds
  ub = [10.,10.,10., 115.2, 24., 10.,10., 115.2, 24.]

  # # Artificial lower bounds
  # lb = [0, -10, 0, 2., 1.5, -10, 0, 2., 1.5]
  # # Artificial upper bounds
  # ub = [10.,10.,10., 50., 50, 10.,10., 50., 50]

  # Bad artificial lower bounds
  # lb = [0, -10, 0, 0., 0., -10, 0, 0., 0.]
  # Bad artificial upper bounds
  # ub = [10.]*9

  # Baseline
  x0 = [0., 5., 5., 8.43, 7.848, 5., 5., 8.43, 7.848]
  # Scaling
  scaling = np.subtract(ub,lb)
  Qscaling = []
  # Create a dictionary for each variable
  for i in range(nx):
    x[f"var{i+1}"] = {"index": i+1,
    "sp_index": J[i],
    f"name": N[i],
    "dim": 1,
    "value": 0.,
    "coupling_type": Ct[i],
    "link": L[i],
    "baseline": x0[i],
    "scaling": scaling[i],
    "lb": lb[i],
    "value": x0[i],
    "ub": ub[i]}
    Qscaling.append(10./scaling[i] if 10./scaling[i] != np.inf and 10./scaling[i] != np.nan else 1.)

  # Instantiate the variableData class for each variable using its according dictionary
  for i in range(nx):
    X.append(variableData(**x[f"var{i+1}"]))


  # Analyses setup; construct disciplinary analyses
  DA1: process = DA(inputs=[X[0], X[1], X[2], X[4]],
  outputs=[X[3]],
  blackbox=Sellar_A1,
  links=2,
  coupling_type=COUPLING_TYPE.FEEDFORWARD
  )

  DA2: process = DA(inputs=[X[5], X[6], X[7]],
  outputs=[X[8]],
  blackbox=Sellar_A2,
  links=1,
  coupling_type=COUPLING_TYPE.FEEDFORWARD
  )

  sp1_MDA: process = MDA(nAnalyses=1, analyses = [DA1], variables=[X[0], X[1], X[2], X[4]], responses=[X[3]])
  sp2_MDA: process = MDA(nAnalyses=1, analyses = [DA2], variables=[X[5], X[6], X[7]], responses=[X[8]])


  # Construct the coordinator
  coord = ADMM(beta = 1.3,
  nsp=2,
  budget = 500,
  index_of_master_SP=1,
  display = True,
  scaling = Qscaling,
  mode = "serial",
  M_update_scheme= w_scheme.MAX,
  store_q_io=True
  )

  # Construct subproblems
  sp1 = SubProblem(nv = 4,
  index = 1,
  vars = [X[0], X[1], X[2], X[4]],
  resps = [X[3]],
  is_main=1,
  analysis= sp1_MDA,
  coordination=coord,
  opt=Sellar_opt1,
  fmin_nop=np.inf,
  budget=10,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST,
  freal=3.160,
  solver="scipy")

  sp2 = SubProblem(nv = 3,
  index = 2,
  vars = [X[5], X[6], X[7]],
  resps = [X[8]],
  is_main=0,
  analysis= sp2_MDA,
  coordination=coord,
  opt=Sellar_opt2,
  fmin_nop=np.inf,
  budget=10,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST
  )

  MDAO: MDO = MDO(
  Architecture = MDO_ARCHITECTURE.IDF,
  Coordinator = coord,
  subProblems = [sp1, sp2],
  variables = X,
  responses = [X[3], X[8]],
  fmin = np.inf,
  hmin = np.inf,
  display = True,
  inc_stop = 1E-9,
  stop = "Iteration budget exhausted",
  tab_inc = [],
  noprogress_stop = 1000
  )

  # Run the MDO problem
  out = MDAO.run()

  # Print summary output
  print(f'------Run_Summary------')
  print(MDAO.stop)
  print(f'q = {MDAO.Coordinator.q}')
  for i in MDAO.Coordinator.master_vars:
    print(f'{i.name}_{i.sp_index} = {i.value}')

  fmin = 0
  hmax = -np.inf
  for j in range(len(MDAO.subProblems)):
    print(f'SP_{MDAO.subProblems[j].index}: fmin= {MDAO.subProblems[j].MDA_process.getOutputs()}, hmin= {MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] , MDAO.subProblems[j].MDA_process.getOutputs())[1]}')
    fmin += sum(MDAO.subProblems[j].MDA_process.getOutputs())
    hmin= MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] , MDAO.subProblems[j].MDA_process.getOutputs())[1]
    if max(hmin) > hmax: 
      hmax = max(hmin) 
  print(f'P_main: fmin= {fmin}, hmax= {hmax}')
  print(f'Final obj value of the main problem: \n {fmin}')


def test_auto_build():
  p_file: str = os.path.abspath("./tests/test_files/Basic_MDO.yaml")
  MDAO: MDO = main(p_file, "build")
  for i in range(len(MDAO.subProblems)):
    temp :MDA = MDAO.subProblems[i].MDA_process
    for j in range(len(temp.analyses)):
      MDAO.subProblems[i].MDA_process.analyses[j].blackbox = globals()[MDAO.subProblems[i].MDA_process.analyses[j].blackbox]
    MDAO.subProblems[i].opt = globals()[MDAO.subProblems[i].opt]
  
  MDAO.run()


def test_comperhensive():
  Sellar()
  # speedReducer()
  # geometric_programming()

test_comperhensive()
