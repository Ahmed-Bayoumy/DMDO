import os
import platform
from DMDO import (
    ADMM, COUPLING_TYPE, DA, MDA, MDO,
    MDO_ARCHITECTURE, PSIZE_UPDATE, SubProblem, USER, main, process, variableData, w_scheme
)
import numpy as np
from numpy import sqrt, inf
import copy
from typing import List
from multiprocessing import freeze_support
from DMDO import logger

user =USER

def termTest():
  print("Termination criterria work!")

def A1(x):
  LAMBDA = 0.0
  for i in range(len(x)):
    if x[i] == 0.:
      x[i] = 1e-12
  return [np.log10(x[0]+LAMBDA) + np.log10(x[1]+LAMBDA) + np.log10(x[2]+LAMBDA)]

def A2(x):
  LAMBDA = 0.0
  for i in range(len(x)):
    if x[i] == 0.:
      x[i] = 1e-12
  return [np.divide(1., (x[0]+LAMBDA)) + np.divide(1., (x[1]+LAMBDA)) + np.divide(1., (x[2]+LAMBDA))]

def opt1(x, y, c=None):
  return [sum(x)+y[0], [0.]]

def opt2(x, y, c=None):
  return [0., [x[1]+y[0]-10.]]

def SR_A1(x):
  """ Speed reducer A1 """
  return [0.7854*x[0]*x[1]**2 *(3.3333*x[2]*x[2] + 14.9335*x[2] - 43.0934)]

def SR_A2(x):
  """ Speed reducer A2 """
  return [(-1.5079*x[0]*x[4]**2) + (7.477 * x[4]**3) + 0.7854 * x[3] * x[4]**2]

def SR_A3(x):
  """ Speed reducer A3 """
  return [-1.5079*x[0]*x[4]**2 + 7.477*x[4]**3 + 0.7854*x[3]*x[4]**2]

def SR_A4(x):
  return [x[0]+x[1]+x[2]]

def SR_opt1(x, y, *args):
  g5 = 27/(x[0]*x[1]**2*x[2]) -1
  g6 = 397.5/(x[0]*x[1]**2*x[2]**2) -1
  g9 = x[1]*x[2]/40 -1
  g10 = 5*x[1]/x[0] -1
  g11 = x[0]/(12*x[1]) -1
  return [y[0], [g5,g6,g9,g10,g11]]

def SR_opt2(x, y, *args):
  g1 = sqrt( ((745*x[3])/(x[1]*x[2]))**2 + 1.69e+6)/(110*x[4]**3) -1
  g3 = (1.5*x[4] + 1.9)/x[3] -1
  g7 = 1.93*x[3]**3/(x[1]*x[2]*x[4]**4) -1
  return [y[0], [g1,g3,g7]]

def SR_opt3(x, y, *args):
  g2 = sqrt( ((745*x[3])/(x[1]*x[2]))**2 + 1.575e+6)/(85*x[4]**3) -1
  g4 = (1.1*x[4] + 1.9)/x[3] -1
  g8 = (1.93*x[3]**3)/(x[1]*x[2]*x[4]**4) -1
  return [y[0], [g2, g4, g8]]

def SR_opt4(x, y, *args):
  return [y[0], [0]]

def GP_A1(z):
  z1 = sqrt(z[0]**2 + z[1]**-2 + z[2]**2)
  z2 = sqrt(z[2]**2 + z[3]**2  + z[4]**2)
  return [z1**2, z2**2]

def GP_opt1(z,y, *args):
  # if isinstance(y, list) and len(y) > 0:
  z1 = y[0]
  z2 = y[1]
  return [z1**2+z2**2, [z1**-2 + z[1]**2 - z[2]**2, z[2]**2 + z2**-2  - z[4]**2]]
  # else:
  #   return [z[0]**2 + z[3]**2, [z[0]**-2 + z[1]**2 - z[2]**2, z[2]**2 + z[3]**-2  - z[4]**2]]

def GP_A2(z):
  z3 = sqrt(z[0]**2 + z[1]**-2 + z[2]**-2 + z[3]**2)
  return [z3]

def GP_opt2(z,y, *args):
  return [0, [z[0]**2 + z[1]**2 - z[3]**2, z[0]**-2 + z[2]**2 - z[3]**2]]

def GP_A3(z):
  z6 = sqrt(z[0]**2 + z[1]**2 + z[2]**2 +z[3] **2)
  return [z6]

def GP_opt3(z, y, *args):
  return [0, [z[0]**2 + z[1]**-2 - z[2]**2, z[0]**2 +z[1]**2 - z[3]**2]]

def test_basic_MDO():
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
    "name": names[i],
    "dim": 1,
    "coupling_type": coupling_t[i],
    "link": links[i],
    "baseline": bl[i],
    "scaling": scaling[i],
    "lb": lb[i],
    "value": bl[i],
    "ub": ub[i],
    "type": "R"}
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
  coord = ADMM(beta = 1.3, gamma = 0.5,
  nsp=2,
  budget = 50,
  index_of_master_SP=1,
  display = True,
  scaling = Qscaling,
  mode = "serial",
  M_update_scheme= w_scheme.NORMAL,
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
  budget=150,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST,
  freal=2.625,
  solver="POLL")

  sp2 = SubProblem(nv = 3,
  index = 2,
  vars = [V[4], V[5], V[6]],
  resps = [V[7]],
  is_main=0,
  analysis= sp2_MDA,
  coordination=coord,
  opt=opt2,
  fmin_nop=np.inf,
  budget=150,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST,
  solver="POLL"
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
  p_file: str = os.path.abspath("./tests/test_files/Basic_MDO.out")
  MDAO.run(mode="serial")

  print('------Run_Summary------')
  print(MDAO.stop)
  print(f'q = {MDAO.Coordinator.q}')
  for i in MDAO.Coordinator.master_vars:
    print(f'{i.name}_{i.sp_index} = {i.value}')

  fmin = 0
  hmax = -inf
  
  for j in range(len(MDAO.subProblems)):
    sp_fmin, hmin = MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] \
      , MDAO.subProblems[j].MDA_process.getOutputs())
    print(f'SP_{MDAO.subProblems[j].index}:'
    f' fmin= {fmin}, '
    f'hmin= {hmin}'
    )
    fmin += sp_fmin
    if max(hmin) > hmax: 
      hmax = max(hmin) 
  print(f'P_main: fmin= {fmin}, hmax= {hmax}')
  print(f'Final obj value of the main problem: \n {fmin}')

def speedReducerOMADS():
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


  names = ["x1", "x2", "x3", "f1",   "x1", "x2", "x3", "x4", "x6", "f2",   "x1", "x2", "x3", "x5", "x7", "f3", "f1", "f2", "f3", "obj"]  # noqa: E501
  spi =   [   1,    1,    1,		1,		  2,		2,		2,		2,		2,		2,      3,    3,		3,		3,    3,	  3, 4, 4, 4, 4]  # noqa: E501
  links = [[2,3],[2,3],[2,3],   4,  [1,3],[1,3],[1,3], None, None,    4,  [1,2],[1,2],[1,2], None, None,    4, 1, 2, 3, None]
  lb =    [2.6 ,  0.7 ,  17., 722.,  2.6 ,  0.7,  17.,  7.3,  2.9, 184.,   2.6 ,  0.7,  17.,  7.3,   5.,942., f1min, f2min, f3min, f1min+f2min+f3min]  # noqa: E501
  ub =    [3.6 ,  0.8 ,  28.,5408.,  3.6 ,  0.8,  28.,  8.3,  3.9, 506.,   3.6 ,  0.8 , 28.,  8.3,  5.5,1369., f1max, f2max, f3max, f1max+f2max+f3max]  # noqa: E501
  bl =    np.add(lb, np.divide(np.subtract(ub, lb), 10.))

  bl[0] = 3.5
  bl[4] = 3.5
  bl[10] = 3.5
  
  coupling_t = \
          [ s,      s,		s,		ff,		s,		s,		s,		un,		un,	 ff,   s,    s,    s,   un,    un,    ff, fb, fb, fb, un]  # noqa: E501
 
  scaling = np.divide(np.subtract(ub, lb), 10.)
  Qscaling = []
  # Variables dictionary with subproblems link
  for i in range(20):
    v[f"var{i+1}"] = {"index": i+1,
    "sp_index": spi[i],
    "name": names[i],
    "dim": 1,
    "coupling_type": coupling_t[i],
    "link": links[i],
    "baseline": bl[i],
    "scaling": scaling[i],
    "lb": lb[i],
    "value": bl[i],
    "ub": ub[i],
    "type": "R"}
    Qscaling.append(10./scaling[i] if 10./scaling[i] != np.inf and 10./scaling[i] != np.nan else 1.)

  for i in range(20):
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

  DA4: process = DA(inputs=[V[16], V[17], V[18]],
  outputs=[V[19]],
  blackbox=SR_A4,
  links=[1,2,3],
  coupling_type=COUPLING_TYPE.FEEDFORWARD
  )
  

  # MDA setup; construct subproblems MDA
  sp1_MDA: process = MDA(nAnalyses=1, analyses = [DA1], variables=[V[0], V[1], V[2]], responses=[V[3]])
  sp2_MDA: process = MDA(nAnalyses=1, analyses = [DA2], variables=[V[4], V[5], V[6], V[7], V[8]], responses=[V[9]])
  sp3_MDA: process = MDA(nAnalyses=1, analyses = [DA3], variables=[V[10], V[11], V[12], V[13], V[14]], responses=[V[15]])
  sp4_MDA: process = MDA(nAnalyses=1, analyses = [DA4], variables=[V[16], V[17], V[18]], responses=[V[19]])


  # Construct the coordinator
  coord = ADMM(beta = 1.8, gamma = 0.5,
  nsp=4,
  budget = 50,
  index_of_master_SP=4,
  display = True,
  scaling = Qscaling,
  mode = "serial",
  M_update_scheme= w_scheme.MEDIAN,
  store_q_io=True
  )

  # Configurations 
  CSP1 = {}
  CSP1["search"] = {
        "type": "sampling",
        "s_method": "ACTIVE",
        "ns": 25,
        "visualize": False
            }
  CSP1["constraintsHandling"] = {
    "Barriers": ["PB","PB","PB","PB","PB"],
    "RHO": 0.0001,
    "h_max": 0.01
  }

  CSP2 = {}
  CSP2["search"] = {
        "type": "sampling",
        "s_method": "ACTIVE",
        "ns": 25,
        "visualize": False
            }
  CSP2["constraintsHandling"] = {
    "Barriers": ["PB","PB","PB"],
    "RHO": 0.0001,
    "h_max": 0.01
  }

  CSP3 = {}
  CSP3["search"] = {
        "type": "sampling",
        "s_method": "ACTIVE",
        "ns": 25,
        "visualize": False
            }
  CSP3["constraintsHandling"] = {
    "Barriers": ["PB","PB","PB"],
    "RHO": 0.0001,
    "h_max": 0.01
  }

  CSP4 = {}

  CSP4["search"] = {
        "type": "sampling",
        "s_method": "ACTIVE",
        "ns": 25,
        "visualize": False
            }
  CSP4["constraintsHandling"] = {
    "Barriers": ["EB","EB","EB"],
    "RHO": 0.0001,
    "h_max": 0.01
  }
  # Construct subproblems
  log = logger()
  sp1 = SubProblem(nv = 3,
  index = 1,
  vars = [V[0], V[1], V[2]],
  resps = [V[3]],
  is_main=0,
  analysis= sp1_MDA,
  coordination=coord,
  opt=SR_opt1,
  fmin_nop=np.inf,
  budget=150,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST,
  freal=2994.47,
  solver="MADS",
  conf=CSP1,
  log=log)

  sp2 = SubProblem(nv = 5,
  index = 2,
  vars = [V[4], V[5], V[6], V[7], V[8]],
  resps = [V[9]],
  is_main=0,
  analysis= sp2_MDA,
  coordination=coord,
  opt=SR_opt2,
  fmin_nop=np.inf,
  budget=150,
  display=False,
  psize = 10.,
  pupdate=PSIZE_UPDATE.MAX,
  solver="POLL",
  conf=CSP2,
  log=log
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
  budget=150,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.SUCCESS,
  solver="MADS",
  conf=CSP3,
  log=log)

  sp4 = SubProblem(nv = 3,
  index = 4,
  vars = [V[16], V[17], V[18]],
  resps = [V[19]],
  is_main=1,
  analysis= sp4_MDA,
  coordination=coord,
  opt=SR_opt4,
  fmin_nop=np.inf,
  budget=150,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST,
  solver="MADS",
  conf=CSP4,
  log=log)

# Construct MDO workflow
  MDAO: MDO = MDO(
  Architecture = MDO_ARCHITECTURE.IDF,
  Coordinator = coord,
  subProblems = [sp1, sp2, sp3, sp4],
  variables = V,
  responses = [V[3], V[9], V[15], V[19]],
  fmin = np.inf,
  hmin = np.inf,
  display = True,
  inc_stop = 1E-9,
  stop = "Iteration budget exhausted",
  tab_inc = [],
  noprogress_stop = 100
  )

  p_file: str = os.path.abspath("./tests/test_files/SR_Scipy.out")
# Run the MDO problem
  MDAO.run(p_file)

  print('------Run_Summary------')
  print(MDAO.stop)
  print(f'q = {MDAO.Coordinator.q}')
  for i in MDAO.Coordinator.master_vars:
    print(f'{i.name}_{i.sp_index} = {i.value}')
  fmin = 0
  hmax = -inf
  for j in range(len(MDAO.subProblems)):
    hmin= MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] , \
      MDAO.subProblems[j].MDA_process.getOutputs())[1]
    print(f'SP_{MDAO.subProblems[j].index}: fmin= {MDAO.subProblems[j].MDA_process.getOutputs()}, hmin='
    f'{hmin}')
    if MDAO.subProblems[j].is_main:
      fmin = sum(MDAO.subProblems[j].MDA_process.getOutputs())
    
    if max(hmin) > hmax: 
      hmax = max(hmin) 
  print(f'P_main: fmin= {fmin}, hmax= {hmax}')
  print(f'Final obj value of the main problem: \n {fmin}')

  return fmin, hmax, max(MDAO.Coordinator.q)

def speedReducerScipy():
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


  names = ["x1", "x2", "x3", "f1",   "x1", "x2", "x3", "x4", "x6", "f2",   "x1", "x2", "x3", "x5", "x7", "f3", "f1", "f2", "f3", "obj"]  # noqa: E501
  spi =   [   1,    1,    1,		1,		  2,		2,		2,		2,		2,		2,      3,    3,		3,		3,    3,	  3, 4, 4, 4, 4]  # noqa: E501
  links = [[2,3],[2,3],[2,3],   4,  [1,3],[1,3],[1,3], None, None,    4,  [1,2],[1,2],[1,2], None, None,    4, 1, 2, 3, None]
  lb =    [2.6 ,  0.7 ,  17., 722.,  2.6 ,  0.7,  17.,  7.3,  2.9, 184.,   2.6 ,  0.7,  17.,  7.3,   5.,942., f1min, f2min, f3min, f1min+f2min+f3min]  # noqa: E501
  ub =    [3.6 ,  0.8 ,  28.,5408.,  3.6 ,  0.8,  28.,  8.3,  3.9, 506.,   3.6 ,  0.8 , 28.,  8.3,  5.5,1369., f1max, f2max, f3max, f1max+f2max+f3max]  # noqa: E501
  bl =    np.add(lb, np.divide(np.subtract(ub, lb), 2.))

  bl[0] = 3.6
  bl[1] = 0.7
  bl[4] = 3.6
  bl[10] = 3.6
  
  coupling_t = \
          [ s,      s,		s,		ff,		s,		s,		s,		un,		un,	 ff,   s,    s,    s,   un,    un,    ff, fb, fb, fb, un]  # noqa: E501
 
  scaling = np.subtract(ub, lb)
  Qscaling = []
  # Variables dictionary with subproblems link
  for i in range(20):
    v[f"var{i+1}"] = {"index": i+1,
    "sp_index": spi[i],
    "name": names[i],
    "dim": 1,
    "coupling_type": coupling_t[i],
    "link": links[i],
    "baseline": bl[i],
    "scaling": scaling[i],
    "lb": lb[i],
    "value": bl[i],
    "ub": ub[i],
    "type": "R"}
    Qscaling.append(1./scaling[i] if 1./scaling[i] != np.inf and 1./scaling[i] != np.nan else 1.)

  for i in range(20):
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

  DA4: process = DA(inputs=[V[16], V[17], V[18]],
  outputs=[V[19]],
  blackbox=SR_A4,
  links=[1,2,3],
  coupling_type=COUPLING_TYPE.FEEDFORWARD
  )
  

  # MDA setup; construct subproblems MDA
  sp1_MDA: process = MDA(nAnalyses=1, analyses = [DA1], variables=[V[0], V[1], V[2]], responses=[V[3]])
  sp2_MDA: process = MDA(nAnalyses=1, analyses = [DA2], variables=[V[4], V[5], V[6], V[7], V[8]], responses=[V[9]])
  sp3_MDA: process = MDA(nAnalyses=1, analyses = [DA3], variables=[V[10], V[11], V[12], V[13], V[14]], responses=[V[15]])
  sp4_MDA: process = MDA(nAnalyses=1, analyses = [DA4], variables=[V[16], V[17], V[18]], responses=[V[19]])


  # Construct the coordinator
  coord = ADMM(beta = 1.3,gamma = 0.5,
  nsp=4,
  budget = 50,
  index_of_master_SP=4,
  display = True,
  scaling = Qscaling,
  mode = "serial",
  M_update_scheme= w_scheme.MEDIAN,
  store_q_io=True
  )

  # Configurations 

  # Construct subproblems
  sp1 = SubProblem(nv = 3,
  index = 1,
  vars = [V[0], V[1], V[2]],
  resps = [V[3]],
  is_main=0,
  analysis= sp1_MDA,
  coordination=coord,
  opt=SR_opt1,
  fmin_nop=np.inf,
  budget=150,
  display=False,
  freal=2994.47,
  solver="scipy",
  scipy={"method": 'SLSQP',
          "options": {"disp": False,
                      "verbose": 0},
          "is_con": True,
          "tol": 1E-12}
  )

  sp2 = SubProblem(nv = 5,
  index = 2,
  vars = [V[4], V[5], V[6], V[7], V[8]],
  resps = [V[9]],
  is_main=0,
  analysis= sp2_MDA,
  coordination=coord,
  opt=SR_opt2,
  fmin_nop=np.inf,
  budget=150,
  display=False,
  solver="scipy",
  scipy={"method": 'SLSQP',
          "options": {"disp": False,
                      "verbose": 0},
          "is_con": True,
          "tol": 1E-12}
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
  budget=150,
  display=False,
  solver="scipy",
  scipy={"method": 'SLSQP',
          "options": {"disp": False,
                      "verbose": 0},
          "is_con": True,
          "tol": 1E-12}
  )

  sp4 = SubProblem(nv = 3,
  index = 4,
  vars = [V[16], V[17], V[18]],
  resps = [V[19]],
  is_main=1,
  analysis= sp4_MDA,
  coordination=coord,
  opt=SR_opt4,
  fmin_nop=np.inf,
  budget=150,
  display=False,
  solver="scipy",
  scipy={"method": 'SLSQP',
          "options": {"disp": False,
                      "verbose": 0},
          "is_con": True,
          "tol": 1E-12}
  )

# Construct MDO workflow
  MDAO: MDO = MDO(
  Architecture = MDO_ARCHITECTURE.IDF,
  Coordinator = coord,
  subProblems = [sp1, sp2, sp3, sp4],
  variables = V,
  responses = [V[3], V[9], V[15], V[19]],
  fmin = np.inf,
  hmin = np.inf,
  display = True,
  inc_stop = 1E-9,
  stop = "Iteration budget exhausted",
  tab_inc = [],
  noprogress_stop = 100
  )


# Run the MDO problem
  p_file: str = os.path.abspath("./tests/test_files/SR_Scipy.out")
  MDAO.run(p_file)

  print('------Run_Summary------')
  print(MDAO.stop)
  print(f'q = {MDAO.Coordinator.q}')
  for i in MDAO.Coordinator.master_vars:
    print(f'{i.name}_{i.sp_index} = {i.value}')
  fmin = 0
  hmax = -inf
  for j in range(len(MDAO.subProblems)):
    print(f'SP_{MDAO.subProblems[j].index}: fmin= {MDAO.subProblems[j].MDA_process.getOutputs()}, '
          f'hmin= {MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] , MDAO.subProblems[j].MDA_process.getOutputs())[1]}')  # noqa: E501
    if MDAO.subProblems[j].is_main:
      fmin = sum(MDAO.subProblems[j].MDA_process.getOutputs())
    hmin= MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] , \
                                  MDAO.subProblems[j].MDA_process.getOutputs())[1]
    if max(hmin) > hmax: 
      hmax = max(hmin) 
  print(f'P_main: fmin= {fmin}, hmax= {hmax}')
  print(f'Final obj value of the main problem: \n {fmin}')

  return fmin, hmax, max(MDAO.Coordinator.q)

def geometric_programming():
  v = {}
  V: List[variableData] = []
  s  = COUPLING_TYPE.SHARED
  ff = COUPLING_TYPE.FEEDFORWARD
  fb = COUPLING_TYPE.FEEDBACK
  un = COUPLING_TYPE.UNCOUPLED


  names = ["z2", "z4", "z5", "z3",   "z7", "z2", "z8", "z9", "z10", "z11",   "z3", "z11", "z12", "z13", "z14"]
  spi =   [   1,    1,    1,		1,		  1,		2,		2,		2,		 2,	    2,      3,     3,		  3,		 3,     3]  # noqa: E501
  links = [   2, None, None,    3,   None,    1, None, None,  None,     3,      1,     2,  None,  None,  None]
  coupling_t = \
          [  fb,    un,	 un,	 fb,		 un,	 ff,	 un,	 un,	  un,	    s,      ff,     s,    un,    un,    un]
 
  lb =    [1e-6]*16
  ub =    [1E6]*16
 
  


  bl = np.subtract(ub, lb)/2


  scaling = np.subtract(ub, lb)/10
  Qscaling = []
  # Variables dictionary with subproblems link
  for i in range(15):
    v[f"var{i+1}"] = {"index": i+1,
    "sp_index": spi[i],
    "name": names[i],
    "dim": 1,
    "coupling_type": coupling_t[i],
    "link": links[i],
    "baseline": bl[i],
    "scaling": scaling[i],
    "lb": lb[i],
    "value": bl[i],
    "ub": ub[i],
    "type": "R"}
    Qscaling.append(1./scaling[i] if 1./scaling[i] != np.inf and 1./scaling[i] != np.nan else 1.)

  for i in range(15):
    V.append(variableData(**v[f"var{i+1}"]))
  
  # Analyses setup; construct disciplinary analyses
  DA1: process = DA(inputs=[V[0], V[1], V[2], V[3], V[4]],
  outputs=[V[5], V[10]],
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
  sp1_MDA: process = MDA(nAnalyses=1, analyses = [DA1], variables=[V[0], V[1], V[2], V[3], V[4]], responses=[V[5], V[10]])
  sp2_MDA: process = MDA(nAnalyses=1, analyses = [DA2], variables=[V[6], V[7], V[8], V[9]], responses=[V[5]])
  sp3_MDA: process = MDA(nAnalyses=1, analyses = [DA3], variables=[V[11], V[12], V[13], V[14]], responses=[V[10]])

  # Construct the coordinator
  coord = ADMM(beta = 1.3,gamma = 0.5,
  nsp=3,
  budget = 100,
  index_of_master_SP=1,
  display = True,
  scaling = Qscaling,
  mode = "serial",
  M_update_scheme= w_scheme.MEDIAN,
  store_q_io=False
  )

  # Configurations 
  CSP1 = {}
  CSP1 ["options"] = {
          "seed": 10000,
          "budget": 500,
          "tol": 1E-4,
          "psize_init": 2.,
          "display": False,
          "opportunistic": False,
          "check_cache": True,
          "store_cache": True,
          "collect_y": False,
          "rich_direction": True,
          "precision": "high",
          "save_results": True,
          "save_coordinates": False,
          "save_all_best": False,
          "parallel_mode": False
        }
  CSP1["search"] = {
        "type": "sampling",
        "s_method": "ACTIVE",
        "ns": 50,
        "visualize": False
            }
  CSP1["constraintsHandling"] = {
    "Barriers": ["EB","EB"],
    "LAMBDA": [1E5, 1E5],
    "RHO": 1,
    "h_max": 0
  }

  CSP2 = {}
  CSP2 ["options"] = {
          "seed": 10000,
          "budget": 500,
          "tol": 1E-4,
          "psize_init": 1,
          "display": False,
          "opportunistic": False,
          "check_cache": True,
          "store_cache": True,
          "collect_y": False,
          "rich_direction": True,
          "precision": "high",
          "save_results": False,
          "save_coordinates": False,
          "save_all_best": False,
          "parallel_mode": False
        }
  CSP2["search"] = {
        "type": "sampling",
        "s_method": "ACTIVE",
        "ns": 50,
        "visualize": False
            }
  CSP2["constraintsHandling"] = {
    "Barriers": ["EB","EB"],
    "LAMBDA": [1E5, 1E5],
    "RHO": 1,
    "h_max": 0
  }

  CSP3 = {}
  CSP3 ["options"] = {
          "seed": 10000,
          "budget": 500,
          "tol": 1E-4,
          "psize_init": 1,
          "display": False,
          "opportunistic": False,
          "check_cache": True,
          "store_cache": True,
          "collect_y": False,
          "rich_direction": True,
          "precision": "high",
          "save_results": False,
          "save_coordinates": False,
          "save_all_best": False,
          "parallel_mode": False
        }
  CSP3["search"] = {
        "type": "sampling",
        "s_method": "ACTIVE",
        "ns": 50,
        "visualize": False
            }
  CSP3["constraintsHandling"] = {
    "Barriers": ["EB","EB"],
    "LAMBDA": [1E5, 1E5],
    "RHO": 1,
    "h_max": 0
  }

  # Construct subproblems
  sp1 = SubProblem(nv = 5,
  index = 1,
  vars = [V[0], V[1], V[2], V[3], V[4]],
  resps = [V[5], V[10]],
  is_main=1,
  analysis= sp1_MDA,
  coordination=coord,
  opt=GP_opt1,
  fmin_nop=np.inf,
  budget=500,
  display=False,
  psize = 2.,
  pupdate=PSIZE_UPDATE.LAST,
  solver="MADS",
  conf=CSP1,
  realistic_objective=False)
  # nis = 500
  # vlim = np.empty((5, 2))
  

  sp2 = SubProblem(nv = 4,
  index = 2,
  vars = [V[6], V[7], V[8], V[9]],
  resps = [V[5]],
  is_main=0,
  analysis= sp2_MDA,
  coordination=coord,
  opt=GP_opt2,
  fmin_nop=np.inf,
  budget=500,
  display=False,
  psize = 2.,
  pupdate=PSIZE_UPDATE.LAST,
  solver="MADS",
  tol=1E-4,
  conf=CSP2)

  # vlim = np.empty((4, 2))
  # for i in range(4):
  #   llb = copy.deepcopy(lb[i])
  #   uub = copy.deepcopy(ub[i])
  #   vlim[i] = [np.array(llb), np.array(uub)]
  # LHS = samplerIndependent.LHS(nis, vlim)
  # samples2 = LHS.generate_samples()

  sp3 = SubProblem(nv = 4,
  index = 3,
  vars = [V[11], V[12], V[13], V[14]],
  resps = [V[10]],
  is_main=0,
  analysis= sp3_MDA,
  coordination=coord,
  opt=GP_opt3,
  fmin_nop=np.inf,
  budget=500,
  display=False,
  psize = 2.,
  pupdate=PSIZE_UPDATE.LAST,
  solver="MADS",
  tol=1E-4,
  conf=CSP3)

  # vlim = np.empty((4, 2))
  # for i in range(4):
  #   llb = copy.deepcopy(lb[i])
  #   uub = copy.deepcopy(ub[i])
  #   vlim[i] = [np.array(llb), np.array(uub)]
  # LHS = samplerIndependent.LHS(nis, vlim)
  # samples3 = LHS.generate_samples()

# Construct MDO workflow
  MDAO: MDO = MDO(
  Architecture = MDO_ARCHITECTURE.IDF,
  Coordinator = coord,
  subProblems = [sp1, sp2, sp3],
  variables = V,
  responses = [V[5], V[10]],
  fmin = np.inf,
  hmin = np.inf,
  display = True,
  inc_stop = 1E-9,
  stop = "Iteration budget exhausted",
  tab_inc = [],
  noprogress_stop = 100
  )

# Run the MDO problem
  p_file: str = os.path.abspath("./tests/test_files/GP.out")
  MDAO.run(p_file)
  print('------Run_Summary------')
  print(MDAO.stop)
  print(f'q = {MDAO.Coordinator.q}')
  for i in MDAO.Coordinator.master_vars:
    print(f'{i.name}_{i.sp_index} = {i.value}')
  fmin = 0
  hmax = -inf
  for j in range(len(MDAO.subProblems)):
    MDAO.subProblems[j].MDA_process.run()
    Y = copy.deepcopy(MDAO.subProblems[j].MDA_process.getOutputs())
    print(f'SP_{MDAO.subProblems[j].index}: fmin= {MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] , Y)[0]}, hmin= {MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] , Y)[1]}')  # noqa: E501
    if MDAO.subProblems[j].is_main:
      fmin += MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] , Y)[0]
    hmin= MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] , Y)[1]
    if max(hmin) > hmax: 
      hmax = max(hmin) 
  print(f'P_main: fmin= {fmin}, hmax= {hmax}')
  print(f'Final obj value of the main problem: \n {fmin}')

  return fmin, hmax, max(MDAO.Coordinator.q)

def Sellar_A1(x):
  return [x[0] + x[1]**2 + x[2] - 0.2*x[3]]

def Sellar_A2(x):
  return [x[0] + x[1] + np.sqrt(x[2])]

def Sellar_opt1(x, y, *args):
  return [x[0]**2 + x[2] + y[0] + np.exp(-x[3]), [3.16-y[0]]]

def Sellar_opt2(x, y, *args):
  return [0., [y[0]-24.]]

def Sellar_scipy():
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
  x0 = [1., 5., 2., 0., 0., 5., 2., 0., 0.]
  # Scaling
  scaling = np.subtract(ub,lb)
  Qscaling = []
  # Create a dictionary for each variable
  for i in range(nx):
    x[f"var{i+1}"] = {"index": i+1,
    "sp_index": J[i],
    "name": N[i],
    "dim": 1,
    "coupling_type": Ct[i],
    "link": L[i],
    "baseline": x0[i],
    "scaling": scaling[i],
    "lb": lb[i],
    "value": x0[i],
    "ub": ub[i],
    "type": "R"}
    Qscaling.append(1./scaling[i] if 1./scaling[i] != np.inf and 1./scaling[i] != np.nan else 1.)

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
  coord = ADMM(beta = 2.0,gamma = 0.5,
  nsp=2,
  budget = 100,
  index_of_master_SP=1,
  display = True,
  scaling = Qscaling,
  mode = "serial",
  M_update_scheme= w_scheme.MEDIAN,
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
  budget=150,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST,
  freal=3.160,
  solver="scipy",
  scipy={"method": 'SLSQP',
          "options": {"disp": False,
                      "verbose": 0},
          "is_con": True,
          "tol": 1E-8})

  sp2 = SubProblem(nv = 3,
  index = 2,
  vars = [X[5], X[6], X[7]],
  resps = [X[8]],
  is_main=0,
  analysis= sp2_MDA,
  coordination=coord,
  opt=Sellar_opt2,
  fmin_nop=np.inf,
  budget=150,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST,
  solver="scipy",
  scipy={"method": 'SLSQP',
          "options": {"disp": False,
                      "verbose": 0},
          "is_con": True,
          "tol": 1E-8})

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
  noprogress_stop = 100)

  # Run the MDO problem
  p_file: str = os.path.abspath("./tests/test_files/Sellar_Scipy.out")
  MDAO.run(p_file)

  # Print summary output
  print('------Run_Summary------')
  print(MDAO.stop)
  print(f'q = {MDAO.Coordinator.q}')
  for i in MDAO.Coordinator.master_vars:
    print(f'{i.name}_{i.sp_index} = {i.value}')

  fmin_main = 0
  hmax = -np.inf
  hmax_main = hmax
  for j in range(len(MDAO.subProblems)):
    fmin = MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] , \
                                   MDAO.subProblems[j].MDA_process.getOutputs())[0]
    hmax = MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] , \
                                   MDAO.subProblems[j].MDA_process.getOutputs())[1]
    print(f'SP_{MDAO.subProblems[j].index}: fmin= {fmin}, hmin= {hmax}')
    if MDAO.subProblems[j].is_main:
      fmin_main = fmin
      hmax_main = hmax

  print(f'P_main: fmin= {fmin_main}, hmax= {hmax_main}')
  print(f'Final obj value of the main problem: \n {fmin_main}')

  return fmin_main, hmax_main, max(MDAO.Coordinator.q)

def Sellar_OMADS_POLL():
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
  # x0 = [1., 5., 2., 0., 0., 5., 2., 0., 0.]
  x0 = [1., 5., 2., 3.16, 2., 5., 2., 3.16, 2.]

  # Scaling
  scaling = np.subtract(ub,lb)
  Qscaling = []
  # Create a dictionary for each variable
  for i in range(nx):
    x[f"var{i+1}"] = {"index": i+1,
    "sp_index": J[i],
    "name": N[i],
    "dim": 1,
    "coupling_type": Ct[i],
    "link": L[i],
    "baseline": x0[i],
    "scaling": scaling[i],
    "lb": lb[i],
    "value": x0[i],
    "ub": ub[i],
    "type": "R"}
    Qscaling.append(1./scaling[i] if 1./scaling[i] != np.inf and 1./scaling[i] != np.nan else 1.)

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
  coord = ADMM(beta = 1.1,gamma = 0.5,
  nsp=2,
  budget = 50,
  index_of_master_SP=1,
  display = True,
  scaling = Qscaling,
  mode = "parallel",
  M_update_scheme= w_scheme.MEDIAN,
  store_q_io=True
  )

  # Construct subproblems
  CSP1 = {"options": {},
          "search": {}}
  CSP1["options"] = {
        "seed": 10000,
        "budget": 150,
        "tol": 0.0000000000001,
        "psize_init": 1,
        "display": False,
        "opportunistic": False,
        "check_cache": True,
        "store_cache": True,
        "collect_y": False,
        "rich_direction": True,
        "precision": "high",
        "save_results": False,
        "save_coordinates": False,
        "save_all_best": False,
        "parallel_mode": False
      }
  CSP1["search"] = {
        "type": "sampling",
        "s_method": "LH",
        "ns": 3,
        "visualize": False
            }
  CSP1["constraintsHandling"] = {
    "Barriers": ["PB"],
    "RHO": 0.0001,
    "h_max": 10
  }

  CSP2 = {}
  CSP2["options"] = {
        "seed": 10000,
        "budget": 150,
        "tol": 0.0000000000001,
        "psize_init": 1,
        "display": False,
        "opportunistic": False,
        "check_cache": True,
        "store_cache": True,
        "collect_y": False,
        "rich_direction": True,
        "precision": "high",
        "save_results": False,
        "save_coordinates": False,
        "save_all_best": False,
        "parallel_mode": False
      }
  CSP2["search"] = {
        "type": "sampling",
        "s_method": "LH",
        "ns": 3,
        "visualize": False
            }
  CSP2["constraintsHandling"] = {
    "Barriers": ["PB"],
    "RHO": 0.0001,
    "h_max": 10
  }
  sp1 = SubProblem(nv = 4,
  index = 1,
  vars = [X[0], X[1], X[2], X[4]],
  resps = [X[3]],
  is_main=1,
  analysis= sp1_MDA,
  coordination=coord,
  opt=Sellar_opt1,
  fmin_nop=np.inf,
  budget=150,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST,
  freal=3.0,
  solver="POLL",
  conf=CSP1)

  sp2 = SubProblem(nv = 3,
  index = 2,
  vars = [X[5], X[6], X[7]],
  resps = [X[8]],
  is_main=0,
  analysis= sp2_MDA,
  coordination=coord,
  opt=Sellar_opt2,
  fmin_nop=np.inf,
  budget=150,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST,
  solver="POLL",
  conf=CSP2)

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
  noprogress_stop = 100)

  # Run the MDO problem
  p_file: str = os.path.abspath("./tests/test_files/Sellar_OMADS.out")
  MDAO.run(p_file)

  # Print summary output
  print('------Run_Summary------')
  print(MDAO.stop)
  print(f'q = {MDAO.Coordinator.q}')
  for i in MDAO.Coordinator.master_vars:
    print(f'{i.name}_{i.sp_index} = {i.value}')

  fmin_main = 0
  hmax = -np.inf
  hmax_main = hmax
  for j in range(len(MDAO.subProblems)):
    fmin = MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] , \
                                   MDAO.subProblems[j].MDA_process.getOutputs())[0]
    hmax = MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] , \
                                   MDAO.subProblems[j].MDA_process.getOutputs())[1]
    print(f'SP_{MDAO.subProblems[j].index}: fmin= {fmin}, hmin= {hmax}')
    if MDAO.subProblems[j].is_main:
      fmin_main = fmin
      hmax_main = hmax

  print(f'P_main: fmin= {fmin_main}, hmax= {hmax_main}')
  print(f'Final obj value of the main problem: \n {fmin_main}')

  return fmin_main, hmax_main, max(MDAO.Coordinator.q)

def Sellar_OMADS_MADS():
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
  # x0 = [1., 5., 2., 0., 0., 5., 2., 0., 0.]
  x0 = [1., 5., 2., 3.16, 2., 5., 2., 3.16, 2.]

  # Scaling
  scaling = np.subtract(ub,lb)
  Qscaling = []
  # Create a dictionary for each variable
  for i in range(nx):
    x[f"var{i+1}"] = {"index": i+1,
    "sp_index": J[i],
    "name": N[i],
    "dim": 1,
    "coupling_type": Ct[i],
    "link": L[i],
    "baseline": x0[i],
    "scaling": scaling[i],
    "lb": lb[i],
    "value": x0[i],
    "ub": ub[i],
    "type": "R"}
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
  coord = ADMM(beta = 1.3,gamma = 0.5,
  nsp=2,
  budget = 25,
  index_of_master_SP=1,
  display = True,
  scaling = Qscaling,
  mode = "serial",
  M_update_scheme= w_scheme.MEDIAN,
  store_q_io=True
  )

  # Construct subproblems
  CSP1 = {"options": {},
          "search": {}}
  CSP1["options"] = {
        "seed": 10000,
        "budget": 150,
        "tol": 0.0000000000001,
        "psize_init": 1,
        "display": False,
        "opportunistic": False,
        "check_cache": True,
        "store_cache": True,
        "collect_y": False,
        "rich_direction": True,
        "precision": "high",
        "save_results": False,
        "save_coordinates": False,
        "save_all_best": False,
        "parallel_mode": False
      }
  CSP1["search"] = {
        "type": "sampling",
        "s_method": "ACTIVE",
        "ns": 50,
        "visualize": False
            }
  CSP1["constraintsHandling"] = {
    "Barriers": ["PB"],
    "RHO": 0.0001,
    "h_max": 10
  }

  CSP2 = {}
  CSP2["options"] = {
        "seed": 10000,
        "budget": 150,
        "tol": 0.0000000000001,
        "psize_init": 1,
        "display": False,
        "opportunistic": False,
        "check_cache": True,
        "store_cache": True,
        "collect_y": False,
        "rich_direction": True,
        "precision": "high",
        "save_results": False,
        "save_coordinates": False,
        "save_all_best": False,
        "parallel_mode": False
      }
  CSP2["search"] = {
        "type": "sampling",
        "s_method": "ACTIVE",
        "ns": 20,
        "visualize": False
            }
  CSP2["constraintsHandling"] = {
    "Barriers": ["PB"],
    "RHO": 0.0001,
    "h_max": 10
  }
  sp1 = SubProblem(nv = 4,
  index = 1,
  vars = [X[0], X[1], X[2], X[4]],
  resps = [X[3]],
  is_main=1,
  analysis= sp1_MDA,
  coordination=coord,
  opt=Sellar_opt1,
  fmin_nop=np.inf,
  budget=150,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST,
  solver="MADS",
  conf=CSP1)

  sp2 = SubProblem(nv = 3,
  index = 2,
  vars = [X[5], X[6], X[7]],
  resps = [X[8]],
  is_main=0,
  analysis= sp2_MDA,
  coordination=coord,
  opt=Sellar_opt2,
  fmin_nop=np.inf,
  budget=150,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST,
  solver="MADS",
  conf=CSP2)

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
  noprogress_stop = 100)

  # Run the MDO problem
  p_file: str = os.path.abspath("./tests/test_files/Sellar_OMADS_MADS.out")
  MDAO.run(p_file)

  # Print summary output
  print('------Run_Summary------')
  print(MDAO.stop)
  print(f'q = {MDAO.Coordinator.q}')
  for i in MDAO.Coordinator.master_vars:
    print(f'{i.name}_{i.sp_index} = {i.value}')

  fmin_main = 0
  hmax = -np.inf
  hmax_main = hmax
  for j in range(len(MDAO.subProblems)):
    fmin = MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] , \
      MDAO.subProblems[j].MDA_process.getOutputs())[0]
    hmax = MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] , \
      MDAO.subProblems[j].MDA_process.getOutputs())[1]
    print(f'SP_{MDAO.subProblems[j].index}: fmin= {fmin}, hmin= {hmax}')
    if MDAO.subProblems[j].is_main:
      fmin_main = fmin
      hmax_main = hmax

  print(f'P_main: fmin= {fmin_main}, hmax= {hmax_main}')
  print(f'Final obj value of the main problem: \n {fmin_main}')

  return fmin_main, hmax_main, max(MDAO.Coordinator.q)

def test_auto_build():
  p_file: str = os.path.abspath("./tests/test_files/Basic_MDO.yaml")
  MDAO: MDO = main({'setup_file': p_file, 'run_mode': 'build', 'mdo_name': 'Basic_auto', 'working_dir': '.'})
  for i in range(len(MDAO.subProblems)):
    temp :MDA = MDAO.subProblems[i].MDA_process
    for j in range(len(temp.analyses)):
      MDAO.subProblems[i].MDA_process.analyses[j].blackbox = globals()[MDAO.subProblems[i].MDA_process.analyses[j].blackbox]
    MDAO.subProblems[i].opt = globals()[MDAO.subProblems[i].opt]
     
def test_Sellar():
  f, h, qmax = Sellar_scipy()
  if abs(f-3.18339395045)/3.18339395045 > 0.22 or max(h)>0.001 or qmax > 1E-4:
    raise IOError(f"Sellar_scipy failed the checking criteria f_diff= {abs(f-3.18339395045)/3.18339395045},"
    f" hmax= {h}, qmax= {qmax}")
  # TODO: This test is only failing on MAC with the 'POLL' local solver because of
  # TODO: recent updates in the numpy package version used. 
  # TODO: This issue has been fixed in an OMADS version that will be released in 2026
  isMac = platform.platform().split('-')[0] != 'macOS'
  if (isMac):
    f, h, qmax = Sellar_OMADS_POLL()
    if abs(f-3.18339395045)/3.18339395045 > 0.22 or max(h)>0.001 or qmax > 1E-4:
      raise IOError(f"Sellar_POLL failed the checking criteria f_diff= {abs(f-3.18339395045)/3.18339395045},"
      f" hmax= {h}, qmax= {qmax}")
  f, h, qmax = Sellar_OMADS_MADS()
  if abs(f-3.18339395045)/3.18339395045 > 0.13 or max(h)>0.001 or qmax > 1E-3:
    raise IOError(f"Sellar_MADS failed the checking criteria f_diff= {abs(f-3.18339395045)/3.18339395045},"
    f" hmax= {h}, qmax= {qmax}")

def test_speedReducer():
  f, h, qmax = speedReducerScipy()
  if abs(f-2713.6640204584155)/2713.6640204584155 > 0.05 or h>0.06 or qmax > 1E-3:
    raise IOError(f"SR_scipy failed the checking criteria f_diff= {abs(f-2713.6640204584155)/2713.6640204584155},"
    f" hmax= {h}, qmax= {qmax}")
  f, h, qmax = speedReducerOMADS()
  if abs(f-2713.6640204584155)/2713.6640204584155 > 0.07 or h>0.0001 or qmax > 5E-3:
    raise IOError(f"SR_OMADS failed the checking criteria f_diff= {abs(f-2713.6640204584155)/2713.6640204584155},"
    f" hmax= {h}, qmax= {qmax}")
  
def test_geometric_programming():
  f, h, qmax = geometric_programming()
  if abs(f-15)/15 > 0.35 or h>0. or qmax > 1E-4:
    raise IOError(f"GP_OMADS failed the checking criteria f_diff= {abs(f-15)/15}, hmax= {h}, qmax= {qmax}")
  

if __name__ == "__main__":
  freeze_support()