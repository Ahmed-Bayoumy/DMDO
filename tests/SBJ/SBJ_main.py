import os

from DMDO import MDA, MDO, USER, main

from SSBJ_Aircraft import SSBJ_Aircraft
from SSBJ_Aerodynamics import SSBJAerodynamics
from SSBJ_Propulsion import SSBJPropulsion
from SSBJ_Structures import WingDesignAnalyzer
import warnings
warnings.filterwarnings("ignore")

# Rest of your code follows


user = USER

user.h = 55000
user.M = 1.4

def SBJ_aircraft_analysis(x):
  ssbj_aircraft = SSBJ_Aircraft(We=x[1], Ws=x[3], Wf=x[4], SFCp=x[0], LDr=x[2])
  return ssbj_aircraft.SBJ_aircraft_analysis()

def SBJ_propulsion_analysis(x):
  ssbj_prop = SSBJPropulsion(D=x[0], T=x[1])
  return ssbj_prop.SBJ_propulsion_analysis()

def SBJ_aerodynamics_analysis(x):
  ssbj_prop = SSBJAerodynamics(Wt=x[0], ESFp=x[1], theta=x[2], tc=x[3], ARw=x[4], LAMBDAw=x[5], Sref=x[6], Sht=x[7], \
                               ARht=x[8], LAMBDAht=x[9], Lw=x[10], Lht=x[11])
  return ssbj_prop.SBJ_aerodynamics_analysis()

def SBJ_structure_analysis(x):
  ssbj_structure = WingDesignAnalyzer(Lift=x[0], tc=x[1],ARw=x[2], LAMBDAw=x[3], Sref=x[4], Sht=x[5], ARht=x[6], \
                                      lambdatr=x[7], t=x[8:17], ts=x[17:])
  return ssbj_structure.SBJ_structure_analysis()

def SBJ_aircraft_opt(x, y, *args):
  ssbj_aircraft = SSBJ_Aircraft(We=x[1], Ws=x[3], Wf=x[4], SFCp=x[0], LDr=x[2])
  return ssbj_aircraft.SBJ_aircraft_opt(Wt=y[0], range=y[1])

def SBJ_propulsion_opt(x, y, *args):
  ssbj_prop = SSBJPropulsion(D=x[0], T=x[1])
  return ssbj_prop.SBJ_propulsion_opt(Temp_E=y[3], Throttle_uA=y[4])

def SBJ_aerodynamics_opt(x, y, *args):
  ssbj_prop = SSBJAerodynamics(Wt=x[0], ESFp=x[1], theta=x[2], tc=x[3], ARw=x[4], LAMBDAw=x[5], Sref=x[6], Sht=x[7], \
                               ARht=x[8], LAMBDAht=x[9], Lw=x[10], Lht=x[11])
  return ssbj_prop.SBJ_aerodynamics_opt(Pg=y[3], CLo=[y[4], y[5]])

def SBJ_structure_opt(x, y, *args):
  ssbj_structure = WingDesignAnalyzer(Lift=x[0], tc=x[1],ARw=x[2], LAMBDAw=x[3], Sref=x[4], Sht=x[5], ARht=x[6], \
                                      lambdatr=x[7], t=x[8:17], ts=x[17:])
  return ssbj_structure.SBJ_structure_opt()

def SBJ_auto_build_and_run():
  csd = os.path.dirname(os.path.abspath(__file__))
  p_file: str = os.path.join(csd, "SBJ.yaml")

  def_dict = {
    "setup_file": p_file,
    "run_mode": "build",
    # "is_resume": 0,
    # "restart_file": None,
    "exec_mode": "parallel",
    "mdo_name": "SBJ",
    "working_dir": csd
  }
  MDAO: MDO = main(def_dict)
  for i in range(len(MDAO.subProblems)):
    temp :MDA = MDAO.subProblems[i].MDA_process
    for j in range(len(temp.analyses)):
      MDAO.subProblems[i].MDA_process.analyses[j].blackbox = globals()[MDAO.subProblems[i].MDA_process.analyses[j].blackbox]
    MDAO.subProblems[i].opt = globals()[MDAO.subProblems[i].opt]
  out = MDAO.run(os.path.join(csd, "SBJ.out"), mode = "parallel")
  print(out)

  

if __name__ == "__main__":
  SBJ_auto_build_and_run()