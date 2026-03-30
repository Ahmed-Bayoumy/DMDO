import os

# import yaml
import random
import shutil
from ruamel.yaml import YAML


from DMDO import MDA, MDO, USER, main

from SSBJ_Aircraft import SSBJ_Aircraft
from SSBJ_Aerodynamics import SSBJAerodynamics
from SSBJ_Propulsion import SSBJPropulsion
from SSBJ_Structures import WingDesignAnalyzer
import warnings
warnings.filterwarnings("ignore")

def prepare_setup(n_repeatitions = 100):
  # Initialize YAML loader with preservation
  yaml = YAML()
  yaml.preserve_quotes = True  # Preserve quotes if any
  yaml.width = 1000             # Prevent line wrapping
  yaml.indent(mapping=2, sequence=4, offset=2)
  csd = os.path.dirname(os.path.abspath(__file__))
  bm_inputs_path = os.path.join(csd, "SBJ_bm_files")
  bm_main_input = os.path.join(csd, "SBJ.yaml")
  if os.path.exists(bm_inputs_path):
    shutil.rmtree(bm_inputs_path)
  
  os.mkdir(bm_inputs_path)

  bm_input_files = [os.path.join(bm_inputs_path, f'SBJ_run{x+1}.yaml') for x in range(n_repeatitions)]

  for bm_input_file in bm_input_files:
    shutil.copyfile(src=bm_main_input, dst=bm_input_file)

    # Read the YAML file
    with open(bm_input_file, 'r') as file:
        data = yaml.load(file)
    # Modify the 'bl' value (index 5) for each variable
    for var_key, var_value in data['variables'].items():
      if isinstance(var_value, list) and len(var_value) >= 6:
          bl_value = var_value[5]
          # Apply ±10% perturbation
          perturbation = random.uniform(-0.1, 0.1)
          perturbed_bl = bl_value * (1 + perturbation)
          var_value[5] = perturbed_bl  # Update the value

    # Optionally, write back to the file (uncomment to modify the file)
    with open(bm_input_file, 'w') as file:
        yaml.dump(data, file)
  
  return bm_input_files




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

def SBJ_auto_build_and_run(p_file, wd, run_name):
  # csd = os.path.dirname(os.path.abspath(__file__))
  # p_file: str = os.path.join(csd, "SBJ.yaml")

  def_dict = {
    "setup_file": p_file,
    "run_mode": "build",
    # "is_resume": 0,
    # "restart_file": None,
    "exec_mode": "parallel",
    "mdo_name": run_name,
    "working_dir": wd
  }
  MDAO: MDO = main(def_dict)
  for i in range(len(MDAO.subProblems)):
    temp :MDA = MDAO.subProblems[i].MDA_process
    for j in range(len(temp.analyses)):
      MDAO.subProblems[i].MDA_process.analyses[j].blackbox = globals()[MDAO.subProblems[i].MDA_process.analyses[j].blackbox]
    MDAO.subProblems[i].opt = globals()[MDAO.subProblems[i].opt]
  out = MDAO.run(os.path.join(wd, f"{run_name}.out"), mode = "parallel")
  print(out)

  

if __name__ == "__main__":
  nreps = 100
  files = prepare_setup(n_repeatitions=nreps)
  for yaml_file in files:
     file_name = os.path.splitext(os.path.basename(yaml_file))[0]
     SBJ_auto_build_and_run(p_file=yaml_file, wd = os.path.dirname(yaml_file), run_name=file_name)