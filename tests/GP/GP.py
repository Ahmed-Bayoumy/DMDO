import sys

import os

from DMDO import MDA, MDO, main, USER
from numpy import sqrt

user =USER

def GP_A1(z: dict):
  z1 = sqrt(z["z3"]**2 + z["z4"]**-2 + z["z5"]**2)
  z2 = sqrt(z["z5"]**2 + z["z6"]**2  + z["z7"]**2)
  return [z1, z2]

def GP_opt1(z: dict, y: dict, *args):
  # if isinstance(y, list) and len(y) > 0:
  z1 = y["z1"]
  z2 = y["z2"]
  return [z1**2+z2**2, [z["z3"]**-2 + z["z4"]**2 - z["z5"]**2, z["z5"]**2 + z["z6"]**-2  - z["z7"]**2]]
  # else:
  #   return [z[0]**2 + z[3]**2, [z[0]**-2 + z[1]**2 - z[2]**2, z[2]**2 + z[3]**-2  - z[4]**2]]

def GP_A2(z: dict):
  z3 = sqrt(z["z8"]**2 + z["z9"]**-2 + z["z10"]**-2 + z["z11"]**2)
  return [z3]

def GP_opt2(z: dict, y: dict, *args):
  return [0, [z["z8"]**2 + z["z9"]**2 - z["z11"]**2, z["z8"]**-2 + z["z10"]**2 - z["z11"]**2]]

def GP_A3(z:dict):
  z6 = sqrt(z["z11"]**2 + z["z12"]**2 + z["z13"]**2 + z["z14"]**2)
  return [z6]

def GP_opt3(z: dict, y: dict, *args):
  return [0, [z["z11"]**2 + z["z12"]**-2 - z["z13"]**2, z["z11"]**2 + z["z12"]**2 - z["z14"]**2]]


def GP_auto_build_and_run(yaml_path: str = None, working_dir: str = None):
  csd = os.path.dirname(os.path.abspath(__file__))

  # Auto-detect yaml from script's parent directory if not provided
  if yaml_path is None:
    yaml_path = os.path.join(csd, "GP.yaml")
  if working_dir is None:
    working_dir = csd

  def_dict = {
    "setup_file": yaml_path,
    "run_mode": "build",
    "exec_mode": "serial",
    "mdo_name": "GP",
    "working_dir": working_dir
  }
  MDAO: MDO = main(def_dict)
  for i in range(len(MDAO.subProblems)):
    temp :MDA = MDAO.subProblems[i].MDA_process
    for j in range(len(temp.analyses)):
      MDAO.subProblems[i].MDA_process.analyses[j].blackbox = globals()[MDAO.subProblems[i].MDA_process.analyses[j].blackbox]
    MDAO.subProblems[i].opt = globals()[MDAO.subProblems[i].opt]
  out_file = os.path.join(working_dir, "GP.out")
  out = MDAO.run(out_file, mode = "serial")
  print(out)


if __name__ == "__main__":
  _yaml = sys.argv[1] if len(sys.argv) > 1 else None
  _wdir = sys.argv[2] if len(sys.argv) > 2 else None
  GP_auto_build_and_run(yaml_path=_yaml, working_dir=_wdir)