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
from dataclasses import dataclass
import os
import pickle
import sys
from typing import Dict, Any
import numpy as np
import yaml

from .MDO import MDO
from ._common import MSG_TYPE, logger
from .preprocess import problemSetup

@dataclass
class ModelInadequacyData:
  type: int
  relative_inadequacies: np.ndarray
  absolute_inadequacies: np.ndarray
  approx_rel_inadeq: np.ndarray
  approx_abs_inadeq: np.ndarray
  reference_model_index: int
  errorSurrogateType: int




# TODO: MDO setup will be simplified when the N2 chart UI is implemented
def main(*args) -> Dict[str, Any]:  # noqa: C901
  # Default run options and initialization
  exec_mode = "Serial"
  isResumeFilePath = os.getcwd()
  runMode: str = "run"
  isResume: bool = False
  log: logger = None
  if len(args) == 2:
    runMode = args[1]
  elif len(args) == 3:
    runMode = args[1]
    isResume = args[2]
    isResumeFilePath = args[3]
  elif len(args) > 3:
    runMode = args[1]
    isResume = args[2]
    isResumeFilePath = args[3]
    exec_mode = args[4]
  else:
    isResume = False
  if (isResume):
    if(os.path.exists(isResumeFilePath + '/coordination_summary_post/coord_chk.pkl')):
      MDAO: MDO = pickle.load(open(isResumeFilePath + '/coordination_summary_post/coord_chk.pkl', "rb"))
      MDAO.run(file=None, resume=True)
    else:
      raise IOError("Could not find the coordination checkpoint file required to resume the coordination process!")
  else:
    log: logger = logger()
    log.initialize(os.path.join(os.getcwd(), "DMDO.log"))
    inp = args[0]
    """ The DMDO main routine """
    if not isinstance(args[0], str): 
      raise IOError(f'{args[0]} is not a string input! Please use an appropriate DMDO yaml file!')
    
    if isinstance(inp, str):
      log.log_msg(msg=f"Reading the input dictionary file {inp}.", msg_type=MSG_TYPE.INFO.value)
      file = copy.deepcopy(inp)
      fext = file.split('.')[1]
      if not (fext == "yaml" or fext == "yml" or fext == "json"):
        msg = f'Cannot use files with {fext} extension. Please use an appropriate yaml file with yml or yaml extension!'
        log.log_msg(msg, MSG_TYPE.ERROR.value)
        raise IOError(msg)
    
    if not os.path.exists(file):
        msg = f'Could not find {file}! Please make sure that the file exists!'
        log.log_msg(msg, MSG_TYPE.ERROR.value)
        raise IOError(msg)
        
    with open(file, "r") as stream:
      try:
          data: Dict = yaml.safe_load(stream)
      except yaml.YAMLError as exc:
          msg = exc
          log.log_msg(msg, MSG_TYPE.ERROR.value)
          raise IOError(exc)

    MDAO: MDO  
    if log is None:
      log = logger()
    if log.log is None:
      log.initialize(os.path.join(os.getcwd(), "DMDO.log"))
    
    log.log_msg(msg="--------------------------------------------------\n",
                msg_type=MSG_TYPE.INFO.value)
    MDAO= problemSetup(data=data, log = log).autoProbSetup()
    MDAO.log = log

    if args[1].lower() != "run":
      MDAO.prepare_post(os.path.join(os.getcwd(), "unknown.out"))
      return MDAO

  if not isResume and runMode.lower() == "run":
    MDAO.run(file = os.path.join(data["OPTIONS"]["CSP1"]["WORK_DIR"], "coordination_summary.out"), \
             resume= isResume, mode=exec_mode)
  elif runMode.lower() == "build":
    if "CSP1" in data["OPTIONS"] and "WORK_DIR" in data["OPTIONS"]["CSP1"]:
      MDAO.prepare_post(os.path.join(data["OPTIONS"]["CSP1"]["WORK_DIR"], "coordination_summary.out"))
    else:
      MDAO.prepare_post(os.path.join(os.getcwd(), "unknown.out"))

  return MDAO

# def A1(x):
#   LAMBDA = 0.0
#   for i in range(len(x)):
#     if x[i] == 0.:
#       x[i] = 1e-12
#   return math.log(x[0]+LAMBDA) + math.log(x[1]+LAMBDA) + math.log(x[2]+LAMBDA)

# def A2(x):
#   LAMBDA = 0.0
#   for i in range(len(x)):
#     if x[i] == 0.:
#       x[i] = 1e-12
#   return np.divide(1., (x[0]+LAMBDA)) + np.divide(1., (x[1]+LAMBDA)) + np.divide(1., (x[2]+LAMBDA))

# def opt1(x, y):
#   return [sum(x)+y[0], [0.]]

# def opt2(x, y):
#   return [0., [x[1]+y[0]-10.]]

if __name__ == "__main__":
  #COMPLETED: Feature: Add more realistic analytical test problems
  #TODO: Feature: Add realistic multi-physics MDO problems that require using open-source physics-based simulation tools
  #COMPLETED: Feature: Move the MDO test functions and BM problems to 
  #COMPLETED: the test folder and prepare the DMDO package to be published on PyPi
  #TODO: Feature: Develop a simple UI widget that facilitates simple MDO setup using the compact table or N2-chart
  #TODO: Feature: Import RAF and SML libraries once they are published on PYPI.com
  #COMPLETED: Bug: Add user and technical documentation 
  #FIXME: Bug: Enable the output report generation that summarizes the MDO history and final results
  p_file: str = ""

  """ Check if an input argument is provided"""
  if len(sys.argv) > 1:
    p_file = os.path.abspath(sys.argv[1])
    main(p_file, sys.argv[2], sys.argv[3:])

  if (p_file != "" and os.path.exists(p_file)):
    main(p_file, "build")

  if p_file == "":
    raise IOError("Undefined input args."
            " Please specify an appropriate DMDO input yamle file")
