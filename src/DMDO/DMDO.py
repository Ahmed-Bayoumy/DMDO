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
import shutil
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

def is_valid_yaml_file(filepath: str) -> bool:
    """
    Validates both the file extension and content of a YAML file.

    Args:
        filepath (str): Path to the file to validate.

    Returns:
        bool: True if file has valid YAML extension and content, False otherwise.
    """
    # 1. Check if file exists
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return False

    # 2. Check if it's a file (not a directory)
    if not os.path.isfile(filepath):
        print(f"Path is not a file: {filepath}")
        return False

    # 3. Check file extension
    valid_extensions = {'.yaml', '.yml'}
    _, ext = os.path.splitext(filepath.lower())
    if ext not in valid_extensions:
        print(f"Invalid file extension: {ext}. Expected .yaml or .yml.")
        return False

    # 4. Try to parse the YAML content
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            yaml_content = f.read()
            if not yaml_content.strip():
                print("YAML file is empty.")
                return False

            # Use safe_load to avoid arbitrary code execution
            parsed = yaml.safe_load(yaml_content)
            if parsed is None:
                print("YAML file is empty or contains only comments.")
                return False

            # Optional: Add schema validation here if needed
            # Example: validate_schema(parsed, expected_schema)

            return True

    except yaml.YAMLError as e:
        print(f"Invalid YAML content: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error while reading file: {e}")
        return False

def is_valid_pickle_file(filepath: str) -> bool:
    """
    Validates if a file is a valid pickle file before attempting to load it.
    
    Args:
        filepath (str): Path to the file to validate.

    Returns:
        bool: True if the file is a valid pickle file, False otherwise.
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return False

    if not os.path.isfile(filepath):
        print(f"Path is not a file: {filepath}")
        return False

    try:
        with open(filepath, 'rb') as f:
            # Read the first few bytes to check for pickle magic number
            header = f.read(4)
            if len(header) < 4:
                print("File is too short to be a valid pickle file.")
                return False

            # Check for pickle magic number (0x80 0x03 or 0x80 0x04)
            if header[0] != 0x80:
                print("Invalid pickle magic number: missing 0x80.")
                return False

            # Check for version (0x03 or 0x04)
            if header[1] not in (0x03, 0x04):
                print(f"Unsupported pickle version: {header[1]}")
                return False

            # Try to load the file to ensure it's valid
            f.seek(0)  # Reset file pointer
            pickle.load(f)
            return True

    except (pickle.UnpicklingError, EOFError, OSError, ValueError) as e:
        print(f"Invalid or corrupted pickle file: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error while validating pickle file: {e}")
        return False

def validate_keys_raise(data: dict, allowed_keys: list):
    allowed_set = set(allowed_keys)
    invalid_keys = [key for key in data if key not in allowed_set]

    if invalid_keys:
        raise ValueError(f"Invalid keys found: {invalid_keys}. Allowed keys: {allowed_keys}")
    return True


def create_directory_if_not_exists(path: str):
    """
    Creates all directories in the given path if they don't exist.
    
    Args:
        path (str): The full path to create (e.g., 'data/logs/2025/04/05')
    """
    try:
        os.makedirs(path, exist_ok=True)
        # print(f"Directory created: {path}")
    except Exception as e:
        print(f"Failed to create directory {path}: {e}")

# Example usage
create_directory_if_not_exists("data/logs/2025/04/05")

# TODO: MDO setup will be simplified when the N2 chart UI is implemented
def main(*args) -> Dict[str, Any]:  # noqa: C901
    # Default run options and initialization
    exec_mode = "Serial"
    restart_file = os.getcwd()
    run_mode: str = "run"
    is_resume: bool = False
    log: logger = None
    inputs = ["setup_file", "run_mode", "is_resume", "restart_file", "exec_mode", "mdo_name", "working_dir"]
    
    # Initialize variables
    setup_file = None
    working_dir = None
    mdo_name = None
    
    # Validate input type
    if not isinstance(args[0], dict):
        raise IOError(
            "The argument passed to DMDO must be a dictionary. "
            "Expected format: { "
            "'setup_file': '/path/to/setup.yaml', "
            "'run_mode': 'run', "
            "'is_resume': '0', "
            "'restart_file': '/path/to/restart.pkl', "
            "'exec_mode': 'serial', "
            "'mdo_name': 'MDO_test', "
            "'working_dir': '/path/to/working/dir' "
            "}"
        )
    
    # Validate required keys
    try:
        validate_keys_raise(args[0], inputs)
    except ValueError as e:
        raise ValueError(f"Missing or invalid keys in input dictionary: {e}")
    
    # Process each key with proper validation
    for key, arg in args[0].items():
        try:
            if key == "setup_file":
                if not isinstance(arg, str):
                    raise TypeError("setup_file must be a string")
                if not is_valid_yaml_file(arg):
                    raise IOError(f"Invalid YAML file path: {arg}")
                setup_file = arg
                
            elif key == "mdo_name":
                if not isinstance(arg, str):
                    raise TypeError("mdo_name must be a string")
                mdo_name = arg
                
            elif key == "working_dir":
                if not isinstance(arg, str):
                    raise TypeError("working_dir must be a string")
                working_dir = arg
                if not os.path.exists(working_dir):
                    create_directory_if_not_exists(working_dir)
                    
            elif key == "run_mode":
                if not isinstance(arg, str):
                    raise TypeError("run_mode must be a string")
                arg_lower = arg.lower()
                if arg_lower not in ["run", "build"]:
                    raise IOError(f"run_mode must be 'run' or 'build', got: {arg}")
                run_mode = arg_lower
                
            elif key == "is_resume":
                if isinstance(arg, bool):
                    is_resume = arg
                elif isinstance(arg, int):
                    is_resume = bool(arg)
                elif isinstance(arg, str):
                    if arg in ["1", "true", "True"]:
                        is_resume = True
                    elif arg in ["0", "false", "False"]:
                        is_resume = False
                    else:
                        raise IOError(f"is_resume must be '1', '0', 'true', or 'false', got: {arg}")
                else:
                    raise TypeError(f"is_resume must be boolean, integer, or string, got: {type(arg).__name__}")
                    
            elif key == "restart_file":
                if not isinstance(arg, str) and arg is not None:
                    raise TypeError("restart_file must be a string")
                if not is_valid_pickle_file(arg):
                    raise IOError(f"Invalid pickle file path: {arg}")
                restart_file = arg
                
            elif key == "exec_mode":
                if not isinstance(arg, str):
                    raise TypeError("exec_mode must be a string")
                arg_lower = arg.lower()
                if arg_lower not in ["serial", "parallel"]:
                    raise IOError(f"exec_mode must be 'serial' or 'parallel', got: {arg}")
                exec_mode = arg_lower
                
        except (TypeError, IOError, ValueError) as e:
            raise type(e)(f"Error processing key '{key}': {e}")
    
    # Additional validation after processing
    if run_mode == "build" and is_resume:
        raise IOError("Cannot resume a build operation")
        
    post_dir = os.path.join(working_dir, mdo_name+"_post")
    if os.path.exists(post_dir):
      shutil.rmtree(post_dir)
    create_directory_if_not_exists(post_dir)

    if (is_resume):
      if(os.path.exists(restart_file)):
        MDAO: MDO = pickle.load(open(restart_file, "rb"))
        MDAO.run(file=None, resume=True)
      else:
        raise IOError("Could not find the coordination checkpoint file required to resume the coordination process!")
    else:
      log: logger = logger()
      log.initialize(os.path.join(post_dir, "DMDO.log"), handler_name="DMDO")
      inp = setup_file
      """ The DMDO main routine """
      if not isinstance(setup_file, str): 
        raise IOError(f'{setup_file} is not a string! Please use an appropriate DMDO yaml file!')
      
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
      data["working_dir"] = working_dir
      data["post_dir"] = post_dir
      data["mdo_name"] = mdo_name
      if log is None:
        log = logger()
      if log.log is None:
        log.initialize(os.path.join(post_dir, "DMDO.log"), handler_name="DMDO")
      
      log.log_msg(msg="--------------------------------------------------\n",
                  msg_type=MSG_TYPE.INFO.value)
      MDAO= problemSetup(data=data, log = log).autoProbSetup()
      MDAO.log = log

      if run_mode != "run" and run_mode != "build":
        MDAO.prepare_post(os.path.join(post_dir, f"{mdo_name}.out"))
        MDAO.initialized = True
        return MDAO

    if not is_resume and run_mode.lower() == "run":
      MDAO.run(file = os.path.join(post_dir, "coordination_summary.out"), \
              resume= is_resume, mode=exec_mode)
    # elif run_mode.lower() == "build": #TODO: check if this is needed
    #   if ("OPTIONS" in data and "WORK_DIR" in data["OPTIONS"]):
    #     MDAO.prepare_post(os.path.join(data["OPTIONS"]["WORK_DIR"], "coordination_summary.out"))
    #   else:
    #     MDAO.prepare_post(os.path.join(os.getcwd(), "unknown.out"))
    MDAO.initialized = True

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
