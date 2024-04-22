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
import logging 
import time
import shutil
import os

@dataclass
class logger:
  log: None = None

  def initialize(self, file: str):
    # Create and configure logger 
    logging.basicConfig(filename=file, 
              format='%(message)s', 
              filemode='w') 

    #Let us Create an object 
    self.log = logging.getLogger() 

    #Now we are going to Set the threshold of logger to DEBUG 
    self.log.setLevel(logging.INFO) 
    cur_time = time.strftime("%H:%M:%S", time.localtime())
    self.log_msg(msg=f"###################################################### \n", msg_type=MSG_TYPE.INFO.value)
    self.log_msg(msg=f"####################### DMDO ######################### \n", msg_type=MSG_TYPE.INFO.value)
    self.log_msg(msg=f"###################### {cur_time} ###################### \n", msg_type=MSG_TYPE.INFO.value)

    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create and configure logger 
    logging.basicConfig(filename=file, 
              format='%(asctime)s %(message)s', 
              filemode='a') 

    # Let us Create an object 
    self.log = None
    self.log = logging.getLogger() 

    # Now we are going to Set the threshold of logger to DEBUG 
    self.log.setLevel(logging.INFO) 
  
  def log_msg(self, msg: str, msg_type: MSG_TYPE):
    if msg_type == MSG_TYPE.DEBUG.value:
      self.log.debug(msg) 
    elif msg_type == MSG_TYPE.INFO.value:
      self.log.info(msg) 
    elif msg_type == MSG_TYPE.WARNING.value:
      self.log.warning(msg) 
    elif msg_type == MSG_TYPE.ERROR.value:
      self.log.error(msg) 
    elif msg_type == MSG_TYPE.CRITICAL.value:
      self.log.critical(msg) 
  
  def relocate_logger(self, source_file: str = None, Dest_file: str = None):
    if Dest_file is not None and source_file is not None and os.path.exists(source_file):
      shutil.copy(source_file, Dest_file)
      if os.path.exists("DSMToDMDO.yaml"):
        shutil.copy("DSMToDMDO.yaml", Dest_file)
      # Remove all handlers associated with the root logger object.
      for handler in logging.root.handlers[:]:
          logging.root.removeHandler(handler)
      handler.close()
      # Create and configure logger 
      logging.basicConfig(filename=os.path.join(Dest_file, "DMDO.log"), 
                format='%(asctime)s %(message)s', 
                filemode='a')
      #Let us Create an object 
      self.log = logging.getLogger() 

      #Now we are going to Set the threshold of logger to DEBUG 
      self.log.setLevel(logging.DEBUG) 
      os.remove(source_file)
