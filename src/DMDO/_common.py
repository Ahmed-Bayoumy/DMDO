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

import os
import shutil

from ._globals import MSG_TYPE, VERSION_NUMBER
import logging 
import time

from dataclasses import dataclass

@dataclass
class logger:
    log: logging.Logger = None
    handlers: dict[str, logging.Handler] = None  # Track handlers by name
    active_handler_name: str = None
    _logger_name: str = None  # Store logger name instead of instance
    _handler_configs: dict[str, dict] = None  # Store handler configs instead of handlers

    def __post_init__(self):
        self.handlers = {}
        self.active_handler_name = None
        self._handler_configs = {}
        self._logger_name = None

    def __getstate__(self):
        # Remove non-picklable attributes
        state = self.__dict__.copy()
        # Remove logger instance and handlers
        if 'log' in state:
            del state['log']
        if 'handlers' in state:
            del state['handlers']
        # Store only configuration data
        state['_handler_configs'] = {}
        for name, handler in self.handlers.items():
            state['_handler_configs'][name] = {
                'file': handler.baseFilename,
                'mode': handler.mode,
                'formatter': handler.formatter._fmt if handler.formatter else None,
                'level': handler.level
            }
        # Store logger name
        if self.log:
            state['_logger_name'] = self.log.name
        return state

    def __setstate__(self, state):
        # Restore state
        self.__dict__.update(state)
        # Reconstruct logger and handlers
        self.log = None
        self.handlers = {}
        self.active_handler_name = None
        
        # Reconstruct handlers from configs
        for name, config in self._handler_configs.items():
            handler = logging.FileHandler(config['file'], mode=config['mode'])
            formatter = logging.Formatter(config['formatter'] or '%(asctime)s %(message)s')
            handler.setFormatter(formatter)
            handler.setLevel(config['level'])
            self.handlers[name] = handler
            
            # Add to logger if this is the active one
            if self.active_handler_name == name:
                self.log = logging.getLogger(config['file'])
                self.log.addHandler(handler)
                self.log.setLevel(config['level'])
        
        # Restore logger name if needed
        if self._logger_name:
            self.log = logging.getLogger(self._logger_name)
        
        # Clean up
        self._handler_configs = {}
        self._logger_name = None

    def initialize(self, file: str, handler_name: str = "default"):
      """
      Initialize a logger with a file handler.
      If handler_name already exists, it will be replaced.
      """
      # Remove any existing handler with the same name
      if handler_name in self.handlers:
          self.log.removeHandler(self.handlers[handler_name])
          self.handlers[handler_name].close()

      # Create a new file handler
      handler = logging.FileHandler(file, mode='a')
      formatter = logging.Formatter('%(asctime)s %(message)s')
      handler.setFormatter(formatter)
      handler.setLevel(logging.INFO)

      # Add handler to the logger
      self.handlers[handler_name] = handler
      self.log = logging.getLogger(f"DMDO_{handler_name}")
      self.log.setLevel(logging.INFO)
      self.log.addHandler(handler)

      # Set this as active handler
      self.active_handler_name = handler_name

      # Optional: remove root handlers to prevent duplicate output
      for h in logging.root.handlers[:]:
          logging.root.removeHandler(h)

      # Log startup message
      cur_time = time.strftime("%H:%M:%S", time.localtime())
      self.log_msg(msg="######################################################", msg_type=MSG_TYPE.INFO.value)
      self.log_msg(msg=f"####################### DMDO {VERSION_NUMBER} #########################", \
                  msg_type=MSG_TYPE.INFO.value)
      self.log_msg(msg=f"###################### {cur_time} ######################", msg_type=MSG_TYPE.INFO.value)

      # Update handler configuration for pickling
      self._handler_configs[handler_name] = {
          'file': file,
          'mode': 'a',
          'formatter': '%(asctime)s %(message)s',
          'level': logging.INFO
      }

    def log_msg(self, msg: str, msg_type: MSG_TYPE):
      """
      Log a message using the currently active handler.
      """
      if self.log is None:
          raise RuntimeError("Logger not initialized. Call initialize() first.")
      
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

    def remove_handler(self, handler_name: str):
        """
        Remove a handler by name.
        """
        if handler_name not in self.handlers:
            print(f"Handler '{handler_name}' not found.")
            return

        handler = self.handlers.pop(handler_name)
        if self.log:
            self.log.removeHandler(handler)
        
        # Remove from handler configs
        if handler_name in self._handler_configs:
            del self._handler_configs[handler_name]
        
        handler.close()
        print(f"Removed handler: {handler_name}")

    def switch_handler(self, handler_name: str):
        """
        Switch to a different handler (file) by name.
        Raises KeyError if handler doesn't exist.
        """
        if handler_name not in self.handlers:
            raise KeyError(f"Handler '{handler_name}' not found. Available: {list(self.handlers.keys())}")

        # Remove current handler from logger if it exists
        if self.active_handler_name and self.active_handler_name in self.handlers:
            self.log.removeHandler(self.handlers[self.active_handler_name])

        # Set new handler
        self.log.addHandler(self.handlers[handler_name])
        self.active_handler_name = handler_name
        # print(f"Switched logger to handler: {handler_name}")

    def add_handler(self, file: str, handler_name: str):
        """
        Add a new handler (file) without switching.
        """
        if handler_name in self.handlers:
            return

        handler = logging.FileHandler(file, mode='a')
        formatter = logging.Formatter('%(asctime)s %(message)s')
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)

        self.handlers[handler_name] = handler
        if self.log:
            self.log.addHandler(handler)
        
        # Store handler configuration
        self._handler_configs[handler_name] = {
            'file': file,
            'mode': 'a',
            'formatter': '%(asctime)s %(message)s',
            'level': logging.INFO
        }
        
        print(f"Added new handler: {handler_name} -> {file}")

    def get_active_handler(self) -> str:
        return self.active_handler_name

    def relocate_logger(self, source_file: str = None, Dest_file: str = None):
        if Dest_file is not None and source_file is not None and os.path.exists(source_file):
            shutil.copy(source_file, Dest_file)
            if os.path.exists("DSMToDMDO.yaml"):
                shutil.copy("DSMToDMDO.yaml", Dest_file)
            
            # Remove all handlers from root logger
            for handler in logging.root.handlers[:]:
                handler.close()
                logging.root.removeHandler(handler)
            
            if os.path.split(Dest_file)[1] == "DMDO.log":
                lname = "DMDO"
            else:
                lname = "SP"
            
            # Remove existing handler if it exists
            if "DMDO" in self.handlers:
                self.log.removeHandler(self.handlers["DMDO"])
                self.handlers["DMDO"].close()
                del self.handlers["DMDO"]
            
            # Create and configure logger
            log_file = os.path.join(Dest_file, "DMDO.log")
            handler = logging.FileHandler(log_file, mode='a')
            formatter = logging.Formatter('%(asctime)s %(message)s')
            handler.setFormatter(formatter)
            handler.setLevel(logging.DEBUG)
            
            self.handlers[lname] = handler
            self.log = logging.getLogger(f"DMDO_{lname}")
            self.log.setLevel(logging.DEBUG)
            self.log.addHandler(handler)
            
            # Update handler configuration
            self._handler_configs[lname] = {
                'file': log_file,
                'mode': 'a',
                'formatter': '%(asctime)s %(message)s',
                'level': logging.DEBUG
            }
            
            # Set active handler
            self.active_handler_name = lname
            
            # Try to remove source file
            try:
                os.remove(source_file)
            except PermissionError:
                # If we can't remove it now, we'll try to remove it later
                # This is a fallback to ensure we don't leave the file behind
                pass

    def close_all_handlers(self):
        """Close all handlers and clean up."""
        for handler in self.handlers.values():
            handler.close()
        self.handlers.clear()
        self.log = None
        self.active_handler_name = None
        self._handler_configs.clear()  # Clear handler configs