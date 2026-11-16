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


from .app.main import main, logger
from .workflow.mdo import MDO, MDO_data
from .core.enums import (BARRIER_TYPE, COUPLING_STRENGTH, COUPLING_TYPE, MDO_ARCHITECTURE, MODEL_TYPE, PSIZE_UPDATE,
                         VALIDATOR, VAR_TYPE, w_scheme, double_precision, USER)
from .subproblem.subproblem import SubProblem, partitionedProblemData, process, variableData
from .core.protocols import Process_data
from .coordination.admm import coordinationData, ADMM, ADMM_data
from .core.protocols import coordinator, search
from .analysis.mda import MDA, MDA_data
from .analysis.da import DA, DA_Data
from .app.builder import problemSetup, optimizationData
from .dashboard import main as dmdo_dash
from .ports import Discipline, SubproblemSolver, SolverRequest, SolverOutcome
from .registry import Registry, blackboxes, register_blackbox, solvers
from .solvers import BaseSolver

__all__: list[str] = [
  'USER', 
  'double_precision', 
  'VAR_TYPE', 
  'VALIDATOR', 
  'BARRIER_TYPE', 
  'PSIZE_UPDATE', 
  'w_scheme', 
  'MODEL_TYPE', \
  'COUPLING_STRENGTH', 
  'COUPLING_TYPE', 
  'MDO_ARCHITECTURE',
  'variableData', 
  'Process_data', 
  'coordinationData', \
  'coordinator', 
  'process', 
  'search', 
  'DA_Data', 
  'optimizationData', 
  'DA', 
  'MDA_data', 
  'MDA', 
  'ADMM', \
  'ADMM_data', 
  'partitionedProblemData', 
  'SubProblem', 
  'MDO_data', 
  'MDO', 
  'problemSetup', 
  'main',
  'logger',
  'dmdo_dash',
  'Discipline',
  'SubproblemSolver',
  'SolverRequest',
  'SolverOutcome',
  'BaseSolver',
  'Registry',
  'solvers',
  'blackboxes',
  'register_blackbox',
  ]
