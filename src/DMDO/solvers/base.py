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
"""Common base for subproblem solver strategies."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from ..ports import SolverOutcome, SolverRequest


class BaseSolver:
  """Default behaviour shared by solver strategies (see :class:`DMDO.ports.SubproblemSolver`).

  Strategies are stateless: per-subproblem state travels in ``SolverRequest.warm_start`` / ``SolverOutcome.warm_start``,
  so a strategy instance can be shared, re-created on every solve, or run in another process.
  """

  #: Registered name, filled in by the registry decorator in :mod:`DMDO.solvers`.
  name: str = ""
  objective_form: str = "vector"
  rejects_out_of_bounds: bool = False
  #: Whether the master subproblem should re-seed its dependent baselines from the master vector before solving.
  refresh_dependent_baseline: bool = False

  def prepare_settings(self, settings: Optional[Dict[str, Any]], sp_index: int,
                       warn: Callable[[str], None]) -> Optional[Dict[str, Any]]:
    """Validate / complete the solver-specific settings given in the subproblem definition."""
    return None

  def select_config(self, conf: Optional[Dict[str, Any]], settings: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pick ``SolverRequest.config`` from the subproblem's ``CONFIG`` block and its prepared solver settings."""
    return conf

  def solve(self, request: SolverRequest) -> SolverOutcome:  # pragma: no cover - abstract
    raise NotImplementedError

  def report_hmin(self, hmin: Any) -> Any:
    return hmin
