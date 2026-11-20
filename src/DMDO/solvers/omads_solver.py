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
"""OMADS-based solver strategies (MADS, poll step only, search step only)."""
from __future__ import annotations

import copy
import platform
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import OMADS

from ..core.enums import PSIZE_UPDATE
from ..utils.rng import seed_all_rngs
from ..ports import SolverOutcome, SolverRequest
from .base import BaseSolver

# Default floor of the carried-over GMESH frame size (OMADS frame units, i.e. fractions of each variable's range
# since DMDO passes scaling=1): 1e-4 * min(range, 100). Override with CONFIG.<name>.frame_floor for wide bounds.
PSIZE_FLOOR_REL = 1e-4
PSIZE_FLOOR_SPAN_CAP = 100.0


class OMADSSolver(BaseSolver):
  """Mesh Adaptive Direct Search via OMADS.

  ``request.config`` is the subproblem's ``CONFIG.<name>`` block (``options``, ``search``, ``constraintsHandling``,
  ``mesh_type``, ``initial_frame_size``, ``frame_floor``, ``frame_cap``). ``request.warm_start`` carries
  ``psize`` (frame/poll size of the previous solve) and ``psize_policy`` (:class:`PSIZE_UPDATE`).
  """

  rejects_out_of_bounds = True

  def _run_optimizer(self, data: Dict[str, Any]) -> Dict[str, Any]:
    out, _, _ = OMADS.mads.main(data)
    return out

  # ----------------------------------------------------------------------------------------------------------------
  def solve(self, request: SolverRequest) -> SolverOutcome:
    conf = request.config if isinstance(request.config, dict) else None
    psize = request.warm_start.get("psize", 1.)
    psize_policy = request.warm_start.get("psize_policy", PSIZE_UPDATE.LAST)

    param = self._build_param(request, conf, psize, psize_policy)
    pinit = min(max(request.tol, max(psize) if isinstance(psize, list) else psize), 1)
    options = self._build_options(request, conf, pinit, psize_policy)
    search = self._build_search(conf, request.budget_scale)
    data = {"evaluator": {"blackbox": request.objective}, "param": param, "options": options, "search": search}

    # Keep all stochastic components reproducible under the configured YAML seed.
    seed_all_rngs(int(options["seed"]))
    out = self._run_optimizer(data)

    return SolverOutcome(
      x=out["xmin"],
      fmin=out["fmin"],
      hmin=out["hmin"],
      raw=out,
      warm_start={"psize": self._next_psize(out, psize_policy), "psuccess": out.get("psuccess")},
    )

  # ----------------------------------------------------------------------------------------------------------------
  @staticmethod
  def _next_psize(out: Dict[str, Any], policy) -> Any:
    if policy == PSIZE_UPDATE.DEFAULT:
      return 1.
    if policy == PSIZE_UPDATE.SUCCESS:
      return out["psuccess"]
    if policy == PSIZE_UPDATE.MAX:
      maxpsize = max(out["psize"]) if isinstance(out["psize"], list) else out["psize"]
      psuccess = max(out["psuccess"]) if isinstance(out["psuccess"], list) else out["psuccess"]
      return max(maxpsize, psuccess)
    if policy == PSIZE_UPDATE.LAST:
      return out["psize"]
    return 1.

  @staticmethod
  def _initial_frames(param: Dict[str, Any], conf, psize, psize_policy) -> Tuple[bool, List[float]]:
    """GMESH ignores psize_init: warm-start it with the per-variable frame size of the previous solve, floored
    relative to the variable range so a later solve can still move after the penalties change."""
    carried = list(np.ravel(psize)) if np.ndim(psize) > 0 else []
    first_frame = conf.get("initial_frame_size") if conf is not None else None
    if param["mesh_type"] == "GMESH" and not carried and first_frame is not None:
      # First solve: optional CONFIG.<name>.initial_frame_size (scalar or per-variable, in fractions of the range)
      # instead of OMADS's default, which is far too coarse for very wide bounds.
      n_var = len(param["lb"])
      return True, [float(v) for v in np.broadcast_to(np.asarray(first_frame, dtype=float), (n_var,))]
    if param["mesh_type"] == "GMESH" and psize_policy != PSIZE_UPDATE.DEFAULT and len(carried) == len(param["lb"]):
      span = np.subtract(param["ub"], param["lb"])
      user_floor = conf.get("frame_floor") if conf is not None else None
      floors = ([float(user_floor)] * len(span) if user_floor is not None
                else [PSIZE_FLOOR_REL * min(s, PSIZE_FLOOR_SPAN_CAP) for s in span])
      frames = [float(max(p, f)) for p, f in zip(carried, floors)]
      cap = conf.get("frame_cap") if conf is not None else None
      if cap is not None:
        # Optional cap (fraction of range): a frame grown by successful polls would otherwise start the next solve
        # with steps across most of the range.
        frames = [min(fr, float(cap)) for fr in frames]
      return True, frames
    return False, []

  def _build_param(self, request: SolverRequest, conf, psize, psize_policy) -> Dict[str, Any]:
    param = {"baseline": request.x0,
             "lb": request.lb,
             "ub": request.ub,
             "var_names": request.var_names,
             "var_type": request.var_types,
             "var_sets": request.var_sets,
             "scaling": [1] * len(request.lb),
             "constants": request.constants,
             "constants_name": request.constants_names,
             "name": request.name,
             "mesh_type": conf["mesh_type"]
             if conf is not None and "mesh_type" in conf and conf["mesh_type"] in ["GMESH", "OMESH"]
             else "GMESH",
             "post_dir": request.post_dir,
             "constraints_type": ["PB"] * 100}
    has_frames, frames = self._initial_frames(param, conf, psize, psize_policy)
    if has_frames:
      param["initial_frame_size"] = frames
    self._apply_constraints_handling(param, conf)
    return param

  @staticmethod
  def _apply_constraints_handling(param: Dict[str, Any], conf) -> None:
    handling = conf.get("constraintsHandling") if conf is not None else None
    if handling is None:
      return
    if handling.get("Barriers") is not None:
      param["constraints_type"] = copy.deepcopy(handling["Barriers"])
    for key in ("rho", "h_max", "lambda_multipliers"):
      if handling.get(key) is not None:
        param[key] = handling[key]
    barriers = handling.get("barriers")
    if barriers is not None:
      if barriers == "PB_all":
        param["constraints_type"] = ["PB"] * 100
      elif barriers == "EB_all":
        param["constraints_type"] = ["EB"] * 100
      else:
        param["constraints_type"] = barriers

  @staticmethod
  def _build_options(request: SolverRequest, conf, pinit, psize_policy) -> Dict[str, Any]:
    if conf is not None and "options" in conf and conf["options"] is not None:
      options = copy.deepcopy(conf["options"])
      options["budget"] = request.budget
      options["seed"] = int(request.seed)
      # Carry the poll size over from the previous solve unless the user asked for a reset.
      if psize_policy != PSIZE_UPDATE.DEFAULT:
        options["psize_init"] = pinit
      options.setdefault("check_cache", True)
      options.setdefault("store_cache", True)
      return options
    is_mac = platform.platform().split('-')[0] == 'macOS'
    return {
      "seed": int(request.seed),
      "budget": request.budget,
      "tol": request.tol,
      "psize_init": pinit,
      "display": request.display,
      "opportunistic": False,
      "check_cache": True,
      "store_cache": True,
      "collect_y": False,
      "rich_direction": True,
      "precision": "high" if is_mac else "medium",
      "save_results": False,
      "save_coordinates": False,
      "save_all_best": False,
      "parallel_mode": False,
    }

  @staticmethod
  def _build_search(conf, budget_scale) -> Dict[str, Any]:
    if conf is not None and "search" in conf and conf["search"] is not None:
      # Copy so a budget expansion does not compound into every later solve.
      search = copy.deepcopy(conf["search"])
      search["ns"] = int(search["ns"] * budget_scale) if "ns" in search and search["ns"] is not None else 5 * budget_scale
      return search
    return {
      "type": "LHS",
      "s_method": "ACTIVE",
      "ns": 10 * budget_scale,
      "visualize": False,
      "criterion": None,
    }


class MADSSolver(OMADSSolver):
  """Full MADS (search + poll)."""

  def _run_optimizer(self, data: Dict[str, Any]) -> Dict[str, Any]:
    try:
      out, _, _ = OMADS.mads.main(data)
    except Exception as exc:
      # OMADS cache can occasionally desync its center index; retry once with cache disabled.
      options = data["options"]
      is_index_error = isinstance(exc, IndexError) or exc.__class__.__name__ == "IndexError"
      cache_enabled = options.get("check_cache", True) or options.get("store_cache", True)
      if not (is_index_error and "list index out of range" in str(exc) and cache_enabled):
        raise
      options["check_cache"] = False
      options["store_cache"] = False
      out, _, _ = OMADS.mads.main(data)
    return out


class _TwoTupleOMADSSolver(OMADSSolver):
  _entry: Callable = None

  def _run_optimizer(self, data: Dict[str, Any]) -> Dict[str, Any]:
    out, _ = type(self)._entry(data)
    return out


class PollSolver(_TwoTupleOMADSSolver):
  """OMADS poll step only."""
  refresh_dependent_baseline = True
  _entry = staticmethod(OMADS.poll.main)


class SearchSolver(_TwoTupleOMADSSolver):
  """OMADS search step only."""
  _entry = staticmethod(OMADS.search.main)
