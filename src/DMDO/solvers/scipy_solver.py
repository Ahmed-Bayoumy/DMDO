"""scipy.optimize.minimize solver strategy."""
from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Optional

import numpy as np
from scipy.optimize import BFGS, Bounds, NonlinearConstraint, minimize

from ..ports import SolverOutcome, SolverRequest
from .base import BaseSolver


class ScipySolver(BaseSolver):
  """Gradient-based subproblem solves with :func:`scipy.optimize.minimize`.

  ``request.config`` is the subproblem's ``scipy`` block: ``method``, ``options``, ``tol``, ``is_con`` (pass the
  coupling/local constraints to the optimizer instead of a quadratic penalty) and ``normalize`` (optimize in the
  unit box). Optional: ``scale_objective`` (divide the objective by max(1, |f(x0)|) in each solve), ``jac`` (passed
  to scipy, e.g. ``"3-point"``) and ``cache_constraints`` (see :meth:`SubProblem.get_con`).
  """

  objective_form = "scalar"

  def prepare_settings(self, settings: Optional[Dict[str, Any]], sp_index: int,
                       warn: Callable[[str], None]) -> Dict[str, Any]:
    if settings is None or "options" not in settings or "method" not in settings:
      warn(f'Inappropriate definition of the scipy settings for subproblem # {sp_index}! '
           'scipy default settings shall be used!')
      return {"options": {'disp': False}, "method": 'SLSQP', "is_con": False, "tol": 1E-4}
    return settings

  def select_config(self, conf, settings):
    return settings

  def report_hmin(self, hmin: Any) -> Any:
    return [float(hmin)]

  def solve(self, request: SolverRequest) -> SolverOutcome:
    settings = request.config
    if not isinstance(settings, dict):
      raise IOError('Scipy solver is selected but its dictionary settings is inappropriately defined!')
    opts = copy.deepcopy(settings["options"])
    # Only trust-constr accepts 'verbose'; other methods emit OptimizeWarning for it.
    if str(settings["method"]).lower() != "trust-constr":
      opts.pop("verbose", None)
    # The inner budget (and the AI's budget expansion) acts on the iteration limit.
    if "maxiter" in opts:
      opts["maxiter"] = max(1, int(round(float(opts["maxiter"]) * float(request.budget_scale))))
    lb = np.asarray(request.lb, dtype=float)
    ub = np.asarray(request.ub, dtype=float)
    objective, constraints = request.objective, request.constraints
    if settings.get("normalize", False):
      # Optimise z in [0, 1] with x = lb + z (ub - lb): finite-difference gradients on comparable scales.
      span = np.where(ub > lb, ub - lb, 1.0)
      to_x = lambda z: list(lb + np.clip(z, 0.0, 1.0) * span)  # noqa: E731
      obj_fun = lambda z, *a: objective(to_x(z), *a)  # noqa: E731
      con_fun = lambda z: constraints(to_x(z))  # noqa: E731
      bnds = Bounds(lb=np.zeros_like(lb), ub=np.ones_like(ub))
      x0 = np.clip((np.asarray(request.x0, dtype=float) - lb) / span, 0.0, 1.0)
    else:
      to_x = lambda z: list(z)  # noqa: E731
      obj_fun = objective
      con_fun = constraints
      bnds, x0 = Bounds(lb=lb, ub=ub), request.x0

    # Optional ``scale_objective``: divide the objective by max(1, |f(x0)|) for this solve. The minimiser is unchanged,
    # but SLSQP's absolute ftol and its identity initial Hessian assume f = O(1); with a raw objective of ~1e4 plus a
    # large augmented-Lagrangian penalty it stops at x0 or fails its line search ("Positive directional derivative").
    f_scale = 1.0
    if settings.get("scale_objective", False):
      f0 = float(obj_fun(x0))
      if np.isfinite(f0):
        f_scale = max(1.0, abs(f0))
      unscaled_obj = obj_fun
      obj_fun = lambda z, *a: unscaled_obj(z, *a) / f_scale  # noqa: E731

    kwargs = {"method": settings["method"], "x0": x0, "options": opts, "tol": settings["tol"], "bounds": bnds}
    if settings.get("jac") is not None:
      kwargs["jac"] = settings["jac"]
    if settings["is_con"]:
      if str(settings["method"]).upper() == "SLSQP":
        # SLSQP takes dict constraints (fun >= 0).
        kwargs["constraints"] = [{"type": "ineq", "fun": lambda z: -np.asarray(con_fun(z), dtype=float)}]
      else:
        kwargs["constraints"] = [NonlinearConstraint(fun=con_fun, lb=-np.inf, ub=0.0, keep_feasible=False,
                                                     jac='2-point', hess=BFGS())]
    res = minimize(obj_fun, **kwargs)
    # Optional ``restore_feasibility``: SLSQP (unlike fmincon's SQP) cannot recover from an infeasible start once its
    # QP linearisation becomes inconsistent; it then stops with status 8 ("Positive directional derivative") at an
    # infeasible point. On such an infeasible result, minimise the squared violation within the bounds and restart.
    if settings["is_con"] and settings.get("restore_feasibility", False):
      con_tol = float(settings.get("con_tol", 1e-6))
      viol = lambda z: float(np.sum(np.maximum(np.asarray(con_fun(z), dtype=float), 0.0) ** 2))  # noqa: E731
      if viol(res.x) > con_tol ** 2:
        phase1 = minimize(viol, res.x, method="L-BFGS-B", bounds=bnds, options={"maxiter": 200, "ftol": 1e-16, "gtol": 1e-12})
        if viol(phase1.x) < viol(res.x):
          res2 = minimize(obj_fun, **{**kwargs, "x0": phase1.x})
          v2 = viol(res2.x)
          if v2 <= con_tol ** 2 or v2 < viol(res.x):
            res = res2
          elif viol(phase1.x) < viol(res.x):
            res = phase1
            res.fun = float(obj_fun(phase1.x))
    x_best = np.asarray(to_x(res.x), dtype=float)
    # Same violation measure as OMADS's progressive barrier: h = sum(max(0, g)^2).
    g = np.asarray(constraints(list(x_best)), dtype=float) if settings["is_con"] else np.zeros(1)
    hmin = float(np.sum(np.maximum(g, 0.0) ** 2))
    # Legacy behaviour (kept for numerical reproducibility): the best point is evaluated here once more, on top of
    # the commit evaluation SubProblem.solve performs after every solver.
    objective(copy.deepcopy(x_best))
    return SolverOutcome(x=copy.deepcopy(x_best), fmin=res.fun * f_scale, hmin=hmin, raw=copy.deepcopy(x_best))
