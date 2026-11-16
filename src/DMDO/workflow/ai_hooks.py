"""AI-regularizer hooks of the coordination loop: detector feed and design rollback.

These run as :class:`DMDO.ports.RunObserver` hooks that are attached by :meth:`DMDO.MDO.run` when the coordinator has
``is_AI_regularizer`` set. Unlike the recorders in :mod:`.observers` they update the run: the detector outputs they
publish on the coordinator are read by its adaptation policy (:mod:`DMDO.coordination.policy`), and the rollback
restores an earlier design. Their state lives on the run and its coordinator, so checkpoints resume them unchanged.
"""
import copy
import os
from typing import Optional

import numpy as np
import pandas as pd

from ..ai.detector import SolveQualityExtractor, latest_detector_metrics
from ..ports import RunObserver
from ..utils.log import MSG_TYPE


def scalarize_hmin(value) -> float:
  """Largest finite constraint violation of a solve (0 when feasible, inf when only non-finite values are positive)."""
  array = np.atleast_1d(np.asarray(value, dtype=float))
  finite = array[np.isfinite(array)]
  if finite.size > 0:
    return float(max(np.max(finite), 0.0))
  return float(np.inf) if np.any(array > 0.0) else 0.0


def scalar_objective(fmin) -> float:
  return float(fmin[0] if isinstance(fmin, (list, tuple, np.ndarray)) else fmin)


def find_detector(run):
  """The run's spike detector; adopted from a subproblem's coordinator copy if the coordinator has none yet."""
  detector = getattr(run.Coordinator, "ai_regularizer", None)
  if detector is None:
    for sp in run.subProblems:
      detector = getattr(sp.coord, "ai_regularizer", None) or getattr(sp, "_AI_regularizer", None)
      if detector is not None:
        break
    if detector is not None:
      run.Coordinator.ai_regularizer = detector
  return detector


class AIRegularizerHooks(RunObserver):
  """Resets the per-iteration detector outputs and feeds the detector once per outer iteration (paper Algorithm 1)."""

  def on_run_start(self, run) -> None:
    detector = find_detector(run)
    if detector is not None and detector.tol_floor <= 0.0:
      detector.tol_floor = float(run.inc_stop)
    # Tolerance reference for the threshold trigger (SpikeAdaptationPolicy.threshold_spike).
    run.Coordinator._tol_ref = float(run.inc_stop)

  def on_iteration_start(self, run, iteration: int) -> None:
    self.reset_outputs(run.Coordinator)

  def on_sweep_end(self, run, iteration: int) -> None:
    self.feed_detector(run, iteration)

  # ------------------------------------------------------------------------------------------------ steps
  @staticmethod
  def reset_outputs(coord) -> None:
    # A premature-freeze event (causal response) also triggers the next iteration's budget reallocation.
    last_detected = bool(getattr(coord, "spike_detected", False) or getattr(coord, "freeze_detected", False))
    coord.last_spike_detected = last_detected
    coord.freeze_detected = False
    # A stale location without a detection would trigger budget expansion every iteration.
    coord.last_spike_location = int(getattr(coord, "spike_location", 0)) if last_detected else 0
    coord.qmax_predicted = 0.
    coord.spike_location = 0
    coord.alarm_detected = False
    coord.spike_detected = False
    coord.ai_event_strength = -1.
    coord.fmin_predicted = 0.
    coord.spike_prob = 0.
    coord.forecast_prob = 0.
    coord.non_improvement_prob = 0.
    coord.objective_non_improvement_prob = 0.
    coord.constraint_non_improvement_prob = 0.
    coord.detector_uncertainty = 1.
    coord.constraint_pressure = 0.

  @staticmethod
  def attribute_spike_subproblem(run) -> int:
    """Subproblem blamed for the largest inconsistency.

    Of the two ends of the worst link, the one with the more erratic inner solve.
    """
    coord = run.Coordinator
    master = int(coord.index_of_master_SP)
    try:
      per_link = [
        float(np.max(np.abs(q))) if isinstance(q, (list, np.ndarray)) else abs(float(q))
        for q in coord.batch_q(coord.q)
      ]
      worst = int(np.argmax(per_link))
      ends = [int(coord.master_vars[coord._linker[row, worst] - 1].sp_index) for row in (0, 1)]
    except Exception:
      return master
    by_index = {sp.index: sp for sp in run.subProblems}
    best, best_range = ends[0], -1.0
    for idx in dict.fromkeys(ends):
      sp = by_index.get(idx)
      fingerprint = SolveQualityExtractor.extract(getattr(sp, "Il_file", None)) if sp is not None else None
      inc_range = float(fingerprint[3]) if fingerprint is not None else 0.0
      if inc_range > best_range:
        best, best_range = idx, inc_range
    return int(best)

  @staticmethod
  def late_improving(sp_file: Optional[str], tail: float = 0.3, rtol: float = 1e-6) -> bool:
    """True when the solve's best merit (objective + penalty) still improved in its last `tail` fraction of evaluations."""
    if not sp_file or not os.path.isfile(sp_file):
      return False
    try:
      df = pd.read_csv(sp_file, skipinitialspace=True)
      df.columns = [str(c).strip() for c in df.columns]
      merit = (pd.to_numeric(df["Objective"], errors="coerce") + pd.to_numeric(df["Penalty"], errors="coerce")).values
    except Exception:
      return False
    merit = merit[np.isfinite(merit)]
    if merit.size < 4:
      return False
    cut = int(np.floor((1.0 - tail) * merit.size))
    best_before, best_after = float(np.min(merit[:cut])), float(np.min(merit[cut:]))
    return best_after < best_before - rtol * max(1.0, abs(best_before))

  def feed_detector(self, run, iteration: int) -> None:
    """Feed the detector with this outer iteration and publish its outputs on the coordinator."""
    coord = run.Coordinator
    detector = find_detector(run)
    if detector is None or not getattr(coord, "detector_per_outer_iteration", True):
      return

    q = np.abs(np.asarray(coord.q, dtype=float))
    inc = float(np.max(q)) if q.size > 0 else 0.0
    obj = scalar_objective(run.fmin)
    hmins = [scalarize_hmin(run.hmin_dict.get(sp.index, [0.])) for sp in run.subProblems]

    owner = self.attribute_spike_subproblem(run)
    owner_sp = next((sp for sp in run.subProblems if sp.index == owner), None)
    detector.update_sp_features(getattr(owner_sp, "Il_file", None))
    alarm = detector.update(obj=obj, inc=inc, hmins=hmins)

    metrics = latest_detector_metrics(detector)
    coord.qmax_predicted = metrics["forecast_prob"]
    coord.fmin_predicted = metrics["objective_non_improvement_prob"]
    coord.spike_prob = metrics["spike_prob"]
    coord.forecast_prob = metrics["forecast_prob"]
    coord.non_improvement_prob = metrics["non_improvement_prob"]
    coord.objective_non_improvement_prob = metrics["objective_non_improvement_prob"]
    coord.constraint_non_improvement_prob = metrics["constraint_non_improvement_prob"]
    coord.detector_uncertainty = metrics["detector_uncertainty"]
    coord.constraint_pressure = metrics["constraint_pressure"]
    coord.ai_event_strength = float(max(metrics.values()))
    coord.spike_location = owner
    coord.alarm_detected = bool(alarm)
    if str(getattr(coord, "spike_response", "accelerate")) == "causal":
      coord._sp_late_improving = {sp.index: self.late_improving(getattr(sp, "Il_file", None)) for sp in run.subProblems}


class RollbackGuard(RunObserver):
  """On a detected spike with a clearly worse objective, restart from an earlier, better design (``rollback`` option).

  Candidates are earlier feasible iterations whose q_max is within rollback_q_factor of the current one (or below
  100x the tolerance), so the restart point is nearly as consistent as the run has become. The penalty weights are
  scaled by rollback_w_scale (floor w_init) and the multipliers v are restored with the design. At most rollback_max
  rollbacks per run, at least rollback_gap iterations apart. The design history is kept on the run (``_rb_history``).
  """

  def on_multipliers_updated(self, run, iteration: int) -> None:
    self.step(run, iteration)

  def step(self, run, iter: int) -> None:  # noqa: C901
    coord = run.Coordinator
    if not (coord.is_AI_regularizer and getattr(coord, "rollback", False)):
      return
    q_now = float(np.max(np.abs(coord.q))) if np.size(coord.q) else 0.0
    obj_now = scalar_objective(run.fmin)
    hmax = max([scalarize_hmin(v) for v in run.hmin_dict.values()] or [0.0])
    history = getattr(run, "_rb_history", [])
    history.append({
      "iter": int(iter), "obj": obj_now, "q": q_now, "hmax": hmax,
      "sp_vars": [[copy.deepcopy(v.value) for v in sp.vars] for sp in run.subProblems],
      "master_vars": copy.deepcopy(coord.master_vars),
      "v": copy.deepcopy(coord.v),
    })
    run._rb_history = history

    n_done = int(getattr(run, "_rb_count", 0))
    last = int(getattr(run, "_rb_last", -10**9))
    if str(getattr(coord, "rollback_trigger", "event")) == "forecast":
      detected = bool(getattr(coord, "_forecast_accepted", False))
    else:
      detected = getattr(coord, "_event_kind", "") in ("spike", "hold")
    # Optional safeguard: a design that has become infeasible is rolled back even without a detector alarm.
    lost_feasibility = bool(getattr(coord, "rollback_on_infeasible", False)) and hmax > 1e-6
    if not (detected or lost_feasibility) or n_done >= int(coord.rollback_max) or iter - last < int(coord.rollback_gap):
      return
    q_cap = max(float(coord.rollback_q_factor) * q_now, 100.0 * float(run.inc_stop))
    candidates = [h for h in history[:-1] if np.isfinite(h["obj"]) and h["hmax"] <= 1e-6 and h["q"] <= q_cap]
    if not candidates:
      return
    best = min(candidates, key=lambda h: h["obj"])
    if not lost_feasibility and not obj_now > best["obj"] + float(coord.rollback_obj_tol) * max(1.0, abs(best["obj"])):
      return
    for sp, values in zip(run.subProblems, best["sp_vars"]):
      for var, value in zip(sp.vars, values):
        var.value = copy.deepcopy(value)
        var.baseline = copy.deepcopy(value)
    coord.master_vars = copy.deepcopy(best["master_vars"])
    # The multipliers belong to the design they were accumulated on; keeping the current ones would push the
    # restored design hard in the direction of the abandoned one.
    coord.v = copy.deepcopy(best["v"])
    coord.w = np.maximum(np.multiply(coord.w, float(coord.rollback_w_scale)), float(getattr(coord, "w_init", 1.0)))
    run._rb_count, run._rb_last = n_done + 1, int(iter)
    why = "feasibility lost" if lost_feasibility and not detected else "detected spike"
    msg = (f"Rollback at iteration {iter} ({why}): objective {obj_now:.6g}, hmax {hmax:.3g} -> design of iteration "
           f"{best['iter']} ({best['obj']:.6g})")
    run.log.log_msg(msg=msg, msg_type=MSG_TYPE.INFO.value)
    print(msg)
