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
"""Coordination adaptation policies: how a coordinator's penalty update reacts to the run's behaviour.

:class:`~DMDO.coordination.admm.ADMM` owns the multiplier and penalty-weight arithmetic. *When* and *how strongly* that
update is adapted (the AI regularizer's spike response) is a separate strategy, chosen per coordinator:

- ``"none"`` (:class:`NeutralPolicy`): plain ADMM, every scale is 1;
- ``"spike"`` (:class:`SpikeAdaptationPolicy`): the AI regularizer (detector forecast + non-improvement gate).

The default is ``"spike"`` when ``is_AI_regularizer`` is set, otherwise ``"none"``; the coordinator attribute
``adaptation_policy`` (YAML key of the same name) overrides it. A policy holds no state of its own: it reads and writes
the coordinator it is given, so checkpoints and existing coordinator attributes are unaffected by the choice. New
policies subclass :class:`CoordinationPolicy` and register under a name (``policies.register``) or through the
``dmdo.policies`` entry-point group.
"""
import math
from typing import Dict, Tuple, Type

import numpy as np

from ..registry import Registry

#: Coordination adaptation policies (classes deriving from :class:`CoordinationPolicy`).
policies: Registry[Type["CoordinationPolicy"]] = Registry("coordination policy", "dmdo.policies")

NO_EVENT: Tuple[bool, int, int] = (False, 0, -1)


class CoordinationPolicy:
  """Base policy: decides the scales applied to one penalty update and records what was applied.

  Subclasses override :meth:`expected_spike_event` and :meth:`compute_scales`; :meth:`multiplier_scales` commits the
  result on the coordinator (budget scale for the next outer iteration, cool-down and adaptation counters, and the
  applied scales reported in the coordination history).
  """

  name = "base"

  def expected_spike_event(self, coord, iteration: int) -> Tuple[bool, int, int]:
    """Return ``(is_expected_spike, spike_subproblem, spike_future_iteration)`` for this outer iteration."""
    return NO_EVENT

  def compute_scales(self, coord, iteration: int) -> Tuple[float, float, float, bool]:
    """Return ``(beta_scale, gamma_scale, budget_scale, spike)`` without committing anything."""
    return 1.0, 1.0, 1.0, False

  def penalty_caused_spike(self, coord) -> bool:
    return False

  def multiplier_scales(self, coord, iteration: int) -> Tuple[float, float, bool]:
    """Return ``(beta_scale, gamma_scale, spike)`` and commit the budget scale and cool-down state."""
    beta_scale, gamma_scale, budget_scale, spike = self.compute_scales(coord, iteration)
    if spike:
      coord.spike_detected = True
    if getattr(coord, "_event_kind", "") == "freeze" and (beta_scale != 1.0 or gamma_scale != 1.0 or budget_scale != 1.0):
      coord.freeze_detected = True
    # The subproblem reads this on the next outer iteration (one-step-ahead adaptation).
    coord._effective_budget_scale = budget_scale
    adapted = beta_scale != 1.0 or gamma_scale != 1.0 or budget_scale != 1.0
    if adapted:
      coord._last_adapt_iter = int(iteration)
      coord._n_adapt = int(getattr(coord, "_n_adapt", 0)) + 1
    kind = getattr(coord, "_event_kind", "")
    if adapted and kind == "spike" and self.penalty_caused_spike(coord):
      kind = "hold"
    coord._applied_scales = (beta_scale, gamma_scale, budget_scale, kind if adapted else "")
    return beta_scale, gamma_scale, spike


@policies.register("none", aliases=("neutral", "admm"))
class NeutralPolicy(CoordinationPolicy):
  """Plain ADMM: the penalty update is never adapted."""

  name = "none"


@policies.register("spike", aliases=("ai_regularizer", "ai", "spike_adaptation"))
class SpikeAdaptationPolicy(CoordinationPolicy):
  """AI regularizer: adapt penalty growth, stall test and inner budget around detected inconsistency spikes.

  Adaptation is applied only when the detector's decisor accepts the forecast AND the non-improvement gate holds (or
  the configured alternative trigger/gate, see the ``adapt_trigger`` and ``ni_gate`` coordinator options).
  """

  name = "spike"

  def expected_spike_event(self, coord, iteration: int) -> Tuple[bool, int, int]:
    """A spike is expected when forecast confidence is accepted by the decisor and associated with non-improvement."""
    if not getattr(coord, "is_AI_regularizer", False):
      return NO_EVENT

    detector = getattr(coord, "ai_regularizer", None)
    if detector is None:
      return NO_EVENT

    level_name = str(getattr(coord, "AI_spike_policy", getattr(coord, "spike_confidence_level", "L2_Normal")))
    decision = detector.decide(level_name=level_name)
    # Kept for the rollback trigger "forecast", which does not use the non-improvement gate.
    coord._forecast_accepted = bool(decision.accept_forecast_prob)

    spike_threshold = float(getattr(coord, "spike_threshold", 0.5))
    spike_location = int(getattr(coord, "spike_location", 0))
    non_improvement_prob = float(getattr(coord, "non_improvement_prob", 0.0))
    objective_non_improvement_prob = float(getattr(coord, "objective_non_improvement_prob", 0.0))
    constraint_non_improvement_prob = float(getattr(coord, "constraint_non_improvement_prob", 0.0))

    non_improvement_gate = max(
      non_improvement_prob,
      objective_non_improvement_prob,
      constraint_non_improvement_prob,
    ) >= spike_threshold
    if str(getattr(coord, "ni_gate", "association")) == "direct":
      # The association probabilities hover near 0.5 on some problems (GP), so read non-improvement directly.
      non_improvement_gate = self.direct_non_improvement(coord)

    if str(getattr(coord, "ni_gate", "association")) == "objective_drop":
      non_improvement_gate = self.harm_gate(coord)
    elif str(getattr(coord, "ni_gate", "association")) == "none":
      non_improvement_gate = True  # act on every triggered event (no usefulness gate)
    if str(getattr(coord, "adapt_trigger", "forecast")) == "random":
      # Placebo control: act at random iterations (rate placebo_rate), no detector decision and no gate. The draw
      # depends only on (placebo_seed, iteration), so repeated calls within an iteration agree.
      u = np.random.default_rng([int(getattr(coord, "placebo_seed", 0)), int(iteration)]).random()
      is_expected_spike = bool(u < float(getattr(coord, "placebo_rate", 0.0)))
    elif str(getattr(coord, "adapt_trigger", "forecast")) == "threshold":
      # Act on a confirmed spike (q_max rose by >= trigger_rise decades) rather than on the one-step forecast,
      # which validation showed to be late and noisy.
      is_expected_spike = self.threshold_spike(coord) and non_improvement_gate
    else:
      is_expected_spike = (
        decision.accept_forecast_prob
        and non_improvement_gate
      )

    # Both the decisor and the non-improvement gate must pass (paper Eq. ni_gate).
    if not is_expected_spike:
      return NO_EVENT

    if spike_location <= 0:
      spike_location = int(getattr(coord, "last_spike_subproblem", 0))

    # forecast_prob is a one-step-ahead signal, so expected spike timing is t+1.
    spike_future_iteration = int(iteration) + 1
    coord.last_spike_subproblem = spike_location
    coord.last_spike_iteration = spike_future_iteration
    coord.spike_detected = True
    return True, spike_location, spike_future_iteration

  def compute_scales(self, coord, iteration: int) -> Tuple[float, float, float, bool]:  # noqa: C901
    """Return (beta_scale, gamma_scale, budget_scale, spike) for this outer iteration.

    In adaptive mode each configured value is an upper bound reached at full intensity. Nothing is committed here,
    so the logger can call it without side effects.

    The beta rule is keyed on the objective: if it just decreased, spike_w_damp_scale (> 1 in the benchmark configs)
    accelerates penalty growth on stalled couplings; otherwise spike_w_amplify_scale (about 1) leaves it nearly
    unchanged.
    """
    neutral = (1.0, 1.0, 1.0)
    if not (getattr(coord, "is_AI_regularizer", False) and getattr(coord, "spike_influence_on_multipliers", False)):
      return (*neutral, False)

    is_expected_spike, _, _ = self.expected_spike_event(coord, iteration)
    max_adapt = int(getattr(coord, "max_adaptations", 0))
    if max_adapt > 0 and int(getattr(coord, "_n_adapt", 0)) >= max_adapt:
      coord._event_kind = "spike" if is_expected_spike else ""
      return (*neutral, is_expected_spike)
    freeze = (not is_expected_spike) and self.premature_freeze(coord, iteration)
    coord._event_kind = "spike" if is_expected_spike else ("freeze" if freeze else "")
    if not (is_expected_spike or freeze):
      return (*neutral, False)

    cooldown = int(getattr(coord, "adapt_cooldown", 0))
    if cooldown > 0 and int(iteration) - int(getattr(coord, "_last_adapt_iter", -10**9)) <= cooldown:
      return (*neutral, is_expected_spike)

    if freeze:
      # Consistency is being bought at the expense of the objective: slow penalty growth and relax the stall test,
      # and move evaluations to the subproblem that was still improving (see SubProblem.solve).
      beta_scale = float(getattr(coord, "spike_freeze_beta_scale", 0.5)) if getattr(coord, "adapt_beta", True) else 1.0
      gamma_scale = float(getattr(coord, "spike_freeze_gamma_scale", 1.5)) if getattr(coord, "adapt_gamma", True) else 1.0
      budget_scale = float(getattr(coord, "budget_scale", 6.0)) if getattr(coord, "adapt_budget", True) else 1.0
      return beta_scale, gamma_scale, budget_scale, False

    obj_hist = getattr(coord, "objective_history", [])
    obj_decreased = (
      len(obj_hist) >= 2
      and np.isfinite(obj_hist[-1])
      and np.isfinite(obj_hist[-2])
      and obj_hist[-1] < obj_hist[-2]
    )
    if obj_decreased:
      configured_beta_scale = float(getattr(coord, "spike_w_damp_scale", 1.8))
    else:
      configured_beta_scale = float(getattr(coord, "spike_w_amplify_scale", 1.05))
    configured_gamma_scale = float(getattr(coord, "spike_gamma_scale", 0.05))
    configured_budget_scale = float(getattr(coord, "budget_scale", 6.0))

    if not getattr(coord, "spike_adaptive_scales", True):
      beta_scale, gamma_scale, budget_scale = configured_beta_scale, configured_gamma_scale, configured_budget_scale
      if self.penalty_caused_spike(coord):
        beta_scale, gamma_scale = float(getattr(coord, "spike_w_hold_scale", 0.0)), 1.0
    else:
      forecast_prob = float(getattr(coord, "forecast_prob", 1.0))
      non_improvement_prob = float(getattr(coord, "non_improvement_prob", 1.0))
      intensity = float(np.clip(0.5 * forecast_prob + 0.5 * non_improvement_prob, 0.0, 1.0))
      budget_intensity = float(np.clip(1.5 * non_improvement_prob, 0.0, 1.0))
      beta_scale = 1.0 + (configured_beta_scale - 1.0) * intensity
      # gamma_scale tightens the stall threshold; the configured value is the tightest extreme.
      gamma_scale = 1.0 - (1.0 - configured_gamma_scale) * intensity
      budget_scale = 1.0 + (configured_budget_scale - 1.0) * budget_intensity
      if self.penalty_caused_spike(coord):
        # Accelerating the penalty would feed the spike: hold w (toward spike_w_hold_scale) and keep gamma.
        hold = float(getattr(coord, "spike_w_hold_scale", 0.0))
        beta_scale = 1.0 - (1.0 - hold) * intensity
        gamma_scale = 1.0

    if not getattr(coord, "adapt_beta", True):
      beta_scale = 1.0
    if not getattr(coord, "adapt_gamma", True):
      gamma_scale = 1.0
    if not getattr(coord, "adapt_budget", True):
      budget_scale = 1.0
    return beta_scale, gamma_scale, budget_scale, True

  # ------------------------------------------------------------------------------------------------ triggers and gates
  @staticmethod
  def threshold_spike(coord) -> bool:
    """q_max rose by at least trigger_rise decades at this iteration, from above twice the tolerance."""
    hist = list(getattr(coord, "_qlog_hist", []))
    if not hist or coord.q is None or not np.size(coord.q):
      return False
    tol = float(getattr(coord, "_tol_ref", 0.0))
    q_now = max(float(np.max(np.abs(coord.q))), 1e-300)
    q_prev = 10.0 ** hist[-1]
    if q_prev < 2.0 * tol:
      return False
    return bool(np.log10(q_now) - hist[-1] >= float(getattr(coord, "trigger_rise", 0.3)))

  @staticmethod
  def harm_gate(coord) -> bool:
    """Act when the objective dropped together with the spike, relative to its recent noise.

    p_harm = Phi(-dS / s), dS = slog f_k - slog f_(k-1), s = median |dS| over the previous harm_window changes (floor
    1e-6). Validation: an objective drop at a spike predicts that the objective will be worse a few iterations later.
    """
    obj = np.asarray(getattr(coord, "objective_history", []), dtype=float)
    if obj.size < 3 or not np.all(np.isfinite(obj[-2:])):
      return False
    so = np.sign(obj) * np.log1p(np.abs(obj))
    dso = np.diff(so)
    window = int(getattr(coord, "harm_window", 5))
    prev = dso[-window - 1:-1]
    prev = prev[np.isfinite(prev)]
    scale = max(float(np.median(np.abs(prev))) if prev.size else 0.0, 1e-6)
    p_harm = 0.5 * (1.0 + math.erf((-dso[-1] / scale) / math.sqrt(2.0)))
    coord.harm_prob = float(p_harm)
    return p_harm >= float(getattr(coord, "harm_threshold", 0.45))

  @staticmethod
  def direct_non_improvement(coord) -> bool:
    """Objective worsened this iteration, or log q_max has been flat (span < 0.05) over the last four iterations."""
    obj_hist = getattr(coord, "objective_history", [])
    worsened = (
      len(obj_hist) >= 2 and np.isfinite(obj_hist[-1]) and np.isfinite(obj_hist[-2])
      and float(obj_hist[-1]) > float(obj_hist[-2]) + 1e-4 * max(1.0, abs(float(obj_hist[-2])))
    )
    qlog = list(getattr(coord, "_qlog_hist", []))
    if coord.q is not None and np.size(coord.q):
      qlog.append(float(np.log10(max(float(np.max(np.abs(coord.q))), 1e-30))))
    stalled = len(qlog) >= 4 and (max(qlog[-4:]) - min(qlog[-4:])) < 0.05
    return bool(worsened or stalled)

  @staticmethod
  def premature_freeze(coord, iteration: int) -> bool:
    """Causal response only: penalties grew last iteration and q_max fell, yet the objective did not improve.

    That is the signature of the coordination converging on consistency before optimality, which the paper's
    accelerating actions make worse.
    """
    if str(getattr(coord, "spike_response", "accelerate")) != "causal":
      return False
    if int(iteration) < int(getattr(coord, "detector_min_history", 5)):
      return False
    if not getattr(coord, "_w_grew_last", False) or coord.q is None or not np.size(coord.q):
      return False
    if float(np.max(np.abs(coord.q))) >= float(getattr(coord, "_qmax_prev", np.inf)):
      return False
    obj_hist = getattr(coord, "objective_history", [])
    if len(obj_hist) < 2 or not (np.isfinite(obj_hist[-1]) and np.isfinite(obj_hist[-2])):
      return False
    tol = float(getattr(coord, "freeze_obj_rtol", 1e-4)) * max(1.0, abs(float(obj_hist[-2])))
    return float(obj_hist[-1]) >= float(obj_hist[-2]) - tol

  def penalty_caused_spike(self, coord) -> bool:
    """True when the bidirectional or causal response is on and q_max rose right after the last penalty increase."""
    if str(getattr(coord, "spike_response", "accelerate")) not in ("bidirectional", "causal"):
      return False
    if not getattr(coord, "_w_grew_last", False) or coord.q is None or not np.size(coord.q):
      return False
    return float(np.max(np.abs(coord.q))) > float(getattr(coord, "_qmax_prev", np.inf))


_instances: Dict[type, CoordinationPolicy] = {}


def policy_for(coord) -> CoordinationPolicy:
  """The adaptation policy of a coordinator (see the module docstring for how it is chosen)."""
  name = getattr(coord, "adaptation_policy", None)
  if not name:
    name = "spike" if getattr(coord, "is_AI_regularizer", False) else "none"
  component = name if isinstance(name, (CoordinationPolicy, type)) else policies.get(name)
  if isinstance(component, CoordinationPolicy):
    return component
  instance = _instances.get(component)
  if instance is None:
    instance = _instances[component] = component()
  return instance
