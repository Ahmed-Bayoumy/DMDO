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

from dataclasses import dataclass

import numpy as np

def compute_confidence(
    unc: float,
    prob: float,
    forecast_prob: float,
    non_improvement_prob: float,
    n_history: int,
    min_history: int,
    window: int,
) -> float:
    """
    Returns a confidence score in [0, 1].
    Higher = more trustworthy inferred probabilities.
    """
    # --- Component 1: Certainty (inverse of uncertainty) ---
    certainty = 1.0 - unc  # unc in [0.05, 1.0] → certainty in [0.0, 0.95]

    # --- Component 2: History maturity ---
    # Ramps from 0 to 1 as history grows from min_history to window
    maturity = float(np.clip(
        (n_history - min_history) / max(window - min_history, 1),
        0.0, 1.0
    ))

    # --- Component 3: Signal agreement ---
    # How consistent are prob, forecast_prob, and non_improvement_prob?
    signals = np.array([prob, forecast_prob, non_improvement_prob])
    signal_std = float(np.std(signals))
    # Low std → signals agree → high agreement score
    agreement = float(np.clip(1.0 - 2.0 * signal_std, 0.0, 1.0))

    # --- Weighted combination ---
    confidence = (
        0.45 * certainty +
        0.35 * maturity +
        0.20 * agreement
    )
    return float(np.clip(confidence, 0.0, 1.0))

@dataclass
class ConfidenceLevel:
    name: str
    min_confidence: float       # Minimum confidence score to trust the inference
    prob_threshold: float       # Minimum prob to flag a spike
    forecast_threshold: float   # Minimum forecast_prob to flag a predictive spike
    non_improvement_threshold: float  # Minimum non_improvement_prob to flag stagnation
    require_alarm: bool         # Whether to also require alarm == 1

# ┌─────────────────┬──────────────┬──────────────┬───────────────────────────────┐
# │ Level           │ Min Conf.    │ Prob Gate    │ Best For                      │
# ├─────────────────┼──────────────┼──────────────┼───────────────────────────────┤
# │ L1 Conservative │ 0.75         │ 0.70         │ Production decisions, alerts  │
# │ L2 Normal       │ 0.50         │ 0.50         │ Standard monitoring, logging  │
# │ L3 Predictive   │ 0.25         │ 0.30         │ Early warning, exploration    │
# └─────────────────┴──────────────┴──────────────┴───────────────────────────────┘

# ── L1: Conservative / Cautious ──────────────────────────────────────────────
# Only act when confidence is high AND signals are strong.
# Minimizes false positives. Best for: production-critical decisions.
L1_CONSERVATIVE = ConfidenceLevel(
    name="L1_Conservative",
    min_confidence=0.75,
    prob_threshold=0.70,
    forecast_threshold=0.65,
    non_improvement_threshold=0.72,
    require_alarm=True,
)

# ── L2: Average / Normal ─────────────────────────────────────────────────────
# Balanced trade-off. Acts on moderately confident signals.
# Best for: standard monitoring, dashboards, logging.
L2_NORMAL = ConfidenceLevel(
    name="L2_Normal",
    min_confidence=0.50,
    prob_threshold=0.50,
    forecast_threshold=0.45,
    non_improvement_threshold=0.55,
    require_alarm=False,
)

# ── L3: Predictive / Greedy ──────────────────────────────────────────────────
# Acts early, even on weak signals. Accepts lower confidence.
# Best for: early warning, exploratory analysis, research.
L3_PREDICTIVE = ConfidenceLevel(
    name="L3_Predictive",
    min_confidence=0.25,
    prob_threshold=0.30,
    forecast_threshold=0.25,
    non_improvement_threshold=0.35,
    require_alarm=False,
)

@dataclass
class InferenceDecision:
    level: str
    confidence: float
    accept_spike_prob: bool
    accept_forecast_prob: bool
    accept_non_improvement: bool
    accept_inconsistency: bool   # Combined: spike OR forecast accepted
    reason: str


_LEVELS = {
    "l1": L1_CONSERVATIVE,
    "l1_conservative": L1_CONSERVATIVE,
    "l2": L2_NORMAL,
    "l2_normal": L2_NORMAL,
    "l3": L3_PREDICTIVE,
    "l3_predictive": L3_PREDICTIVE,
}


def resolve_level(level_name: str) -> ConfidenceLevel:
    """Map a policy name to its ConfidenceLevel; unknown names raise instead of silently using L2."""
    normalized = str(level_name).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in _LEVELS:
        raise ValueError(
            f"Unknown AI_spike_policy '{level_name}'. Valid names: L1_Conservative, L2_Normal, L3_Predictive "
            "(aliases L1, L2, L3)."
        )
    return _LEVELS[normalized]

