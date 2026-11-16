import math
import os
import random
import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, List, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._decisor import ConfidenceLevel, InferenceDecision, compute_confidence, resolve_level


# =============================================
# Indicators and features used by the detector:
# ============================================
# Max_inc_predicted: The detector's predicted near-future inconsistency signal level
# used as an early warning proxy for a spike in Max. inconsistency. In this wiring,
# this is fed from the detector's forecast-side probability track.

# Spike_prob: The current-step probability that the present point is a spike event
# in Max. inconsistency, based on local jump/curvature evidence plus Bayesian
# residual evidence.

# Forecast_prob: The one-step-ahead probability that the next iteration will exceed
# the detector's spike threshold (forecasted spike likelihood).

# Non_improvement_prob: The combined probability that a spike is associated with
# overall non-improvement, where non-improvement means objective worsening or
# flatness, and/or worsening constraint-violation behavior.

# Obj_non_improvement_prob: The probability that objective behavior is non-improving
# around a spike context. In this implementation, flat objective counts as partial
# non-improvement, and increasing objective counts as strong non-improvement.

# Constraint_non_improvement_prob: The probability that constraint behavior is
# non-improving around a spike context, driven by hmin_* severity and divergence
# above the threshold level.

# Detector_uncertainty: The detector's confidence complement for the spike decision
# at the current step. Higher values mean less certainty in the current inference;
# lower values mean stronger confidence.

# Constraint_pressure: A normalized severity indicator from the hmin_* vector
# capturing how strongly constraints are "under stress," combining exceedance
# above 0.001, spread across hmin_* entries, and recent growth/divergence trend.

# Spike_location: is the subproblem index that triggered a detector alarm at that step.


# ============================================
# Important Remarks on Confidence Computation:
# ============================================
# Confidence is separate from probability. A high prob with low confidence
# (e.g., only 3 data points) is correctly rejected at L1/L2.
# Maturity ramp prevents early false positives during warm-up, which aligns with your existing min_history guard
# Signal agreement penalizes cases where prob and forecast_prob wildly disagree,
# a sign the model is uncertain about which signal to trust.
# require_alarm at L1 adds the existing geometric z-score gate as an extra safety net for conservative mode

# ==========
# AI Models
# ==========



class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, generator: Optional[torch.Generator] = None):
        super().__init__()
        self._generator = generator if generator is not None else torch.Generator()
        if generator is None:
            self._generator.manual_seed(torch.initial_seed())
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_mu, generator=self._generator)
        nn.init.constant_(self.weight_rho, -3.0)
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_rho, -3.0)

    @staticmethod
    def _sigma(rho):
        return F.softplus(rho) + 1e-6

    def forward(self, x):
        weight_sigma = self._sigma(self.weight_rho)
        bias_sigma = self._sigma(self.bias_rho)
        weight_noise = torch.randn(
            self.weight_mu.shape,
            device=self.weight_mu.device,
            dtype=self.weight_mu.dtype,
            generator=self._generator,
        )
        bias_noise = torch.randn(
            self.bias_mu.shape,
            device=self.bias_mu.device,
            dtype=self.bias_mu.dtype,
            generator=self._generator,
        )
        weight = self.weight_mu + weight_noise * weight_sigma
        bias = self.bias_mu + bias_noise * bias_sigma
        return F.linear(x, weight, bias)


class BayesianLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, generator: Optional[torch.Generator] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_linear = BayesianLinear(input_size, 4 * hidden_size, generator=generator)
        self.hidden_linear = BayesianLinear(hidden_size, 4 * hidden_size, generator=generator)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h_t = x.new_zeros(batch_size, self.hidden_size)
        c_t = x.new_zeros(batch_size, self.hidden_size)
        outputs = []

        for step in range(seq_len):
            gates = self.input_linear(x[:, step, :]) + self.hidden_linear(h_t)
            in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, dim=-1)
            in_gate = torch.sigmoid(in_gate)
            forget_gate = torch.sigmoid(forget_gate)
            cell_gate = torch.tanh(cell_gate)
            out_gate = torch.sigmoid(out_gate)

            c_t = forget_gate * c_t + in_gate * cell_gate
            h_t = out_gate * torch.tanh(c_t)
            outputs.append(h_t)

        return torch.stack(outputs, dim=1), (h_t.unsqueeze(0), c_t.unsqueeze(0))


class BayesianGRULayer(nn.Module):
    def __init__(self, input_size, hidden_size, generator: Optional[torch.Generator] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_linear = BayesianLinear(input_size, 3 * hidden_size, generator=generator)
        self.hidden_linear = BayesianLinear(hidden_size, 3 * hidden_size, generator=generator)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h_t = x.new_zeros(batch_size, self.hidden_size)
        outputs = []

        for step in range(seq_len):
            x_proj = self.input_linear(x[:, step, :])
            h_proj = self.hidden_linear(h_t)
            x_reset, x_update, x_new = x_proj.chunk(3, dim=-1)
            h_reset, h_update, h_new = h_proj.chunk(3, dim=-1)

            reset_gate = torch.sigmoid(x_reset + h_reset)
            update_gate = torch.sigmoid(x_update + h_update)
            candidate = torch.tanh(x_new + reset_gate * h_new)
            h_t = (1.0 - update_gate) * candidate + update_gate * h_t
            outputs.append(h_t)

        return torch.stack(outputs, dim=1), h_t.unsqueeze(0)


class _BayesianSequenceModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, dropout=0.2, mc_samples=16, generator: Optional[torch.Generator] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.mc_samples = mc_samples
        self._generator = generator if generator is not None else torch.Generator()
        if generator is None:
            self._generator.manual_seed(torch.initial_seed())
        self.fc = BayesianLinear(hidden_size, 1, generator=self._generator)
        self.last_uncertainty = 0.0

    def _encode(self, x):
        raise NotImplementedError

    def _single_sample(self, x):
        encoded = self._encode(x)
        if self.dropout > 0.0:
            keep_prob = 1.0 - self.dropout
            dropout_mask = torch.rand(
                encoded.shape,
                device=encoded.device,
                dtype=encoded.dtype,
                generator=self._generator,
            ) < keep_prob
            encoded = encoded * dropout_mask.to(encoded.dtype) / keep_prob
        return self.fc(encoded)

    def forward(self, x):
        prediction = self._single_sample(x)
        self.last_uncertainty = 0.0
        return prediction

    def predict_distribution(self, x, samples=None):
        sample_count = self.mc_samples if samples is None else samples
        predictions = [self._single_sample(x) for _ in range(sample_count)]
        stacked = torch.stack(predictions, dim=0)
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0, unbiased=False)
        self.last_uncertainty = float(std.mean().item())
        return mean, std


class BayesianLSTM(_BayesianSequenceModel):
    def __init__(self, input_size=4, hidden_size=32, generator: Optional[torch.Generator] = None):
        super().__init__(input_size=input_size, hidden_size=hidden_size, generator=generator)
        self.rnn = BayesianLSTMLayer(input_size, hidden_size, generator=self._generator)

    def _encode(self, x):
        out, _ = self.rnn(x)
        return out[:, -1, :]


class BayesianGRU(_BayesianSequenceModel):
    def __init__(self, input_size=4, hidden_size=32, generator: Optional[torch.Generator] = None):
        super().__init__(input_size=input_size, hidden_size=hidden_size, generator=generator)
        self.rnn = BayesianGRULayer(input_size, hidden_size, generator=self._generator)

    def _encode(self, x):
        out, _ = self.rnn(x)
        return out[:, -1, :]


class _MLPBackbone(nn.Module):
    """Deterministic feature extractor: raw phi (n_raw) → learned embedding z (embedding_dim).

    Frozen during online RLS updates; re-trained jointly with a temporary linear
    head on the accumulated (phi, y) buffer every `refit_every` steps (Strategy-2).
    """

    def __init__(self, in_features: int = 13, hidden: int = 16, out_features: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def embed(self, phi: np.ndarray) -> np.ndarray:
        """Numpy interface: embeds a single raw feature vector."""
        self.eval()
        with torch.no_grad():
            t = torch.tensor(phi, dtype=torch.float32).unsqueeze(0)
            return self.forward(t).squeeze(0).cpu().numpy()


# ==============
# Hybrid detector
# ==============

def _sigmoid(value: float) -> float:
    # Numerically stable logistic for extreme evidence values.
    x = float(value)
    if x >= 60.0:
        return 1.0
    if x <= -60.0:
        return 0.0
    if x >= 0.0:
        z = math.exp(-x)
        return float(1.0 / (1.0 + z))
    z = math.exp(x)
    return float(z / (1.0 + z))


def _normal_cdf(value: float) -> float:
    return float(0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0))))


def _signed_log1p(value):
    """sign(x) * log(1 + |x|): monotone and defined for negative objectives."""
    arr = np.asarray(value, dtype=float)
    return np.sign(arr) * np.log1p(np.abs(arr))


_PRIOR_VAR = 25.0
_COV_TRACE_CAP = 1e4
_Z_CLIP = 6.0


def _rls_step(theta, cov, z, y, forgetting, noise, trace_cap=_COV_TRACE_CAP):
    """Forgetting-factor Bayesian RLS: precision <- lam * precision + z z^T / noise.

    The noise variance enters the gain so the covariance is consistent with the
    predictive variance noise + z^T cov z used by the detector.
    """
    denom = forgetting * noise + float(z @ cov @ z)
    gain = (cov @ z) / max(denom, 1e-12)
    theta = theta + gain * (y - float(z @ theta))
    cov = (cov - np.outer(gain, z) @ cov) / forgetting
    cov = 0.5 * (cov + cov.T)
    trace = float(np.trace(cov))
    if trace > trace_cap:
        cov = cov * (trace_cap / trace)
    return theta, cov


class _IdentityBackbone:
    """Ablation backbone: RLS heads act directly on the raw feature vector."""

    def __init__(self, in_features: int = 13) -> None:
        self.in_features = int(in_features)

    def embed(self, phi: np.ndarray) -> np.ndarray:
        return np.asarray(phi, dtype=float).copy()


def _set_deterministic_seed(seed: int) -> None:
    seed_value = int(seed)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except (AttributeError, RuntimeError):
        pass


# ======================
# SP-level data ingestion
# ======================

class SolveQualityExtractor:
    """Extracts a fixed-size quality fingerprint from a completed SP-solve output CSV.

    Returns a 7-element numpy array:
      [feasibility_rate, obj_improvement, obj_roughness,
       inc_range, inc_trend, penalty_trend, n_evals_log]
    All values are finite, clipped, and zero-filled on any read failure.
    """

    N_FEATURES: int = 7
    _ZERO: np.ndarray = np.zeros(N_FEATURES, dtype=float)

    @staticmethod
    def extract(filepath: Optional[str]) -> np.ndarray:
        if not filepath or not os.path.isfile(filepath):
            return SolveQualityExtractor._ZERO.copy()

        try:
            df = pd.read_csv(filepath, sep=None, engine="python")
        except Exception:
            return SolveQualityExtractor._ZERO.copy()

        cols = {str(c).strip(): c for c in df.columns}
        inc_col = cols.get("Max. inconsistency")
        obj_col = cols.get("Objective")
        penalty_col = cols.get("Penalty")
        status_col = cols.get("Coupling Status")

        if inc_col is None or obj_col is None or len(df) < 2:
            return SolveQualityExtractor._ZERO.copy()

        try:
            inc = pd.to_numeric(df[inc_col], errors="coerce").fillna(0.0).values.astype(float)
            obj = pd.to_numeric(df[obj_col], errors="coerce").fillna(0.0).values.astype(float)
            n = max(len(obj), 1)

            # Feature 1: feasibility rate
            if status_col is not None:
                feasibility_rate = float((df[status_col].astype(str).str.strip() == "Feasible").sum()) / n
            else:
                feasibility_rate = 0.5

            # Feature 2: normalized objective improvement (minimization sense)
            obj_first = float(obj[0])
            obj_best = float(np.nanmin(obj))
            obj_improvement = float(np.clip((obj_first - obj_best) / (abs(obj_first) + 1.0), -1.0, 1.0))

            # Feature 3: landscape roughness (normalized objective std)
            obj_roughness = float(np.clip(np.nanstd(obj) / (abs(np.nanmean(obj)) + 1.0), 0.0, 10.0))

            # Feature 4: log-space inconsistency range within solve
            log_inc = np.log10(np.clip(inc, 1e-12, None))
            inc_range = float(np.clip(np.nanmax(log_inc) - np.nanmin(log_inc), 0.0, 10.0))

            # Feature 5: log-space inconsistency slope per evaluation
            inc_trend = float(np.clip((log_inc[-1] - log_inc[0]) / max(n - 1, 1), -1.0, 1.0))

            # Feature 6: relative penalty change across the solve
            if penalty_col is not None:
                penalty = pd.to_numeric(df[penalty_col], errors="coerce").fillna(0.0).values.astype(float)
                penalty_trend = float(np.clip(
                    (float(penalty[-1]) - float(penalty[0])) / (abs(float(penalty[0])) + 1.0),
                    -1.0, 1.0
                ))
            else:
                penalty_trend = 0.0

            # Feature 7: log-normalized evaluation count (proxy for solver effort)
            n_evals_log = float(np.log10(max(n, 1)))

            features = np.array([
                feasibility_rate,
                obj_improvement,
                obj_roughness,
                inc_range,
                inc_trend,
                penalty_trend,
                n_evals_log,
            ], dtype=float)
            return np.where(np.isfinite(features), features, 0.0)

        except Exception:
            return SolveQualityExtractor._ZERO.copy()


# ========================
# Online Bayesian RLS head
# ========================

class _OnlineBayesianHead:
    """Online Bayesian linear regression head (forgetting-factor RLS) in embedding space.

    Heads differ only in their forgetting factor, i.e. their memory horizon. The caller
    owns the one-step lag: update(z_prev, y) assimilates the pair (z(k-1), y_k).
    """

    def __init__(self, n_features: int, forgetting: float, prior_var: float = _PRIOR_VAR) -> None:
        self.forgetting = float(np.clip(forgetting, 0.90, 0.999))
        self.prior_var = float(prior_var)
        self.reset(n_features)

    def reset(self, n_features: Optional[int] = None) -> None:
        """Return the posterior to the prior (used before a backbone replay)."""
        if n_features is not None:
            self._n = int(n_features)
        self._theta = np.zeros(self._n, dtype=float)
        self._cov = np.eye(self._n, dtype=float) * self.prior_var

    def update(self, z_prev: np.ndarray, y: float, noise: float) -> None:
        self._theta, self._cov = _rls_step(self._theta, self._cov, z_prev, y, self.forgetting, noise)

    def predict(self, z: np.ndarray) -> tuple:
        """Returns (predictive mean, parameter variance z^T Sigma z); add the noise variance separately."""
        return float(z @ self._theta), float(max(z @ self._cov @ z, 0.0))


def evaluate_inference(
    detector: "BayesianOnlineSpikeDetector",
    level: ConfidenceLevel,
) -> InferenceDecision:
    """
    Evaluates the latest detector state against a confidence level threshold.
    Returns an InferenceDecision describing what to accept.
    """
    if not detector.state["prob"]:
        return InferenceDecision(
            level=level.name,
            confidence=0.0,
            accept_spike_prob=False,
            accept_forecast_prob=False,
            accept_non_improvement=False,
            accept_inconsistency=False,
            reason="No data yet",
        )

    # Pull latest values
    prob = detector.state["prob"][-1]
    forecast_prob = detector.state["forecast_prob"][-1]
    non_improvement_prob = detector.state["non_improvement_prob"][-1]
    unc = detector.state["unc"][-1]
    alarm = detector.state["alarm"][-1]
    n_history = len(detector.inc_hist)

    # Compute confidence
    confidence = compute_confidence(
        unc=unc,
        prob=prob,
        forecast_prob=forecast_prob,
        non_improvement_prob=non_improvement_prob,
        n_history=n_history,
        min_history=detector.min_history,
        window=detector.window,
    )

    # Gate 1: Is confidence sufficient?
    confidence_ok = confidence >= level.min_confidence

    # Gate 2: Alarm requirement
    alarm_ok = (not level.require_alarm) or (alarm == 1)

    # Individual signal acceptance
    accept_spike = confidence_ok and alarm_ok and (prob >= level.prob_threshold)
    # Adaptation is driven by the forecast channel, so the conservative level's alarm gate must apply to it too.
    accept_forecast = confidence_ok and alarm_ok and (forecast_prob >= level.forecast_threshold)
    accept_non_improvement = confidence_ok and (
        non_improvement_prob >= level.non_improvement_threshold
    )

    # Inconsistency = spike signal is trustworthy (either current or forecast)
    accept_inconsistency = accept_spike or accept_forecast

    # Build a human-readable reason
    reasons = []
    if not confidence_ok:
        reasons.append(f"confidence {confidence:.2f} < {level.min_confidence}")
    if level.require_alarm and not alarm_ok:
        reasons.append("alarm not triggered")
    if not reasons:
        reasons.append("all gates passed")

    return InferenceDecision(
        level=level.name,
        confidence=confidence,
        accept_spike_prob=accept_spike,
        accept_forecast_prob=accept_forecast,
        accept_non_improvement=accept_non_improvement,
        accept_inconsistency=accept_inconsistency,
        reason="; ".join(reasons),
    )



class BayesianOnlineSpikeDetector:
    def __init__(
        self,
        datafile: Optional[str] = None,
        data_file: Optional[str] = None,
        model_type: str = "LSTM",
        window: int = 8,
        min_history: int = 5,
        forgetting: float = 0.97,
        spike_sigma: float = 1.75,
        flat_objective_tol: float = 1e-4,
        hmin_threshold: float = 1e-3,
        seed: int = 42,
        embedding_dim: int = 8,
        refit_every: int = 20,
        *,
        tol_floor: float = 0.0,
        tol_floor_mult: float = 2.0,
        hmin_scale: float = 1.0,
        noise_init: float = 0.08,
        noise_floor: float = 1e-3,
        noise_beta: float = 0.1,
        score_std_floor: float = 0.02,
        evidence_offset: float = 1.85,
        evidence_weights: Sequence[float] = (1.1, 0.9, 0.7),
        ensemble_forgetting: Sequence[float] = (0.95, 0.97, 0.99),
        backbone: str = "mlp",
        use_sp_features: bool = True,
        use_maturity: bool = True,
        use_geometric_alarm: bool = True,
    ):
        _set_deterministic_seed(seed)
        self._base_seed: int = int(seed)
        self._refit_count: int = 0
        # Private generator so refit weight init never touches the global torch RNG.
        # Seeded once here and re-seeded per refit inside _refit_backbone.
        self._torch_gen: torch.Generator = torch.Generator()
        self._torch_gen.manual_seed(self._base_seed)

        self.model_type = model_type
        self.window = max(5, int(window))
        self.min_history = max(3, min(int(min_history), self.window))
        self.forgetting = float(np.clip(forgetting, 0.9, 0.999))
        self.spike_sigma = spike_sigma
        self.flat_objective_tol = flat_objective_tol
        self.hmin_threshold = hmin_threshold
        # Convergence-tolerance floor: below tol_floor * tol_floor_mult detection is pointless (solver noise).
        self.tol_floor = float(tol_floor)
        self.tol_floor_mult = float(tol_floor_mult)
        self.hmin_scale = max(float(hmin_scale), 1e-12)
        self.score_std_floor = float(score_std_floor)
        self.evidence_offset = float(evidence_offset)
        self.evidence_weights = tuple(float(w) for w in evidence_weights)
        self.use_sp_features = bool(use_sp_features)
        self.use_maturity = bool(use_maturity)
        self.use_geometric_alarm = bool(use_geometric_alarm)

        self.state = {
            "score": [],
            "prob": [],
            "unc": [],
            "risk": [],
            "alarm": [],
            "forecast_prob": [],
            "non_improvement_prob": [],
            "objective_non_improvement_prob": [],
            "constraint_non_improvement_prob": [],
            "constraint_pressure": [],
        }
        # Per-step diagnostics used by the offline detector evaluation.
        self.diag = {"z_local": [], "z_resid": [], "z_fore": [], "noise_var": [], "epi_frac": []}

        self.obj_hist = []
        self.inc_hist = []
        self.hmin_hist = []
        self._ell0: Optional[float] = None

        self._score_m1 = 0.0
        self._score_m2 = 1e-4
        self._stat_beta = 0.18
        self._event_buffer = deque(maxlen=3)

        # Feature vector is 6 coord-level + 7 SP-fingerprint = 13 total
        self._sp_features: np.ndarray = np.zeros(SolveQualityExtractor.N_FEATURES, dtype=float)
        self._refit_every = 0 if int(refit_every) <= 0 else max(4, int(refit_every))
        self._backbone_kind = str(backbone).strip().lower()
        if self._backbone_kind == "identity":
            self._embedding_dim = 13
            self._backbone = _IdentityBackbone(in_features=13)
            self._refit_every = 0
        elif self._backbone_kind == "mlp":
            self._embedding_dim = max(4, int(embedding_dim))
            self._backbone = _MLPBackbone(in_features=13, hidden=16, out_features=self._embedding_dim)
        else:
            raise ValueError(f"Unknown detector backbone '{backbone}'. Use 'mlp' or 'identity'.")
        self._nlm_phi: deque = deque(maxlen=200)
        self._nlm_y: deque = deque(maxlen=200)
        self._prev_phi: Optional[np.ndarray] = None
        self._cur_phi: Optional[np.ndarray] = None
        self._steps_since_refit: int = 0
        self._theta = np.zeros(self._embedding_dim, dtype=float)
        self._cov = np.eye(self._embedding_dim, dtype=float) * _PRIOR_VAR
        # Online observation-noise variance (log10 units^2), shared by the reference head and the ensemble.
        self._noise = float(noise_init)
        self._noise_floor = float(noise_floor)
        self._noise_beta = float(noise_beta)
        # Embedding z(k-1) of the previous step; paired with the next observation y_k.
        self._last_phi: Optional[np.ndarray] = None
        # Ensemble of NLM heads with different forgetting rates (memory horizons).
        self._ensemble: List[_OnlineBayesianHead] = [
            _OnlineBayesianHead(self._embedding_dim, forgetting=float(f)) for f in ensemble_forgetting
        ]

        self._assoc_alpha = 1.0
        self._assoc_beta = 1.0
        self._obj_assoc_alpha = 1.0
        self._obj_assoc_beta = 1.0
        self._con_assoc_alpha = 1.0
        self._con_assoc_beta = 1.0

        source = data_file or datafile
        if source is not None:
            self._warm_start_from_file(source)

    @property
    def alarm_history(self):
        return self.state["alarm"]

    @property
    def risk_history(self):
        return self.state["risk"]

    @property
    def prob_history(self):
        return self.state["prob"]

    @property
    def unc_history(self):
        return self.state["unc"]

    @property
    def forecast_prob_history(self):
        return self.state["forecast_prob"]

    @property
    def non_improvement_prob_history(self):
        return self.state["non_improvement_prob"]

    @property
    def objective_non_improvement_prob_history(self):
        return self.state["objective_non_improvement_prob"]

    @property
    def constraint_non_improvement_prob_history(self):
        return self.state["constraint_non_improvement_prob"]

    def _warm_start_from_file(self, datafile: str):
        if not datafile or not os.path.isfile(datafile):
            return

        try:
            df = pd.read_csv(datafile, sep=None, engine="python")
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            return

        columns = {str(column).strip(): column for column in df.columns}
        obj_col = columns.get("Objective")
        inc_col = columns.get("Max. inconsistency")
        if obj_col is None or inc_col is None:
            return

        hmin_columns = [column for key, column in columns.items() if key.startswith("hmin_")]
        for _, row in df.iterrows():
            try:
                hmins = [float(row[column]) for column in hmin_columns] if hmin_columns else None
                self.update(obj=float(row[obj_col]), inc=float(row[inc_col]), hmins=hmins)
            except (TypeError, ValueError):
                continue

    def _safe_log_inc(self, value: float) -> float:
        return float(np.log10(np.clip(value, 1e-12, None)))

    def _sanitize_value(self, value: float, floor: float = 0.0, severe_cap: Optional[float] = None) -> float:
        numeric = float(value)
        if math.isnan(numeric):
            return floor
        if math.isinf(numeric):
            if numeric > 0.0:
                return severe_cap if severe_cap is not None else max(floor, 1.0)
            return floor
        return max(numeric, floor)

    def _sanitize_objective(self, value: float) -> float:
        """Objectives may be negative (e.g. negated range); only non-finite values are repaired."""
        numeric = float(value)
        if math.isnan(numeric):
            return float(self.obj_hist[-1]) if self.obj_hist else 0.0
        return float(np.clip(numeric, -1e12, 1e12))

    def _history_tail(self, values: Sequence[float]) -> np.ndarray:
        return np.asarray(values[-self.window:], dtype=float)

    def update_sp_features(self, filepath: Optional[str]) -> None:
        """Ingest a completed SP-solve output CSV and cache its quality fingerprint.

        Call this once per outer iteration (before detector.update()) with the log of the
        subproblem attributed to the largest inconsistency.
        """
        if not self.use_sp_features:
            self._sp_features = np.zeros(SolveQualityExtractor.N_FEATURES, dtype=float)
            return
        self._sp_features = SolveQualityExtractor.extract(filepath)

    def _constraint_pressure(self, hmins: Optional[Sequence[float]]) -> float:
        if not hmins:
            return 0.0

        current = np.maximum(np.asarray(hmins, dtype=float), 0.0)
        current_max = float(np.max(current)) if current.size else 0.0
        severity = np.clip((current_max - self.hmin_threshold) / max(self.hmin_threshold, 1e-8), 0.0, 4.0)
        severity = np.tanh(severity)

        spread = float(np.std(current)) if current.size > 1 else current_max
        spread_term = np.tanh(spread / max(self.hmin_threshold, 1e-8)) if spread > 0.0 else 0.0

        growth = 0.0
        if self.hmin_hist:
            previous = np.maximum(np.asarray(self.hmin_hist[-1], dtype=float), 0.0)
            prev_max = float(np.max(previous)) if previous.size else 0.0
            growth = np.clip((current_max - prev_max) / max(current_max, self.hmin_threshold, 1e-8), 0.0, 1.0)

        return float(np.clip(0.55 * severity + 0.2 * spread_term + 0.25 * growth, 0.0, 1.0))

    def _objective_non_improvement_score(self) -> float:
        if len(self.obj_hist) < 2:
            return 0.0

        current = float(self.obj_hist[-1])
        previous = float(self.obj_hist[-2])
        tol = self.flat_objective_tol * max(abs(previous), 1.0)
        delta = current - previous
        if delta > tol:
            return 1.0
        # ADMM objectives rarely improve every iteration; flat counts as weak
        # non-improvement (0.35) rather than strong (0.75) to avoid inflating
        # non_improvement_prob and triggering excessive spike responses.
        if abs(delta) <= tol:
            return 0.35
        return 0.10

    def _constraint_non_improvement_score(self, current_pressure: float) -> float:
        if current_pressure <= 0.0:
            return 0.0
        if len(self.hmin_hist) < 2:
            return current_pressure

        previous = np.maximum(np.asarray(self.hmin_hist[-2], dtype=float), 0.0)
        current = np.maximum(np.asarray(self.hmin_hist[-1], dtype=float), 0.0)
        prev_max = float(np.max(previous)) if previous.size else 0.0
        curr_max = float(np.max(current)) if current.size else 0.0
        diverging = 1.0 if curr_max > max(prev_max, self.hmin_threshold) else 0.35
        return float(np.clip(0.65 * current_pressure + 0.35 * diverging, 0.0, 1.0))

    def _make_feature_vector(self) -> np.ndarray:
        li = self._history_tail([self._safe_log_inc(value) for value in self.inc_hist])
        obj = _signed_log1p(self._history_tail(self.obj_hist))
        dx = np.diff(li, prepend=li[0])
        ddx = np.diff(dx, prepend=dx[0])
        obj_dx = np.diff(obj, prepend=obj[0])
        constraint_pressure = self._constraint_pressure(self.hmin_hist[-1] if self.hmin_hist else None)
        sp = getattr(self, "_sp_features", np.zeros(SolveQualityExtractor.N_FEATURES, dtype=float))
        # Level is measured relative to the first observed inconsistency so the features do not
        # depend on the units of the coupling variables.
        ell0 = self._ell0 if self._ell0 is not None else float(li[0])
        return np.asarray(
            [
                1.0,
                li[-1] - ell0,
                float(dx[-1]),
                float(ddx[-1]),
                float(obj_dx[-1]),
                constraint_pressure,
                # SP-level fingerprint (7 features from inner-loop trajectory):
                float(sp[0]),  # feasibility_rate
                float(sp[1]),  # obj_improvement
                float(sp[2]),  # obj_roughness
                float(sp[3]),  # inc_range
                float(sp[4]),  # inc_trend
                float(sp[5]),  # penalty_trend
                float(sp[6]),  # n_evals_log
            ],
            dtype=float,
        )

    def _local_spike_score(self) -> Dict[str, float]:
        li = self._history_tail([self._safe_log_inc(value) for value in self.inc_hist])
        dx = np.diff(li, prepend=li[0])
        ddx = np.diff(dx, prepend=dx[0])
        baseline = float(np.median(li[:-1])) if li.size > 1 else float(li[-1])
        overshoot = max(li[-1] - baseline, 0.0)
        raw_score = 0.55 * abs(ddx[-1]) + 0.3 * max(dx[-1], 0.0) + 0.15 * overshoot

        # Standardize against statistics from BEFORE this sample so a spike cannot mask itself.
        prior_std = max(math.sqrt(max(self._score_m2 - self._score_m1 ** 2, 0.0)), self.score_std_floor)
        z_value = float(np.clip((raw_score - self._score_m1) / prior_std, -_Z_CLIP, _Z_CLIP))

        b = self._stat_beta
        self._score_m1 = (1.0 - b) * self._score_m1 + b * raw_score
        self._score_m2 = (1.0 - b) * self._score_m2 + b * raw_score ** 2
        return {"score": float(raw_score), "z": z_value, "volatility": float(prior_std)}

    def _embed(self, phi: np.ndarray) -> np.ndarray:
        """Project the raw 13-dim feature vector through the backbone to get the embedding z."""
        return self._backbone.embed(phi)

    @staticmethod
    def _trend_baseline(values: np.ndarray, horizon: int) -> tuple:
        """OLS line through equally spaced values; returns (value `horizon` steps past the last sample, residual std)."""
        m = int(values.size)
        if m < 3:
            return (float(values.mean()) if m else 0.0), (float(values.std()) if m > 1 else 0.0)
        t = np.arange(m, dtype=float)
        slope, intercept = np.polyfit(t, values, 1)
        resid = values - (slope * t + intercept)
        std = float(np.sqrt(np.sum(resid ** 2) / max(m - 2, 1)))
        return float(intercept + slope * (m - 1 + horizon)), std

    def _predictive_scores(self, current_y: float) -> Dict[str, float]:
        """Score the new observation y_k (relative log-inconsistency) and forecast y_{k+1}.

        The pair (z(k-1), y_k) is assimilated AFTER the residual is computed, so the residual
        is a genuine one-step-ahead innovation.
        """
        residual_z = 0.0

        current_phi = self._make_feature_vector()
        current_z = self._embed(current_phi)

        if self._last_phi is not None:
            noise = self._noise
            pred_mean = float(self._last_phi @ self._theta)
            param_var = float(max(self._last_phi @ self._cov @ self._last_phi, 0.0))
            pred_var = noise + param_var
            residual = current_y - pred_mean
            residual_z = float(np.clip(residual / math.sqrt(max(pred_var, 1e-12)), -_Z_CLIP, _Z_CLIP))

            self._theta, self._cov = _rls_step(self._theta, self._cov, self._last_phi, current_y, self.forgetting, noise)
            for head in self._ensemble:
                head.update(self._last_phi, current_y, noise)

            # Robust (clipped) noise tracking so a spike cannot inflate the noise floor it is judged against.
            sample = max(min(residual ** 2, 9.0 * pred_var) - param_var, 0.0)
            self._noise = float(max((1.0 - self._noise_beta) * noise + self._noise_beta * sample, self._noise_floor))

            # Lagged training pair (psi(k-1), y_k) for the periodic backbone re-fit.
            if self._prev_phi is not None:
                self._nlm_phi.append(self._prev_phi.copy())
                self._nlm_y.append(float(current_y))

        noise = self._noise
        next_mean = float(current_z @ self._theta)
        next_param_var = float(max(current_z @ self._cov @ current_z, 0.0))
        next_std = math.sqrt(max(noise + next_param_var, 1e-12))

        # Detrended baseline: the inconsistency normally decays, so a flat historical mean would bias the forecast score.
        li = self._history_tail([self._safe_log_inc(value) for value in self.inc_hist]) - float(self._ell0)
        base_next, base_std = self._trend_baseline(li[:-1], horizon=2)
        scale = max(base_std, 0.5 * math.sqrt(noise), 1e-3)
        forecast_z = float(np.clip((next_mean - base_next) / scale, -_Z_CLIP, _Z_CLIP))
        threshold = base_next + self.spike_sigma * scale
        forecast_prob = 1.0 - _normal_cdf((threshold - next_mean) / max(next_std, 1e-8))

        # Ensemble predictive distribution for y_{k+1}: epistemic = parameter + between-head variance, aleatoric = noise.
        head_means: List[float] = []
        head_param_vars: List[float] = []
        for head in self._ensemble:
            mean_h, var_h = head.predict(current_z)
            head_means.append(mean_h)
            head_param_vars.append(var_h)
        if head_means:
            epistemic_unc = float(np.mean(head_param_vars) + np.var(head_means))
        else:
            epistemic_unc = next_param_var
        aleatoric_unc = float(noise)

        self._cur_phi = current_phi
        self._prev_phi = current_phi
        self._last_phi = current_z
        return {
            "residual_z": float(residual_z),
            "forecast_z": float(forecast_z),
            "forecast_prob": float(np.clip(forecast_prob, 0.0, 1.0)),
            "epistemic_unc": float(np.clip(epistemic_unc, 0.0, 1e4)),
            "aleatoric_unc": float(np.clip(aleatoric_unc, 1e-8, 1e4)),
        }

    def _refit_backbone(self) -> None:
        """Re-train the backbone on the lagged buffer, then replay it to rebuild every RLS posterior.

        1. Train backbone + a temporary linear head on lagged pairs (psi(j), y_{j+1}) with a Huber loss
           and weight decay, so rare spikes do not dominate the fit.
        2. Discard the temporary head; freeze the backbone.
        3. Reset all posteriors and replay the buffer chronologically under the new embedding.
        """
        n = len(self._nlm_phi)
        if n < max(self.min_history, 8) or self._backbone_kind != "mlp":
            return

        phi_arr = np.array(list(self._nlm_phi), dtype=float)
        y_arr = np.array(list(self._nlm_y), dtype=float)
        phi_t = torch.tensor(phi_arr, dtype=torch.float32)
        y_t = torch.tensor(y_arr, dtype=torch.float32)

        # Re-seed the private generator to a step-anchored value so every refit is reproducible
        # and independent of the global torch RNG.
        self._torch_gen.manual_seed(self._base_seed + self._refit_count * 1000003)

        linear_head = nn.Linear(self._embedding_dim, 1, bias=False)
        with torch.no_grad():
            nn.init.kaiming_uniform_(linear_head.weight, a=math.sqrt(5), generator=self._torch_gen)
        optimizer = torch.optim.Adam(
            list(self._backbone.parameters()) + list(linear_head.parameters()),
            lr=3e-3,
            weight_decay=1e-3,
        )
        self._backbone.train()
        linear_head.train()
        for _ in range(80):
            optimizer.zero_grad()
            z = self._backbone(phi_t)
            loss = F.smooth_l1_loss(linear_head(z).squeeze(-1), y_t)
            loss.backward()
            optimizer.step()
        self._backbone.eval()

        self._theta = np.zeros(self._embedding_dim, dtype=float)
        self._cov = np.eye(self._embedding_dim, dtype=float) * _PRIOR_VAR
        for head in self._ensemble:
            head.reset()

        noise = self._noise
        for phi_raw, y in zip(phi_arr, y_arr):
            z = self._embed(phi_raw)
            self._theta, self._cov = _rls_step(self._theta, self._cov, z, float(y), self.forgetting, noise)
            for head in self._ensemble:
                head.update(z, float(y), noise)

        # The latest raw features pair with the NEXT observation under the new embedding.
        if self._cur_phi is not None:
            self._last_phi = self._embed(self._cur_phi)
        self._refit_count += 1

    def update(self, obj, inc, hmins: Optional[Sequence[float]] = None):
        obj = self._sanitize_objective(obj)
        inc = self._sanitize_value(inc, floor=1e-12, severe_cap=1e6)
        hmin_values = [
            self._sanitize_value(value, floor=0.0, severe_cap=self.hmin_threshold * 1e3) / self.hmin_scale
            for value in hmins
        ] if hmins is not None else []

        self.obj_hist.append(obj)
        self.inc_hist.append(inc)
        self.hmin_hist.append(hmin_values)
        if self._ell0 is None:
            self._ell0 = self._safe_log_inc(inc)

        # The detector learns from the very first sample; only the evidence is gated by maturity.
        current_y = self._safe_log_inc(inc) - self._ell0
        local_metrics = self._local_spike_score()
        predictive_metrics = self._predictive_scores(current_y)
        current_pressure = self._constraint_pressure(hmin_values)

        self._steps_since_refit += 1
        refit_due = self._refit_every > 0 and self._steps_since_refit >= self._refit_every

        if len(self.inc_hist) < self.min_history:
            self._store(
                score=0.0,
                prob=0.0,
                unc=1.0,
                alarm=0,
                forecast_prob=0.0,
                non_improvement_prob=0.0,
                objective_non_improvement_prob=0.0,
                constraint_non_improvement_prob=current_pressure,
                constraint_pressure=current_pressure,
            )
            self._store_diag(local_metrics["z"], predictive_metrics, 1.0)
            if refit_due:
                self._refit_backbone()
                self._steps_since_refit = 0
            return 0

        objective_score = self._objective_non_improvement_score()
        constraint_score = self._constraint_non_improvement_score(current_pressure)
        event_score = float(np.clip(0.55 * objective_score + 0.45 * constraint_score, 0.0, 1.0))

        # Maturity ramp: shrinks evidence toward the prior (prob~0.14) during warm-up
        maturity = float(np.clip(
            (len(self.inc_hist) - self.min_history) / max(self.window - self.min_history, 1),
            0.0, 1.0,
        )) if self.use_maturity else 1.0
        w_local, w_resid, w_fore = self.evidence_weights
        evidence = (
            w_local * local_metrics["z"]
            + w_resid * predictive_metrics["residual_z"]
            + w_fore * predictive_metrics["forecast_z"]
        )
        evidence = evidence * maturity - self.evidence_offset
        prob = float(np.clip(_sigmoid(evidence), 0.0, 1.0))
        forecast_prob = predictive_metrics["forecast_prob"]

        # Near the convergence tolerance the log-inconsistency fluctuates at solver-noise level; do not flag it.
        in_noise_floor = self.tol_floor > 0.0 and inc < self.tol_floor * self.tol_floor_mult
        if in_noise_floor:
            prob = 0.0
            forecast_prob = 0.0

        # unc: 50% signal-based + 50% epistemic fraction (high early, decays as the posteriors tighten)
        signal_unc = 1.0 / (1.0 + abs(local_metrics["z"]) + abs(predictive_metrics["residual_z"]))
        epistemic_unc = predictive_metrics.get("epistemic_unc", 0.0)
        aleatoric_unc = predictive_metrics.get("aleatoric_unc", 1.0)
        epistemic_frac = epistemic_unc / (epistemic_unc + aleatoric_unc + 1e-8)
        unc = float(np.clip(0.5 * signal_unc + 0.5 * epistemic_frac, 0.05, 1.0))

        spike_weight = max(prob, forecast_prob)
        self._assoc_alpha += spike_weight * event_score
        self._assoc_beta += spike_weight * (1.0 - event_score)
        self._obj_assoc_alpha += spike_weight * objective_score
        self._obj_assoc_beta += spike_weight * (1.0 - objective_score)
        self._con_assoc_alpha += spike_weight * constraint_score
        self._con_assoc_beta += spike_weight * (1.0 - constraint_score)

        assoc_prob = self._assoc_alpha / (self._assoc_alpha + self._assoc_beta)
        obj_assoc_prob = self._obj_assoc_alpha / (self._obj_assoc_alpha + self._obj_assoc_beta)
        con_assoc_prob = self._con_assoc_alpha / (self._con_assoc_alpha + self._con_assoc_beta)

        non_improvement_prob = float(np.clip(0.55 * assoc_prob + 0.3 * event_score + 0.15 * prob, 0.0, 1.0))
        objective_non_improvement_prob = float(np.clip(0.65 * obj_assoc_prob + 0.35 * objective_score, 0.0, 1.0))
        constraint_non_improvement_prob = float(np.clip(0.65 * con_assoc_prob + 0.35 * constraint_score, 0.0, 1.0))

        log_inc = np.asarray([self._safe_log_inc(value) for value in self.inc_hist], dtype=float)
        dx_all = np.diff(log_inc, prepend=log_inc[0])
        ddx_all = np.diff(dx_all, prepend=dx_all[0])
        geom_score = float(0.6 * abs(ddx_all[-1]) + 0.4 * abs(dx_all[-1]))

        past_geom_scores = np.asarray(0.6 * np.abs(ddx_all[:-1]) + 0.4 * np.abs(dx_all[:-1]), dtype=float)
        recent_scores = past_geom_scores[-24:] if past_geom_scores.size > 0 else np.asarray([], dtype=float)
        if recent_scores.size >= 5:
            score_mean = float(np.mean(recent_scores))
            score_std = float(np.std(recent_scores) + 1e-8)
            noise_level = float(np.median(recent_scores))
        else:
            score_mean = 0.0
            score_std = 1e-8
            noise_level = 0.0

        obj_trend = 0.0
        if len(self.obj_hist) >= 4:
            recent_obj = _signed_log1p(np.asarray(self.obj_hist[-(self.window - 1):], dtype=float))
            obj_trend = float(np.mean(np.diff(recent_obj)))

        adaptive_z_thr = 2.0 + 45.0 * min(noise_level, 0.06)
        if obj_trend > 3e-4:
            adaptive_z_thr -= 0.4
        elif obj_trend < -3e-4:
            adaptive_z_thr += 0.8

        z_geom = (geom_score - score_mean) / score_std
        score_gate = geom_score > max(0.04, score_mean + 0.15 * score_std)
        relative_jump = 1.0
        if len(self.inc_hist) >= 2:
            prev_inc = max(float(self.inc_hist[-2]), 1e-12)
            relative_jump = float(self.inc_hist[-1]) / prev_inc

        relative_spike = bool(
            relative_jump >= 3.0
            and geom_score > max(0.03, score_mean + 0.05 * score_std)
            and forecast_prob >= 0.25
        )
        spike_detected = bool(((z_geom > adaptive_z_thr) and score_gate) or relative_spike)
        if in_noise_floor:
            spike_detected = False

        self._event_buffer.append(int(spike_detected))
        alarm = int(spike_detected and sum(self._event_buffer) >= 2)
        if not self.use_geometric_alarm:
            alarm = 1

        risk = float(np.clip(max(non_improvement_prob, prob, forecast_prob), 0.0, 1.0))

        self._store(
            score=local_metrics["score"],
            prob=prob,
            unc=unc,
            alarm=alarm,
            forecast_prob=forecast_prob,
            non_improvement_prob=non_improvement_prob,
            objective_non_improvement_prob=objective_non_improvement_prob,
            constraint_non_improvement_prob=constraint_non_improvement_prob,
            constraint_pressure=current_pressure,
            risk=risk,
        )
        self._store_diag(local_metrics["z"], predictive_metrics, epistemic_frac)

        if refit_due:
            self._refit_backbone()
            self._steps_since_refit = 0

        return alarm

    def _store_diag(self, z_local: float, predictive_metrics: Dict[str, float], epistemic_frac: float) -> None:
        self.diag["z_local"].append(float(z_local))
        self.diag["z_resid"].append(float(predictive_metrics["residual_z"]))
        self.diag["z_fore"].append(float(predictive_metrics["forecast_z"]))
        self.diag["noise_var"].append(float(self._noise))
        self.diag["epi_frac"].append(float(epistemic_frac))

    def _store(
        self,
        score: float,
        prob: float,
        unc: float,
        alarm: int,
        forecast_prob: float,
        non_improvement_prob: float,
        objective_non_improvement_prob: float,
        constraint_non_improvement_prob: float,
        constraint_pressure: float,
        risk: Optional[float] = None,
    ):
        self.state["score"].append(score)
        self.state["prob"].append(prob)
        self.state["unc"].append(unc)
        self.state["risk"].append(prob if risk is None else risk)
        self.state["alarm"].append(alarm)
        self.state["forecast_prob"].append(forecast_prob)
        self.state["non_improvement_prob"].append(non_improvement_prob)
        self.state["objective_non_improvement_prob"].append(objective_non_improvement_prob)
        self.state["constraint_non_improvement_prob"].append(constraint_non_improvement_prob)
        self.state["constraint_pressure"].append(constraint_pressure)

    def decide(self, level_name: str = "L2") -> InferenceDecision:
        """
        Evaluate current inference trustworthiness at a given confidence level.

        Parameters
        ----------
        level_name : str
            One of "L1", "L2", "L3" (case-insensitive).

        Returns
        -------
        InferenceDecision
            Structured result indicating what signals to trust.
        """
        level = resolve_level(level_name)
        return evaluate_inference(self, level)


class HybridAnomalyDetector(BayesianOnlineSpikeDetector):
    pass


def latest_detector_metrics(detector: BayesianOnlineSpikeDetector) -> Dict[str, float]:
    """Most recent detector outputs in the form the coordinator consumes."""
    state = detector.state

    def last(key: str, default: float) -> float:
        return float(state[key][-1]) if state[key] else default

    return {
        "forecast_prob": last("forecast_prob", 0.0),
        "objective_non_improvement_prob": last("objective_non_improvement_prob", 0.0),
        "constraint_non_improvement_prob": last("constraint_non_improvement_prob", 0.0),
        "non_improvement_prob": last("non_improvement_prob", 0.0),
        "spike_prob": last("prob", 0.0),
        "detector_uncertainty": last("unc", 1.0),
        "constraint_pressure": last("constraint_pressure", 0.0),
    }
