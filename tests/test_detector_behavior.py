import numpy as np
import pytest

from DMDO.ai.decisor import resolve_level
from DMDO.ai.detector import HybridAnomalyDetector, _rls_step, _signed_log1p


def _stream(n=40, spike_at=(20, 21)):
    q = 1e-1 * np.exp(-0.12 * np.arange(n))
    for k in spike_at:
        q[k] *= 8.0
    f = 10.0 - 0.05 * np.arange(n)
    return q, f


def test_rls_without_forgetting_equals_batch_bayesian_posterior():
    rng = np.random.default_rng(0)
    d, n, noise, prior = 4, 30, 0.3, 25.0
    Z = rng.normal(size=(n, d))
    y = Z @ np.array([1.0, -2.0, 0.5, 0.0]) + rng.normal(scale=np.sqrt(noise), size=n)

    theta, cov = np.zeros(d), np.eye(d) * prior
    for z, yk in zip(Z, y):
        theta, cov = _rls_step(theta, cov, z, yk, forgetting=1.0, noise=noise, trace_cap=1e12)

    precision = np.eye(d) / prior + Z.T @ Z / noise
    cov_batch = np.linalg.inv(precision)
    theta_batch = cov_batch @ (Z.T @ y / noise)
    assert np.allclose(cov, cov_batch, atol=1e-8)
    assert np.allclose(theta, theta_batch, atol=1e-8)


def test_unknown_policy_name_is_rejected():
    with pytest.raises(ValueError):
        resolve_level("L4_Aggressive")
    det = HybridAnomalyDetector(seed=1)
    with pytest.raises(ValueError):
        det.decide(level_name="nonsense")
    assert resolve_level("L1").name == "L1_Conservative"


def test_negative_objective_is_preserved():
    det = HybridAnomalyDetector(seed=1)
    q, _ = _stream()
    for k, qk in enumerate(q):
        det.update(obj=-3000.0 + 5.0 * k, inc=qk)
    assert det.obj_hist[0] == pytest.approx(-3000.0)
    assert np.all(np.isfinite(det.state["prob"]))
    assert _signed_log1p(-3.0) == pytest.approx(-np.log(4.0))


def test_refit_buffer_is_lagged_and_does_not_leak_target():
    det = HybridAnomalyDetector(seed=1, refit_every=10)
    q, f = _stream()
    for qk, fk in zip(q, f):
        det.update(obj=fk, inc=qk)
    phi = np.asarray(list(det._nlm_phi))
    y = np.asarray(list(det._nlm_y))
    # feature index 1 is the relative level of the SAME iteration as the feature vector, i.e. y shifted by one step
    assert len(phi) == len(y) > 5
    assert not np.allclose(phi[:, 1], y)
    assert np.allclose(phi[1:, 1], y[:-1], atol=1e-9)


def test_scale_invariance_of_detector_scores():
    q, f = _stream()
    a = HybridAnomalyDetector(seed=3)
    b = HybridAnomalyDetector(seed=3)
    for qk, fk in zip(q, f):
        a.update(obj=fk, inc=qk)
        b.update(obj=fk, inc=1000.0 * qk)
    assert np.allclose(a.state["prob"], b.state["prob"], atol=1e-6)
    assert np.allclose(a.state["forecast_prob"], b.state["forecast_prob"], atol=1e-6)


def test_noise_floor_suppresses_detection():
    det = HybridAnomalyDetector(seed=1, tol_floor=1.0, tol_floor_mult=2.0)
    q, f = _stream()
    for qk, fk in zip(q, f):
        det.update(obj=fk, inc=qk)  # q_max is always below 2 * tol_floor
    assert max(det.state["prob"]) == 0.0
    assert max(det.state["forecast_prob"]) == 0.0
    assert sum(det.state["alarm"]) == 0


def test_epistemic_fraction_starts_high_and_decays():
    det = HybridAnomalyDetector(seed=2)
    q, f = _stream(n=60, spike_at=())
    for qk, fk in zip(q, f):
        det.update(obj=fk, inc=qk)
    frac = np.asarray(det.diag["epi_frac"])
    assert frac[:3].mean() > 0.9
    assert frac[-5:].mean() < frac[:3].mean()


def test_ablation_options_run():
    q, f = _stream()
    for opts in ({"backbone": "identity"}, {"ensemble_forgetting": [0.97]}, {"use_sp_features": False},
                 {"use_maturity": False}, {"use_geometric_alarm": False}):
        det = HybridAnomalyDetector(seed=1, **opts)
        for qk, fk in zip(q, f):
            det.update(obj=fk, inc=qk)
        assert len(det.state["prob"]) == len(q)
