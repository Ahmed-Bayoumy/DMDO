import random
import tempfile
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import torch

from DMDO.SP import SubProblem, _seed_all_rngs
from DMDO.coordinator import ADMM
from DMDO._detector import BayesianGRU, BayesianLSTM, HybridAnomalyDetector


def _write_minimal_coordination_history(path):
    rows = [
        "Time,Iteration #,Max. inconsistency,Objective",
        "09:00:00,0,0.01,10.0",
        "09:00:01,1,0.02,9.8",
    ]
    path.write_text("\n".join(rows), encoding="utf-8")


def _make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _assert_same_prediction(first, second):
    first_mean, first_std = first
    second_mean, second_std = second
    assert torch.allclose(first_mean, second_mean)
    assert torch.allclose(first_std, second_std)


def _make_synthetic_series(length: int = 30):
    obj = []
    inc = []
    hmins = []

    for step in range(length):
        obj.append(100.0 + 0.08 * step + 0.15 * np.sin(step / 3.0))

        inc_value = 1e-3 * (1.0 + 0.04 * np.cos(step / 4.0))
        if step in {10, 20}:
            inc_value *= 8.0
        inc.append(inc_value)

        hmins.append([
            max(0.0, 7e-4 * (1.0 + 0.03 * step)),
            max(0.0, 5e-4 * (1.0 + 0.02 * step)),
        ])

    return obj, inc, hmins


def _run_hybrid_detector(seed: int, model_type: str):
    detector = HybridAnomalyDetector(window=6, min_history=5, model_type=model_type, seed=seed)
    obj, inc, hmins = _make_synthetic_series()

    for obj_value, inc_value, hmin_values in zip(obj, inc, hmins):
        detector.update(obj_value, inc_value, hmins=hmin_values)

    return detector


def _assert_detector_state_equal(first, second):
    assert first._refit_count == second._refit_count
    assert first._steps_since_refit == second._steps_since_refit
    assert np.allclose(first._theta, second._theta)
    assert np.allclose(first._cov, second._cov)
    assert np.allclose(first._last_phi, second._last_phi)
    assert np.allclose(first._sp_features, second._sp_features)

    for key in first.state:
        first_value = np.asarray(first.state[key])
        second_value = np.asarray(second.state[key])
        if first_value.dtype.kind in {"f", "i"}:
            assert np.allclose(first_value, second_value)
        else:
            assert np.array_equal(first_value, second_value)

    assert np.allclose(
        first._backbone.embed(first._make_feature_vector()),
        second._backbone.embed(second._make_feature_vector()),
    )


def _assert_decisions_equal(first, second):
    for level_name in ("L1", "L2", "L3"):
        first_decision = asdict(first.decide(level_name=level_name))
        second_decision = asdict(second.decide(level_name=level_name))
        assert first_decision == second_decision


def test_bayesian_lstm_uses_local_rng_state():
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    x = torch.randn(2, 5, 4)

    first_model = BayesianLSTM(input_size=4, hidden_size=8, generator=_make_generator(777))
    first_prediction = first_model.predict_distribution(x, samples=5)

    torch.manual_seed(999)
    np.random.seed(999)
    random.seed(999)
    _ = torch.randn(64)
    _ = np.random.randn(64)
    _ = [random.random() for _ in range(8)]

    second_model = BayesianLSTM(input_size=4, hidden_size=8, generator=_make_generator(777))
    second_prediction = second_model.predict_distribution(x, samples=5)

    _assert_same_prediction(first_prediction, second_prediction)


def test_bayesian_gru_uses_local_rng_state():
    torch.manual_seed(456)
    np.random.seed(456)
    random.seed(456)
    x = torch.randn(2, 5, 4)

    first_model = BayesianGRU(input_size=4, hidden_size=8, generator=_make_generator(31415))
    first_prediction = first_model.predict_distribution(x, samples=5)

    torch.manual_seed(2024)
    np.random.seed(2024)
    random.seed(2024)
    _ = torch.randn(64)
    _ = np.random.randn(64)
    _ = [random.random() for _ in range(8)]

    second_model = BayesianGRU(input_size=4, hidden_size=8, generator=_make_generator(31415))
    second_prediction = second_model.predict_distribution(x, samples=5)

    _assert_same_prediction(first_prediction, second_prediction)


def test_hybrid_anomaly_detector_is_reproducible_for_same_seed():
    for model_type in ("LSTM", "GRU"):
        torch.manual_seed(321)
        np.random.seed(321)
        random.seed(321)
        first = _run_hybrid_detector(seed=2468, model_type=model_type)

        torch.manual_seed(9999)
        np.random.seed(9999)
        random.seed(9999)
        _ = torch.randn(128)
        _ = np.random.randn(128)
        _ = [random.random() for _ in range(16)]

        second = _run_hybrid_detector(seed=2468, model_type=model_type)

        _assert_detector_state_equal(first, second)
        _assert_decisions_equal(first, second)


def test_sp_seed_helper_reproduces_python_numpy_and_torch_rngs():
    _seed_all_rngs(13579)
    first_values = (
        random.random(),
        np.random.rand(4),
        torch.rand(4),
    )

    _seed_all_rngs(13579)
    second_values = (
        random.random(),
        np.random.rand(4),
        torch.rand(4),
    )

    assert first_values[0] == second_values[0]
    assert np.allclose(first_values[1], second_values[1])
    assert torch.allclose(first_values[2], second_values[2])


def test_subproblem_base_seed_uses_config_seed_or_fallback():
    subproblem = SubProblem.__new__(SubProblem)
    subproblem.RANDOM_SEED = 1234

    assert SubProblem._get_base_seed(subproblem, {"options": {"seed": "987"}}) == 987
    assert SubProblem._get_base_seed(subproblem, {"options": {"seed": None}}) == 1234
    assert SubProblem._get_base_seed(subproblem, {}) == 1234


def test_subproblem_wires_seed_into_hybrid_detector():
    coord = SimpleNamespace(is_AI_regularizer=True, ai_regularizer=None, refit_every=17)

    subproblem = SubProblem(
        nv=0,
        index=3,
        vars=[],
        resps=[],
        is_main=0,
        analysis=None,
        coordination=coord,
        opt=lambda *args, **kwargs: None,
        fmin_nop=np.inf,
        budget=1,
        display=False,
        conf={"options": {"seed": 2468}},
    )

    assert subproblem.RANDOM_SEED == 2468
    assert coord.ai_regularizer is not None
    assert coord.ai_regularizer._base_seed == 2471
    assert coord.ai_regularizer._refit_every == 17


def test_subproblem_detector_does_not_warm_start_from_history_by_default():
    with tempfile.TemporaryDirectory() as temp_dir:
        coord_history = Path(temp_dir) / "Coordination_history.out"
        _write_minimal_coordination_history(coord_history)

        coord = SimpleNamespace(is_AI_regularizer=True, ai_regularizer=None, refit_every=17)
        SubProblem(
            nv=0,
            index=1,
            vars=[],
            resps=[],
            is_main=0,
            analysis=None,
            coordination=coord,
            opt=lambda *args, **kwargs: None,
            fmin_nop=np.inf,
            budget=1,
            display=False,
            conf={"options": {"seed": 2468}},
            wd=temp_dir,
        )

        assert coord.ai_regularizer is not None
        assert len(coord.ai_regularizer.inc_hist) == 0
        assert len(coord.ai_regularizer.obj_hist) == 0


def test_subproblem_detector_warm_starts_from_history_when_enabled():
    with tempfile.TemporaryDirectory() as temp_dir:
        coord_history = Path(temp_dir) / "Coordination_history.out"
        _write_minimal_coordination_history(coord_history)

        coord = SimpleNamespace(is_AI_regularizer=True, ai_regularizer=None, refit_every=17)
        SubProblem(
            nv=0,
            index=1,
            vars=[],
            resps=[],
            is_main=0,
            analysis=None,
            coordination=coord,
            opt=lambda *args, **kwargs: None,
            fmin_nop=np.inf,
            budget=1,
            display=False,
            conf={"options": {"seed": 2468, "detector_warm_start_from_history": True}},
            wd=temp_dir,
        )

        assert coord.ai_regularizer is not None
        assert len(coord.ai_regularizer.inc_hist) > 0
        assert len(coord.ai_regularizer.obj_hist) > 0


def test_coordinator_attr_controls_detector_warm_start():
    coord = ADMM(
        nsp=2,
        beta=1.3,
        budget=10,
        index_of_master_SP=1,
        display=False,
        scaling=1.0,
        mode="serial",
        M_update_scheme=0,
        is_AI_regularizer=True,
        detector_warm_start_from_history=True,
    )

    assert coord.detector_warm_start_from_history is True

    subproblem = SubProblem(
        nv=0,
        index=1,
        vars=[],
        resps=[],
        is_main=0,
        analysis=None,
        coordination=coord,
        opt=lambda *args, **kwargs: None,
        fmin_nop=np.inf,
        budget=1,
        display=False,
        conf={"options": {"seed": 2468}},
    )

    assert subproblem.coord.ai_regularizer is not None