"""Reference and reduction checks for the methods (PROJECT_BRIEF §11)."""

import numpy as np
import pytest

from delayed_local_cp.experiments import runner
from delayed_local_cp.methods.baselines import build_method
from delayed_local_cp.methods.base import RoundContext, conformal_quantile
from delayed_local_cp.methods.online import weighted_quantile
from delayed_local_cp.streams.delay_queue import DelayQueue, LabelOracle
from delayed_local_cp.streams.stream import Stream
from delayed_local_cp.tracking.registry import Registry

from conftest import make_cfgs

ALL_METHODS = [
    {"type": "delayed_aci"}, {"type": "dtaci"}, {"type": "olcp"}, {"type": "static_lcp", "n_cal": 200},
    {"type": "ipw_aci"}, {"type": "weighted_cp", "refit_every": 100}, {"type": "dalo_cp"},
    {"type": "dalo_cp", "age_tau": 50, "arrival_ipw": True, "borrow": {"on": True}, "guard": {"on": True}},
]


def _toy(n=800, k=2, seed=0, delay=None):
    rng = np.random.default_rng(seed)
    regions = rng.integers(0, k, n)
    x = np.column_stack([(regions + rng.random(n)) / k])
    y = rng.normal(size=n)
    d = np.ones(n, dtype=np.int64) if delay is None else delay(rng, regions)
    return Stream(x, regions, y, d, np.zeros(n), k, {"score_scale": 0.67})


def _simulate(cfg, st, alpha=0.1):
    m = build_method(cfg, alpha, st.n_regions, st.meta)
    q = DelayQueue()
    o = LabelOracle(st.y, st.arrival)
    out = np.zeros(st.n)
    for t in range(st.n):
        for i, r in q.pop_due(t):
            m.on_arrival(i, st.x[i], r, st.yhat[i], o.reveal(i, t), t)
        out[t] = m.issue(t, st.x[t], int(st.regions[t]), st.yhat[t],
                         RoundContext(t, q.pending_counts(st.n_regions), q.n_pending))
        q.push(t, int(st.arrival[t]), int(st.regions[t]))
    return out


def test_weighted_quantile_hand_values():
    s, w = np.array([3.0, 1.0, 2.0]), np.ones(3)
    assert weighted_quantile(s, w, 0.5) == 2.0                  # cum 1,2,3; target 1.5
    assert weighted_quantile(s, w, 1.0) == 3.0
    assert weighted_quantile(s, w, 0.9, inf_mass=1.0) == np.inf # target 3.6 > 3
    assert weighted_quantile(s, np.array([0.0, 10.0, 0.0]), 0.5) == 1.0
    # unit weights with inf mass 1 equal the split-conformal rank
    z = np.arange(1.0, 20.0)
    assert weighted_quantile(z, np.ones(19), 0.9, 1.0) == conformal_quantile(z, 0.9)


def test_delayed_aci_with_unit_delay_equals_aci():
    st = _toy()
    a = _simulate({"type": "aci", "gamma": 0.01, "window": 100}, st)
    b = _simulate({"type": "delayed_aci", "gamma": 0.01, "window": 100}, st)
    np.testing.assert_allclose(a, b)


def test_dtaci_with_one_expert_equals_aci():
    st = _toy()
    a = _simulate({"type": "aci", "gamma": 0.01, "window": 100}, st)
    b = _simulate({"type": "dtaci", "gammas": [0.01], "window": 100}, st)
    np.testing.assert_allclose(a, b)


def test_project_option_gives_finite_thresholds_once_labels_exist():
    st = _toy()
    q = _simulate({"type": "regional_aci", "gamma": 0.2, "window": 50, "project": True}, st)
    assert np.isfinite(q[50:]).all()


def test_regional_aci_with_one_region_equals_aci():
    st = _toy(k=1)
    a = _simulate({"type": "aci", "gamma": 0.01, "window": 100}, st)
    b = _simulate({"type": "regional_aci", "gamma": 0.01, "window": 100}, st)
    np.testing.assert_allclose(a, b)


def test_dalo_all_off_one_region_equals_split_cp():
    st = _toy(k=1)
    a = _simulate({"type": "split_cp", "window": 100}, st)
    b = _simulate({"type": "dalo_cp", "eta_rel": 0.0, "window": 100}, st)
    finite = np.isfinite(a)   # DALO's base uses the largest score where split CP is still infinite (cold start)
    assert finite.sum() > 700
    np.testing.assert_allclose(a[finite], b[finite])
    assert np.isfinite(b[1:]).all()


def test_dalo_correction_matches_telescoping_identity():
    """With eta > 0: c_r equals eta * sum(err - alpha) over arrived labels of r."""
    st = _toy(delay=lambda rng, r: 1 + rng.integers(0, 30, len(r)))
    cfg = {"type": "dalo_cp", "eta_rel": 0.05, "window": 100}
    m = build_method(cfg, 0.1, st.n_regions, st.meta)
    q = DelayQueue()
    errs = [[], []]
    qs = {}
    for t in range(st.n):
        for i, r in q.pop_due(t):
            s = abs(st.y[i] - st.yhat[i])
            errs[r].append(float(s > qs[i]))
            m.on_arrival(i, st.x[i], r, st.yhat[i], float(st.y[i]), t)
        qs[t] = m.issue(t, st.x[t], int(st.regions[t]), st.yhat[t],
                        RoundContext(t, q.pending_counts(2), q.n_pending))
        q.push(t, int(st.arrival[t]), int(st.regions[t]))
    for r in (0, 1):
        np.testing.assert_allclose(m.c[r], m.eta * (np.sum(errs[r]) - 0.1 * len(errs[r])))


def test_olcp_bandwidth_formula():
    m = build_method({"type": "olcp", "window": 200}, 0.1, 1, {})
    assert m._h0(1) == pytest.approx((4 / 3) ** (1 / 5) * 200 ** (-1 / 5))


@pytest.mark.parametrize("mcfg", ALL_METHODS, ids=lambda c: c["type"] + ("_full" if "borrow" in c else ""))
def test_no_leakage_for_every_method(runs_dir, monkeypatch, mcfg):
    """Labels arriving at or after t0 cannot change any interval before t0."""
    base = make_cfgs()[0]
    cfg = dict(base, method=dict(mcfg, label=mcfg["type"]), run_id=f"leak_{mcfg['type']}_{len(mcfg)}")
    rng = np.random.default_rng(1)
    n = 900
    regions = rng.integers(0, 2, n)
    x = np.column_stack([(regions + rng.random(n)) / 2])
    y = rng.normal(size=n)
    d = np.where(regions == 1, 1 + rng.geometric(1 / 20, n), 1).astype(np.int64)
    st = Stream(x, regions, y, d, np.zeros(n), 2, {"score_scale": 0.67})
    reg = Registry(runner.RUNS_DIR / "registry.sqlite")
    monkeypatch.setattr(runner, "build_stream", lambda c: st)
    assert runner.run_single(cfg, reg) == "completed"
    q_ref = np.load(runner.run_dir_for(cfg) / "rounds.npz")["q"]
    t0 = 450
    y_bad = y.copy()
    y_bad[st.arrival >= t0] += 1e3
    bad = Stream(x, regions, y_bad, d, np.zeros(n), 2, st.meta)
    cfg2 = dict(cfg, run_id=cfg["run_id"] + "_p")
    monkeypatch.setattr(runner, "build_stream", lambda c: bad)
    assert runner.run_single(cfg2, reg) == "completed"
    q_bad = np.load(runner.run_dir_for(cfg2) / "rounds.npz")["q"]
    np.testing.assert_array_equal(q_ref[:t0], q_bad[:t0])


@pytest.mark.parametrize("mcfg", ALL_METHODS, ids=lambda c: c["type"] + ("_full" if "borrow" in c else ""))
def test_resume_equals_uninterrupted_for_every_method(runs_dir, registry, mcfg):
    base = make_cfgs()[0]
    full = dict(base, method=dict(mcfg, label=mcfg["type"]), run_id=f"res_{mcfg['type']}_{len(mcfg)}")
    cut = dict(full, run_id=full["run_id"] + "_cut")
    assert runner.run_single(full, registry) == "completed"
    assert runner.run_single(cut, registry, stop_after=613) == "interrupted"
    assert runner.run_single(cut, registry) == "completed"
    a = np.load(runner.run_dir_for(full) / "rounds.npz")["q"]
    b = np.load(runner.run_dir_for(cut) / "rounds.npz")["q"]
    np.testing.assert_array_equal(a, b)


def test_cqr_transform_preserves_coverage_and_width():
    from delayed_local_cp.streams.stream import apply_cqr

    st = _toy(n=200)
    lo, hi = np.full(200, -1.0) + 0.1 * st.x[:, 0], np.full(200, 1.0) + 0.5 * st.x[:, 0]
    c = apply_cqr(st, lo, hi, 0.5)
    q = np.full(200, float(((hi - lo) / 2).max()) + 0.2)   # q' = C + 0.2, i.e. CQR margin 0.2
    m, h = (hi + lo) / 2, (hi - lo) / 2
    np.testing.assert_array_equal(np.abs(c.y) <= q, np.abs(st.y - m) <= h + 0.2)
    np.testing.assert_allclose(q + c.meta["width_offset"], h + 0.2)


def test_weighted_cp_ratio_detects_covariate_change():
    m = build_method({"type": "weighted_cp"}, 0.1, 1, {})
    rng = np.random.default_rng(0)
    for v in rng.normal(0, 1, (300, 1)):
        m.xs.append(v)
        m.scores.append(1.0)
    for v in rng.normal(2, 1, (300, 1)):
        m.recent.append(v)
    m._refit()
    r = m._ratio(np.array([[-2.0], [4.0]]))
    assert r[0] < 1 < r[1]


def test_synthetic_cqr_stream_runs_end_to_end(runs_dir, registry):
    cfg = make_cfgs(dataset={"use": "synthetic_two_region", "n_rounds": 800, "n_train": 300, "model_iters": 20, "cqr": True}, shifts=[])[1]
    assert runner.run_single(cfg, registry) == "completed"
    assert "width_offset" in np.load(runner.run_dir_for(cfg) / "rounds.npz").files
