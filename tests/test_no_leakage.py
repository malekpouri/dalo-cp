"""Future or pending labels must not affect any issued interval (PROJECT_BRIEF §8, §11)."""

import numpy as np
import pytest

from delayed_local_cp.data.registry import build_stream
from delayed_local_cp.experiments import runner

from conftest import make_cfgs


def _q(cfg, stream, monkeypatch):
    monkeypatch.setattr(runner, "build_stream", lambda c: stream)
    assert runner.run_single(cfg, registry=_reg()) == "completed"
    return np.load(runner.run_dir_for(cfg) / "rounds.npz")["q"]


def _reg():
    from delayed_local_cp.tracking.registry import Registry
    return Registry(runner.RUNS_DIR / "registry.sqlite")


@pytest.mark.parametrize("method_idx", [0, 1, 2])
@pytest.mark.parametrize("t0", [200, 701, 1100])
def test_labels_arriving_at_or_after_t0_cannot_change_intervals_before_t0(runs_dir, monkeypatch, method_idx, t0):
    cfg = make_cfgs()[method_idx]
    stream = build_stream(cfg)
    q_ref = _q(cfg, stream, monkeypatch)

    y_bad = stream.y.copy()
    future = stream.arrival >= t0
    y_bad[future] += 1e3   # wildly wrong labels for everything not yet arrived at round t0
    bad = type(stream)(stream.x, stream.regions, y_bad, stream.delay, stream.yhat, stream.n_regions, stream.meta)
    cfg2 = dict(cfg, run_id=cfg["run_id"] + "_perturbed")
    q_bad = _q(cfg2, bad, monkeypatch)

    np.testing.assert_array_equal(q_ref[:t0], q_bad[:t0])
    assert not np.array_equal(q_ref[t0:], q_bad[t0:]), "perturbation had no effect; test is vacuous"


def test_own_label_never_used_for_own_interval(runs_dir, monkeypatch):
    cfg = make_cfgs()[2]
    stream = build_stream(cfg)
    q_ref = _q(cfg, stream, monkeypatch)
    for t in (10, 500, 1400):
        y_bad = stream.y.copy()
        y_bad[t] += 1e3
        bad = type(stream)(stream.x, stream.regions, y_bad, stream.delay, stream.yhat, stream.n_regions, stream.meta)
        c = dict(cfg, run_id=f"{cfg['run_id']}_own{t}")
        q_bad = _q(c, bad, monkeypatch)
        np.testing.assert_array_equal(q_ref[: t + 1], q_bad[: t + 1])
