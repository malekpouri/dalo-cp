import json

import numpy as np

from delayed_local_cp.experiments import runner, sweep
from delayed_local_cp.tracking import signals
from delayed_local_cp.utils.io import read_jsonl

from conftest import make_cfgs


def _outputs(cfg):
    rd = runner.run_dir_for(cfg)
    d = np.load(rd / "rounds.npz")
    s = json.loads((rd / "summary.json").read_text())
    s.pop("run_id")
    m = [{k: v for k, v in r.items() if k != "wall"} for r in read_jsonl(rd / "metrics.jsonl")]
    return {k: d[k] for k in d.files}, s, m


def test_same_seed_same_outputs_different_seed_differs(runs_dir, registry):
    a, b = make_cfgs()[1], dict(make_cfgs()[1])
    b["run_id"] = a["run_id"] + "_again"
    assert runner.run_single(a, registry) == "completed"
    assert runner.run_single(b, registry) == "completed"
    ra, sa, ma = _outputs(a)
    rb, sb, mb = _outputs(b)
    for k in ra:
        np.testing.assert_array_equal(ra[k], rb[k])
    assert sa == sb and ma == mb
    c = make_cfgs(seeds=[1])[1]
    runner.run_single(c, registry)
    rc, _, _ = _outputs(c)
    assert not np.array_equal(ra["y"], rc["y"])


def test_resumed_run_equals_uninterrupted_run(runs_dir, registry):
    for idx in range(3):
        full = make_cfgs()[idx]
        cut = dict(full, run_id=full["run_id"] + "_cut")
        assert runner.run_single(full, registry) == "completed"
        # interrupt twice at awkward points (not checkpoint multiples), then finish
        assert runner.run_single(cut, registry, stop_after=437) == "interrupted"
        assert (runner.run_dir_for(cut) / "checkpoint.pkl").exists()
        assert not (runner.run_dir_for(cut) / "summary.json").exists()
        assert registry.get(cut["run_id"])["status"] == "interrupted"
        assert runner.run_single(cut, registry, stop_after=1003) == "interrupted"
        assert runner.run_single(cut, registry) == "completed"
        assert not (runner.run_dir_for(cut) / "checkpoint.pkl").exists()
        rf, sf, mf = _outputs(full)
        rc, sc, mc = _outputs(cut)
        for k in rf:
            np.testing.assert_array_equal(rf[k], rc[k])
        assert sf == sc
        assert mf == mc


def test_signal_stop_checkpoints_and_resume_completes(runs_dir, registry):
    cfg = make_cfgs()[2]
    signals.request_stop()
    assert runner.run_single(cfg, registry) == "interrupted"
    signals.reset()
    assert runner.run_single(cfg, registry) == "completed"


def test_completed_runs_are_skipped_and_stale_runs_marked(runs_dir, registry):
    cfgs = make_cfgs()
    res = sweep.execute(cfgs[:1], jobs=1, registry=registry)
    assert list(res.values()) == ["completed"]
    mtime = (runner.run_dir_for(cfgs[0]) / "summary.json").stat().st_mtime
    sweep.execute(cfgs[:1], jobs=1, registry=registry)
    assert (runner.run_dir_for(cfgs[0]) / "summary.json").stat().st_mtime == mtime
    # simulate a crash: a 'running' entry with an old heartbeat
    registry.register("ghost", experiment="t", method="m", dataset="d", seed=0, config_hash="h",
                      git_commit="c", run_dir="x")
    registry.set_running("ghost", "c")
    assert registry.mark_stale(stale_after_s=-1) == ["ghost"]
    assert registry.get("ghost")["status"] == "interrupted"


def test_run_ids_unique_and_oracle_label():
    cfgs = make_cfgs(methods=[{"use": "aci"}, {"use": "aci", "oracle_no_delay": True}], seeds=[0, 1])
    ids = [c["run_id"] for c in cfgs]
    assert len(set(ids)) == 4
    assert {c["method"]["label"] for c in cfgs} == {"aci", "aci_oracle"}
