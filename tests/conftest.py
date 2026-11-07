import os
import tempfile

# Keep test runs out of the real runs/ directory (set before the package is imported).
os.environ.setdefault("DALOCP_RUNS", tempfile.mkdtemp(prefix="dalocp-test-runs-"))

import pytest  # noqa: E402

from delayed_local_cp.experiments import runner, sweep  # noqa: E402
from delayed_local_cp.tracking import signals  # noqa: E402
from delayed_local_cp.tracking.registry import Registry  # noqa: E402

SMALL_EXP = {
    "name": "t",
    "alpha": 0.1,
    "dataset": {"use": "synthetic_two_region", "n_rounds": 1500, "n_train": 300, "model_iters": 20},
    "delay": {"use": "two_region_geometric", "per_region": [1, 25]},
    "shifts": [{"kind": "mean", "regions": [1], "start": 700, "size": 0.8, "profile": "abrupt"}],
    "eval": {"window": 50},
    "runtime": {"log_every": 50, "checkpoint_every_rounds": 300},
    "methods": [{"use": "split_cp"}, {"use": "aci"}, {"use": "regional_aci"}],
    "seeds": [0],
}


def make_cfgs(**over):
    exp = dict(SMALL_EXP)
    exp.update(over)
    return sweep.expand(exp)


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    d = tmp_path / "runs"
    monkeypatch.setattr(runner, "RUNS_DIR", d)
    signals.reset()
    yield d
    signals.reset()


@pytest.fixture
def registry(tmp_path):
    return Registry(tmp_path / "registry.sqlite")
