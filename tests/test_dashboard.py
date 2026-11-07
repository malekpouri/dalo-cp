"""The dashboard renders without exceptions, empty and with runs present."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from delayed_local_cp.experiments import sweep

from conftest import make_cfgs

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


def _render():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_dashboard_renders_with_runs():
    _render()
    sweep.execute(make_cfgs(name="dash"), jobs=1)  # uses the session test runs dir
    at = _render()
    assert any("DALO-CP" in t.value for t in at.title)
