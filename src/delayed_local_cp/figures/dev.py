"""Development previews (not paper figures). Written to paper/figures/dev/ as PNG."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..evaluation.metrics import covered, rolling_region_coverage
from ..tracking.registry import Registry
from ..utils.paths import PAPER_DIR, RUNS_DIR
from . import style


def smoke_preview() -> Path | None:
    import matplotlib.pyplot as plt
    import yaml

    reg = Registry()
    done = [r for r in reg.list(status="completed") if r["experiment"].startswith("smoke")]
    if not done:
        return None
    latest = done[-1]["experiment"]
    runs = [r for r in done if r["experiment"] == latest]
    style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(style.DOUBLE_COL_IN, 2.2), sharey=True)
    for row in runs:
        rd = RUNS_DIR / row["run_id"]
        cfg = yaml.safe_load((rd / "config.yaml").read_text())
        if cfg["seed"] != min(r["seed"] for r in runs):
            continue
        d = np.load(rd / "rounds.npz")
        cov = covered(d["y"], d["yhat"] - d["q"], d["yhat"] + d["q"])
        k = int(d["region"].max()) + 1
        roll = rolling_region_coverage(cov, d["region"], k, int(cfg["eval"]["window"]))
        st = style.method_style(row["method"])
        for r, ax in zip(range(k), axes):
            m = d["region"] == r
            ax.plot(d["t"][m], roll[m], color=st["color"], ls=st["ls"], label=st["label"])
    alpha = cfg["alpha"]
    for r, ax in enumerate(axes):
        ax.axhline(1 - alpha, color="0.5", lw=0.8)
        for s in cfg.get("shifts", []):
            ax.axvline(s["start"], color="0.7", lw=0.8, ls=":")
        ax.set_title(f"Region {r}")
        ax.set_xlabel("Round")
    axes[0].set_ylabel("Rolling coverage")
    axes[0].legend(frameon=False, loc="lower left")
    out = PAPER_DIR / "figures" / "dev" / f"{latest}_rolling_coverage.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out
