"""M2 feasibility analysis (protocol in reports/decisions.md, "M2 protocol").

Reads completed runs of the experiment, computes per-seed quantities from
rounds.npz, and evaluates gate criteria G1-G3 without touching the simulation.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import yaml
from scipy import stats

from ..tracking.registry import Registry
from ..utils.io import atomic_write_json, atomic_write_text
from ..utils.paths import PAPER_DIR, RESULTS_DIR, RUNS_DIR

POST_WINDOW = 1000
SLOW_REGION = 1


def _mean_ci(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = float(x.mean())
    if n < 2:
        return {"mean": m, "lo": float("nan"), "hi": float("nan"), "n": n}
    h = float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))
    return {"mean": m, "lo": m - h, "hi": m + h, "half": h, "n": n}


def _per_run(rd, metric: str = "post_slow"):
    """Returns (marginal, gate metric, config). ``metric``: 'post_slow' (M2) or 'worst' (M2b)."""
    if metric == "worst":
        s = json.loads((rd / "summary.json").read_text())
        cfg = yaml.safe_load((rd / "config.yaml").read_text())
        return float(s["marginal_coverage"]), float(s["worst_region_window_coverage"]), cfg
    d = np.load(rd / "rounds.npz")
    cov = np.abs(d["y"] - d["yhat"]) <= d["q"]
    cfg = yaml.safe_load((rd / "config.yaml").read_text())
    t0 = int(cfg["shifts"][0]["start"])
    t = d["t"]
    m = (d["region"] == SLOW_REGION) & (t >= t0) & (t < t0 + POST_WINDOW)
    return float(cov.mean()), float(cov[m].mean()), cfg


def analyze(experiment: str = "m2_feasibility", metric: str = "post_slow") -> dict:
    reg = Registry()
    rows = [r for r in reg.list(experiment=experiment) if r["status"] == "completed"]
    data = defaultdict(dict)   # (method, m) -> seed -> (marginal, post_slow)
    for r in rows:
        rd = RUNS_DIR / r["run_id"]
        marg, post, cfg = _per_run(rd, metric)
        m = int(cfg["grid_point"]["delay.per_region"][1])
        data[(r["method"], m)][int(cfg["seed"])] = (marg, post)

    ms = sorted({m for _, m in data})
    methods = sorted({k for k, _ in data})
    out = {"experiment": experiment, "metric": metric, "post_window": POST_WINDOW, "slow_region": SLOW_REGION,
           "n_runs": len(rows), "delay_means": ms, "methods": methods, "by_method": {}, "gaps": {}}

    for meth in methods:
        out["by_method"][meth] = {}
        for m in ms:
            v = data.get((meth, m), {})
            if not v:
                continue
            arr = np.array(list(v.values()))
            out["by_method"][meth][m] = {"marginal": _mean_ci(arr[:, 0]), "post_slow": _mean_ci(arr[:, 1])}

    def paired_gap(base: str, oracle: str, m: int):
        a, b = data.get((base, m), {}), data.get((oracle, m), {})
        seeds = sorted(set(a) & set(b))
        return np.array([b[s][1] - a[s][1] for s in seeds]), seeds

    for base in ("aci", "regional_aci"):
        g = {}
        per_seed = {}
        for m in ms:
            gap, seeds = paired_gap(base, f"{base}_oracle", m)
            if len(gap):
                g[m] = _mean_ci(gap)
                per_seed[m] = dict(zip(seeds, gap))
        out["gaps"][base] = g
        if 5 in per_seed and 100 in per_seed:
            common = sorted(set(per_seed[5]) & set(per_seed[100]))
            out["gaps"][base + "_growth_100_minus_5"] = _mean_ci(
                np.array([per_seed[100][s] - per_seed[5][s] for s in common]))
    # ACI vs Regional ACI on the slow region post-shift (positive = regional better)
    cmp = {}
    for m in ms:
        a, b = data.get(("aci", m), {}), data.get(("regional_aci", m), {})
        seeds = sorted(set(a) & set(b))
        if seeds:
            cmp[m] = _mean_ci(np.array([b[s][1] - a[s][1] for s in seeds]))
    out["regional_minus_global_post_slow"] = cmp

    # Gate (representative baseline: global ACI)
    aci = out["by_method"].get("aci", {})
    g = out["gaps"].get("aci", {})
    g1 = all(0.88 <= aci[m]["marginal"]["mean"] <= 0.92 for m in ms if m in aci)
    g2 = 100 in g and g[100]["mean"] >= 0.05 and g[100]["lo"] > 0
    adj = []
    for a, b in zip(ms[:-1], ms[1:]):
        drop = g[a]["mean"] - g[b]["mean"]
        adj.append({"from": a, "to": b, "drop": drop, "tolerance": g[b].get("half", 0.0),
                    "ok": drop <= g[b].get("half", 0.0)})
    growth = out["gaps"].get("aci_growth_100_minus_5")
    g3 = all(x["ok"] for x in adj) and growth is not None and growth["lo"] > 0
    out["gate"] = {"G1_marginal_in_range": bool(g1), "G2_gap_at_max_ge_0.05": bool(g2),
                   "G3_gap_grows": bool(g3), "G3_adjacent_checks": adj, "pass": bool(g1 and g2 and g3)}
    return out


def write_outputs(res: dict) -> list[str]:
    paths = []
    p = RESULTS_DIR / "summaries" / f"{res['experiment']}.json"
    atomic_write_json(p, res)
    paths.append(str(p))
    lines = ["delay_mean,method,marginal_mean,marginal_lo,marginal_hi,post_slow_mean,post_slow_lo,post_slow_hi"]
    for meth, by_m in res["by_method"].items():
        for m, v in sorted(by_m.items()):
            mc, ps = v["marginal"], v["post_slow"]
            lines.append(f"{m},{meth},{mc['mean']:.4f},{mc['lo']:.4f},{mc['hi']:.4f},"
                         f"{ps['mean']:.4f},{ps['lo']:.4f},{ps['hi']:.4f}")
    lines.append("")
    lines.append("delay_mean,baseline,gap_mean,gap_lo,gap_hi")
    for base in ("aci", "regional_aci"):
        for m, v in sorted(res["gaps"][base].items()):
            lines.append(f"{m},{base},{v['mean']:.4f},{v['lo']:.4f},{v['hi']:.4f}")
    p = RESULTS_DIR / "tables" / f"{res['experiment']}.csv"
    atomic_write_text(p, "\n".join(lines) + "\n")
    paths.append(str(p))
    paths.append(str(_figure(res)))
    return paths


def _figure(res: dict):
    import matplotlib.pyplot as plt

    from ..figures import style

    style.apply()
    fig, ax = plt.subplots(figsize=(style.SINGLE_COL_IN, 2.4))
    for base in ("aci", "regional_aci"):
        g = res["gaps"][base]
        ms = sorted(g)
        mean = [g[m]["mean"] for m in ms]
        lo = [g[m]["lo"] for m in ms]
        hi = [g[m]["hi"] for m in ms]
        st = style.method_style(base)
        ax.plot(ms, mean, color=st["color"], ls=st["ls"], marker="o", ms=3, label=st["label"])
        ax.fill_between(ms, lo, hi, color=st["color"], alpha=0.2, lw=0)
    ax.axhline(0.05, color="0.5", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("Slow-region mean delay / fast-region delay")
    ax.set_ylabel("Worst region-window coverage\ngap vs no-delay oracle" if res.get("metric") == "worst"
                  else "Post-shift slow-region\ncoverage gap vs no-delay oracle")
    ax.legend(frameon=False)
    out = PAPER_DIR / "figures" / "dev" / f"{res['experiment']}_gap_vs_delay.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def main(experiment: str = "m2_feasibility", metric: str = "post_slow") -> dict:
    res = analyze(experiment, metric)
    paths = write_outputs(res)
    print(json.dumps(res["gate"], indent=2))
    for p in paths:
        print("wrote", p)
    return res
