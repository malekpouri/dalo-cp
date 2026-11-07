"""M3 analysis: per-run metrics from rounds.npz and per-arm method tables (protocol: reports/decisions.md)."""

from __future__ import annotations

import csv
import glob
import json
import math
from collections import defaultdict

import numpy as np
import yaml

from ..utils.io import atomic_write_json
from ..utils.paths import RESULTS_DIR, RUNS_DIR

TOL = 0.05


def run_metrics(rd, burn_in: int) -> dict:
    d = np.load(rd / "rounds.npz")
    cfg = yaml.safe_load((rd / "config.yaml").read_text())
    alpha, w = float(cfg["alpha"]), int(cfg["eval"]["window"])
    y, yhat, q, reg, t = d["y"], d["yhat"], d["q"], d["region"], d["t"]
    cov = np.abs(y - yhat) <= q
    keep = t >= burn_in
    k = int(cfg["dataset"].get("n_regions", 3))
    def worst_for(win: int) -> float:
        best = math.inf
        for r in range(k):
            c = cov[keep & (reg == r)].astype(float)
            m = len(c) // win
            if m:
                best = min(best, float(c[:m * win].reshape(m, win).mean(axis=1).min()))
        return best

    worst = worst_for(w)
    finite = np.isfinite(q[keep])
    # An infinite interval is scored as the narrowest interval that also covers every round of this stream, with
    # half-width max|y - yhat|. Coverage is unchanged and unbounded intervals can no longer hide from the width.
    cap = float(np.max(np.abs(y - yhat)))
    qc = np.minimum(q[keep], cap)
    miss = np.maximum(np.abs(y[keep] - yhat[keep]) - qc, 0.0)
    # CQR streams: true half-width is q + width_offset (streams/stream.py); floor at 0
    half = np.maximum(qc + d["width_offset"][keep], 0.0) if "width_offset" in d.files else qc
    out = {"marginal": float(cov[keep].mean()), "worst": worst, "worst_50": worst_for(50), "worst_200": worst_for(200), "width": float(2 * half.mean()),
           "median_width": float(2 * np.median(half)),
           "interval_score": float(np.mean(2 * half + (2 / alpha) * miss)), "frac_inf": float(1 - finite.mean())}
    # Recovery: rounds from the shift until rolling coverage over w post-shift region rounds first reaches
    # 1 - alpha - TOL; maximum over shifted regions; censored at the end of the stream.
    shifts = cfg.get("shifts") or []
    if shifts:
        s = shifts[0]
        t0 = int(s["start"])
        if "size_per_region" in s:
            regs = [r for r, v in enumerate(s["size_per_region"]) if v != 0]
        else:
            regs = list(range(k)) if s.get("regions", "all") == "all" else list(s["regions"])
        rec = []
        for r in regs:
            idx = np.flatnonzero((reg == r) & (t >= t0))
            censor = float(len(t) - t0)
            if len(idx) < w:
                rec.append(censor)
                continue
            c = np.concatenate([[0.0], np.cumsum(cov[idx].astype(float))])
            roll = (c[w:] - c[:-w]) / w
            hit = np.flatnonzero(roll >= 1 - alpha - TOL - 1e-12)
            rec.append(float(idx[hit[0] + w - 1] - t0) if len(hit) else censor)
        out["recovery"] = max(rec)
    out["label"] = cfg["method"]["label"]
    out["seed"] = int(cfg["seed"])
    out["segment"] = cfg["dataset"].get("segment")
    return out


def table(prefix: str, burn_in: int) -> list[dict]:
    per = defaultdict(list)
    for f in glob.glob(str(RUNS_DIR / f"{prefix}__*" / "summary.json")):
        rd = RUNS_DIR / f.split("/")[-2]
        m = run_metrics(rd, burn_in)
        per[m["label"]].append(m)
    rows = []
    for label, runs in per.items():
        row = {"label": label, "n": len(runs)}
        for key in ("marginal", "worst", "width", "median_width", "interval_score", "frac_inf", "recovery"):
            vals = [r[key] for r in runs if key in r]
            if vals:
                row[key] = float(np.mean(vals))
                row[key + "_se"] = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        rows.append(row)
    return sorted(rows, key=lambda r: -r.get("worst", 0))


def main(name: str) -> int:
    from ..experiments.sweep import load_experiment
    from ..utils.paths import CONFIGS_DIR

    files = sorted(glob.glob(str(CONFIGS_DIR / "experiments" / f"{name}*.yaml")))
    res = {}
    for f in files:
        exp = load_experiment(f)
        burn = int(exp.get("eval", {}).get("burn_in", 500))
        rows = table(exp["name"], burn)
        res[exp["name"]] = rows
        print(f"\n== {exp['name']} (burn-in {burn})")
        print(f"{'label':30s} {'n':>4s} {'marg':>6s} {'worst':>6s} {'se':>5s} {'width':>6s} {'medw':>6s} {'iscore':>7s} {'inf':>5s} {'recov':>6s}")
        for r in rows:
            print(f"{r['label']:30s} {r['n']:4d} {r.get('marginal', math.nan):6.3f} {r.get('worst', math.nan):6.3f} "
                  f"{r.get('worst_se', math.nan):5.3f} {r.get('width', math.nan):6.3f} {r.get('median_width', math.nan):6.3f} "
                  f"{r.get('interval_score', math.nan):7.3f} {r.get('frac_inf', math.nan):5.3f} {r.get('recovery', math.nan):6.0f}")
    atomic_write_json(RESULTS_DIR / "summaries" / f"{name}.json", res)
    return 0
