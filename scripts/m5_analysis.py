"""M5 pre-registered analysis (reports/decisions.md, "M5 protocol").

Writes results/summaries/m5_results.json and results/tables/m5_main.csv. No threshold is set here that the protocol did not state.
"""
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from scipy import stats

from delayed_local_cp.evaluation.m3 import run_metrics
from delayed_local_cp.utils.paths import RESULTS_DIR, RUNS_DIR

DALO, ORACLE = "dalo_agei25_e0.05_b", "dalo_oracle"
HELD_OUT = ["a06_gradual", "a07_recurring", "a08_variance", "a09_covariate_and_mean", "a10_y_dependent_delay", "a11_heavy_tailed_delay", "a12_continuous_x_delay"]
SELECTION = ["a01_local_geometric", "a02_local_fixed", "a03_smooth_alternating", "a04_smooth_gradient", "a05_nonsmooth_fast_only"]
REQUIRED_INFO = ["a01_local_geometric", "a03_smooth_alternating", "a04_smooth_gradient", "elec2"]
FAM = ["regional_aci", "delayed_aci", "ipw_aci", "dtaci", "olcp", "aci", "split_cp", "static_lcp", "weighted_cp"]
B = 2000


def family(label):
    if label == DALO:
        return "dalo_cp"
    if "oracle" in label:  # no-delay references are not competitors
        return None
    for f in FAM:
        if label.startswith(f):
            return f
    return None


def _one(args):
    name, burn = args
    rd = RUNS_DIR / name
    m = run_metrics(rd, burn)
    return name, m


def load(design):
    prefix = f"m5_{design}"
    burn = 300 if "elec2" in design else 500  # TRACE protocol: burn-in 500
    dirs = [f.split("/")[-2] for f in glob.glob(str(RUNS_DIR / f"{prefix}__*" / "summary.json"))]
    with Pool(10) as pool:
        rows = pool.map(_one, [(d, burn) for d in dirs], chunksize=32)
    per = defaultdict(lambda: defaultdict(list))
    panel = "elec2" in design  # TRACE units are seeds: each seed is its own thinned stream
    for _, m in rows:
        per[m["label"]][m["segment"] if panel else m["seed"]].append(m)
    keys = ("marginal", "worst", "worst_50", "worst_200", "width", "median_width", "interval_score", "frac_inf", "recovery")
    return {l: {u: {k: float(np.mean([r[k] for r in rs])) for k in keys if k in rs[0]} for u, rs in units.items()} for l, units in per.items()}


def mean(d, k):
    return float(np.mean([v[k] for v in d.values()]))


def holm(p):
    out, top = {}, 0.0
    ks = sorted(p, key=p.get)
    for i, k in enumerate(ks):
        top = max(top, min(1.0, (len(ks) - i) * p[k]))
        out[k] = top
    return out


def near_valid(d, target):
    return target - 0.02 <= mean(d, "marginal") <= target + 0.02


def design_summary(D, target):
    bases = [l for l in D if family(l) not in (None, "dalo_cp") and near_valid(D[l], target)]
    units = sorted(D[DALO])
    res = {"table": {l: {k: mean(D[l], k) for k in next(iter(D[l].values()))} for l in D}, "near_valid_baselines": bases, "dalo_near_valid": near_valid(D[DALO], target)}
    if not bases:
        return res
    comp = max(bases, key=lambda l: mean(D[l], "worst"))
    pv = {}
    for b in bases:
        g = np.array([D[DALO][u]["worst"] - D[b][u]["worst"] for u in units])
        pv[b] = float(stats.ttest_1samp(g, 0, alternative="greater").pvalue) if np.any(g) else 1.0
    adj = holm(pv)
    g = np.array([D[DALO][u]["worst"] - D[comp][u]["worst"] for u in units])
    h = float(stats.t.ppf(0.975, len(g) - 1) * g.std(ddof=1) / np.sqrt(len(g)))
    res.update({"comparator": comp, "n_units": len(units), "gain": float(g.mean()), "gain_lo": float(g.mean() - h), "gain_hi": float(g.mean() + h), "p_holm": adj[comp],
                "width_ratio": mean(D[DALO], "width") / mean(D[comp], "width"), "gain_ge_0.03": bool(g.mean() >= 0.03 and adj[comp] < 0.05),
                "width_ok": bool(mean(D[DALO], "width") <= 1.10 * mean(D[comp], "width"))})
    if "recovery" in next(iter(D[DALO].values())):
        a, c = (np.array([D[l][u]["recovery"] for u in units]) for l in (DALO, comp))
        res["recovery_dalo"], res["recovery_comparator"] = float(a.mean()), float(c.mean())
        res["recovery_p"] = float(stats.wilcoxon(a, c, alternative="less").pvalue) if np.any(a != c) else 1.0
    return res


def regret_table(data, designs, target, rng=None):
    """Per family: regret in each design; a family that is not near-valid in a design has infinite regret there."""
    out = defaultdict(dict)
    for d in designs:
        Dd = data[d]
        means = {}
        for l, units in Dd.items():
            f = family(l)
            if f is None:
                continue
            ks = list(units)
            idx = ks if rng is None else [ks[i] for i in rng.integers(0, len(ks), len(ks))]
            means[f] = (float(np.mean([units[k]["worst"] for k in idx])), float(np.mean([units[k]["marginal"] for k in idx])))
        nv = {f: v[0] for f, v in means.items() if target - 0.02 <= v[1] <= target + 0.02}
        top = max(nv.values())
        for f, (w, _) in means.items():
            out[f][d] = (top - w) if f in nv else float("inf")
    return out


def maxreg(tab, f):
    return max(tab[f].values())


def main():
    if not all(glob.glob(str(RUNS_DIR / f"m5_{d}__*" / "summary.json")) for d in ("elec2", "cqr_elec2", "alpha05_elec2", "trace_p", "trace_d")):
        sys.exit("required sweeps are not complete")
    designs = SELECTION + HELD_OUT + ["elec2"]
    data = {d: load(d) for d in designs}
    res = {"designs": {d: design_summary(data[d], 0.9) for d in designs}}
    rec = {d: res["designs"][d]["recovery_p"] for d in designs if "recovery_p" in res["designs"][d]}
    for d, v in holm(rec).items():
        res["designs"][d]["recovery_p_holm"] = v
    all_d, new_d = designs, HELD_OUT
    robust = {}
    rng = np.random.default_rng(20261106)
    for tag, ds in (("held_out", new_d), ("all", all_d)):
        tab = regret_table(data, ds, 0.9)
        comp = sorted((f for f in tab if f != "dalo_cp"), key=lambda f: maxreg(tab, f))
        best = comp[0]
        diffs = []
        for _ in range(B):
            t = regret_table(data, ds, 0.9, rng)
            others = [maxreg(t, f) for f in t if f != "dalo_cp"]
            diffs.append(min(others) - maxreg(t, "dalo_cp"))
        # An infinite regret means "not near-valid in that replicate". Keep the sign, so DALO-CP failing counts against it; both infinite is no margin.
        diffs = np.nan_to_num(np.array(diffs), nan=0.0, posinf=1.0, neginf=-1.0)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        robust[tag] = {"max_regret": {f: maxreg(tab, f) for f in tab}, "mean_regret": {f: float(np.mean([v for v in tab[f].values() if np.isfinite(v)])) for f in tab},
                       "dalo_regret_by_design": tab["dalo_cp"], "best_competitor": best, "margin_lo": float(lo), "margin_hi": float(hi), "margin": float(maxreg(tab, best) - maxreg(tab, "dalo_cp"))}
    # Descriptive, not part of the gate: does one design drive the margin?
    loo = {}
    for drop in all_d:
        tab = regret_table(data, [d for d in all_d if d != drop], 0.9)
        loo[drop] = min(maxreg(tab, f) for f in tab if f != "dalo_cp") - maxreg(tab, "dalo_cp")
    robust["all"]["margin_leave_one_design_out"] = loo
    robust["competitor_regret_by_design"] = {f: regret_table(data, all_d, 0.9)[f] for f in ("regional_aci", "olcp")}
    w = []
    for d in all_d:
        s = res["designs"][d]
        if "width_ratio" in s:
            w.append(s["width_ratio"])
    gate = {"R1_held_out_margin_ci_above_0": bool(robust["held_out"]["margin_lo"] > 0), "R2_all_margin_ci_above_0": bool(robust["all"]["margin_lo"] > 0),
            "R3_near_valid_everywhere": all(res["designs"][d]["dalo_near_valid"] for d in all_d), "R4_mean_width_ratio": float(np.mean(w)), "R4_ok": bool(np.mean(w) <= 1.10)}
    gate["pass"] = all(gate[k] for k in ("R1_held_out_margin_ci_above_0", "R2_all_margin_ci_above_0", "R3_near_valid_everywhere", "R4_ok"))
    res["robustness"], res["robustness_gate"] = robust, gate
    res["brief_gate_0.03"] = {d: {k: res["designs"][d].get(k) for k in ("comparator", "gain", "gain_lo", "gain_hi", "p_holm", "gain_ge_0.03", "width_ratio", "width_ok", "recovery_p_holm")} for d in REQUIRED_INFO}
    os.makedirs(RESULTS_DIR / "tables", exist_ok=True)
    with open(RESULTS_DIR / "tables" / "m5_main.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["design", "label", "marginal", "worst", "worst_50", "worst_200", "width", "median_width", "interval_score", "frac_inf", "recovery"])
        for d in designs:
            for l, v in res["designs"][d]["table"].items():
                wr.writerow([d, l] + [round(v.get(k, float("nan")), 5) for k in ("marginal", "worst", "worst_50", "worst_200", "width", "median_width", "interval_score", "frac_inf", "recovery")])
    sens = {}
    for d, target in [(x, 0.95) for x in ("alpha05_a01_local_geometric", "alpha05_a03_smooth_alternating", "alpha05_a06_gradual", "alpha05_elec2")] + \
                     [(x, 0.9) for x in ("cqr_a01_local_geometric", "cqr_a03_smooth_alternating", "cqr_a08_variance", "cqr_elec2")]:
        sens[d] = design_summary(load(d), target)
    res["sensitivity"] = sens
    res["f5"], res["f10"] = {}, {}
    # TRACE: descriptive, outside R1-R4 (TRACE protocol). The gap uses each method's no-delay reference where one was run.
    res["trace"] = {}
    for part in ("trace_p", "trace_d"):
        D = load(part)
        s = design_summary(D, 0.9)
        tab = regret_table({part: D}, [part], 0.9)
        s["regret"] = {f: tab[f][part] for f in tab}
        for lab, ref in ((DALO, ORACLE), ("regional_aci_g0.005_p", "regional_aci_oracle")):
            g = np.array([D[ref][u]["worst"] - D[lab][u]["worst"] for u in sorted(D[lab])])
            s.setdefault("gap_to_no_delay", {})[lab] = {"mean": float(g.mean()), "lo": float(g.mean() - stats.t.ppf(0.975, len(g) - 1) * g.std(ddof=1) / np.sqrt(len(g))),
                                                         "hi": float(g.mean() + stats.t.ppf(0.975, len(g) - 1) * g.std(ddof=1) / np.sqrt(len(g)))}
        res["trace"][part] = s
    json.dump(res, open(RESULTS_DIR / "summaries" / "m5_results.json", "w"), indent=1, default=float)
    print("ROBUSTNESS GATE", json.dumps(gate))
    for part, s in res["trace"].items():
        print(part, s.get("comparator"), "gain", s.get("gain"), [s.get("gain_lo"), s.get("gain_hi")], "width", s.get("width_ratio"), "nv", s["dalo_near_valid"], "gap", s["gap_to_no_delay"], "regret", s["regret"])
    for tag in robust:
        r = robust[tag]
        print(tag, "margin", round(r["margin"], 4), [round(r["margin_lo"], 4), round(r["margin_hi"], 4)], "best competitor", r["best_competitor"])
        print("  max regret", {k: round(v, 3) for k, v in sorted(r["max_regret"].items(), key=lambda kv: kv[1])})
    for d in designs:
        s = res["designs"][d]
        print(f"{d:26s} {s.get('comparator', '-'):24s} gain {s.get('gain', float('nan')):+.3f} [{s.get('gain_lo', float('nan')):+.3f},{s.get('gain_hi', float('nan')):+.3f}] p {s.get('p_holm', float('nan')):.3g} width {s.get('width_ratio', float('nan')):.3f} "
              f"rec {s.get('recovery_dalo', float('nan')):.0f}/{s.get('recovery_comparator', float('nan')):.0f} nv {s['dalo_near_valid']}")


if __name__ == "__main__":
    main()
