"""Summarises results/summaries/theory_checks.json for theory/claims.md: how informative the checks were."""
import json

import numpy as np

from delayed_local_cp.theory import bounds as B

R = json.load(open("results/summaries/theory_checks.json"))
T = R["c2"]["table"]
WIN = (25, 50, 100, 200, 400)
out = {"c1_c3": R["c1_c3"], "c2c": R["c2c"]}
pairs = [(r["kind"], r["mean"], r["method"], L, r[f"cov_{L}"], r[f"ub_{L}"]) for r in T for L in WIN]
star = [(r["kind"], r["mean"], L, r[f"ub_{L}"], r[f"ubstar_{L}"], r[f"cov_{L}"]) for r in T for L in WIN]
nt_star = [p for p in star if p[4] < 0.999]
out["tv_star"] = {"violations": R["c2"]["tv_star_violations_beyond_3se"], "never_weaker": all(p[4] <= p[3] + 1e-9 for p in star),
                  "nontrivial": len(nt_star), "nontrivial_fixed": sum(p[0] == "fixed" for p in nt_star), "nontrivial_geometric": sum(p[0] == "geometric" for p in nt_star),
                  "mean_slack_nontrivial": float(np.mean([p[4] - p[5] for p in nt_star])) if nt_star else None}
W = R["c2"]["widening"]
out["widening"] = {"comparisons": len(W), "violations_beyond_3se": int(sum(x["a"] - 3 * np.sqrt(max(x["a"] * (1 - x["a"]), 1e-4) / 500) > x["bound"] for x in W)),
                   "nontrivial": sum(x["bound"] < 1 for x in W), "nontrivial_and_a_above_half_bound": sum(x["bound"] < 1 and x["a"] > x["bound"] / 2 for x in W),
                   "max_a_over_bound_nontrivial": float(max((x["a"] / x["bound"] for x in W if x["bound"] < 1), default=float("nan")))}
nontriv = [p for p in pairs if p[5] < 0.999]
out["bound"] = {"comparisons": len(pairs), "violations": R["c2"]["bound_violations_beyond_3se"], "nontrivial": len(nontriv),
                "nontrivial_fixed": sum(p[0] == "fixed" for p in nontriv), "nontrivial_geometric": sum(p[0] == "geometric" for p in nontriv),
                "blind_window_comparisons": sum(p[0] == "fixed" and p[3] <= p[1] for p in pairs),
                "mean_slack_blind": float(np.mean([p[5] - p[4] for p in pairs if p[0] == "fixed" and p[3] <= p[1] and p[1] >= 50 and p[5] < 0.999])),
                "min_slack_nontrivial": float(min(p[5] - p[4] for p in nontriv))}
# Lag prediction, restricted to delayed cases where the method is stable (no oscillation in E0)
base = {r["method"]: r["q0_sd"] for r in T if r["mean"] == 1 and r["kind"] == "fixed"}
errs = {}
for r in T:
    if r["mean"] < 20 or r["q0_sd"] > 2 * base[r["method"]]:
        continue
    errs.setdefault((r["method"], r["kind"]), []).append((r["mean"], r["lag_to_target"], r["predicted_lag"]))
out["lag_prediction_stable_cases"] = {f"{k[0]}/{k[1]}": {"rows": v, "mean_abs_rel_error": float(np.mean([abs(l - p) / l for _, l, p in v]))} for k, v in errs.items()}
# Excess miscoverage attributable to delay vs the blind-period prediction p * D * (1 - alpha - kappa(q0, delta))
ex = []
for r in T:
    if r["kind"] == "fixed" and r["method"].startswith("tracker") and r["mean"] >= 50 and r.get("eta_over_stability_limit", 9) < 1:
        p0 = next(x["peak_excess"] for x in T if x["method"] == r["method"] and x["mean"] == 1 and x["kind"] == "fixed")
        q = r["e0_width"] / 2
        ex.append({"method": r["method"], "D": r["mean"], "observed": r["peak_excess"] - p0,
                   "predicted": 0.5 * r["mean"] * (0.9 - float(B.kappa(q, 0.6, 0.3)))})
out["excess_vs_blind_prediction"] = ex
# Stability: E0 width growth against the no-delay width
stab = []
for r in T:
    w0 = next(x["e0_width"] for x in T if x["method"] == r["method"] and x["mean"] == 1 and x["kind"] == "fixed")
    if r["mean"] in (100, 400):
        stab.append({"method": r["method"], "kind": r["kind"], "m": r["mean"], "width_ratio_no_shift": r["e0_width"] / w0,
                     "stab": r.get("eta_over_stability_limit")})
out["stability"] = stab
out["e0_kappa_at_q0"] = float(B.kappa(0.4975, 0.6, 0.3))
json.dump(out, open("results/summaries/theory_summary.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("bound", "tv_star", "widening", "c2c")}, indent=0))
for k, v in out["lag_prediction_stable_cases"].items():
    print(k, round(v["mean_abs_rel_error"], 3), v["rows"])
