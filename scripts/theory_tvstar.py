"""Draft analysis for theory/proofs/claim2_extensions_draft.md: is TV* stronger than the current bound, and what does the widening bound require?"""
import json

import numpy as np

from delayed_local_cp.theory import bounds as B
from delayed_local_cp.utils.paths import RESULTS_DIR

SIGMA, P, ALPHA = 0.3, 0.5, 0.1
out = {"note": "p = 0.5, sigma = 0.3. Threshold: first L where the per-round bound with kappa_bar(q0) reaches 0.9."}
HORIZON = 600


def first(path, level):
    hit = np.flatnonzero(path >= level)
    return int(hit[0]) if len(hit) else None


rows = []
for delta_sig in (0.2, 0.5, 1.0, 2.0, 4.0):       # K = delta^2 / (2 sigma^2) = 0.02, 0.125, 0.5, 2, 8
    delta = delta_sig * SIGMA
    q0 = 1.645 * SIGMA
    need = 0.9 - float(B.kappa_envelope(q0, delta, SIGMA))
    for kind, m in (("geometric", 100), ("geometric", 400), ("geometric", 1600), ("fixed", 100), ("fixed", 400)):
        h = max(HORIZON, 2 * m) if kind == "fixed" else HORIZON
        cur, star = B.tv_path(delta, SIGMA, kind, m, P, h), B.tv_star_path(delta, SIGMA, kind, m, P, h)
        rows.append({"K": delta_sig ** 2 / 2, "kind": kind, "m": m, "tv_needed": need, "L_current": first(cur, need), "L_star": first(star, need),
                     "L_first_label": first(B.expected_arrivals(kind, m, P, h), 1.0)})
out["first_L"] = rows

# Same I(L), different TV*: does TV* separate delay laws that the budget cannot?
same = []
for delta_sig, n_labels in ((0.5, 32), (1.0, 8), (2.0, 2), (4.0, 1)):    # n_labels post-shift rounds past the blind period, so I(L) is about 2 (4 for K = 8)
    delta = delta_sig * SIGMA
    k = B.gaussian_kl(delta, SIGMA)
    for m in (100, 400):
        L_fixed = m + n_labels - 1
        info = k * float(B.expected_arrivals("fixed", m, P, L_fixed)[L_fixed])
        lam_g = k * B.expected_arrivals("geometric", m, P, 3000)
        L_geo = int(np.searchsorted(lam_g, info))
        same.append({"K": k, "m": m, "I": info, "I_geometric": float(lam_g[L_geo]), "L_fixed": L_fixed, "L_geometric": L_geo,
                     "tv_star_fixed": float(B.tv_star_path(delta, SIGMA, "fixed", m, P, L_fixed)[-1]),
                     "tv_star_geometric": float(B.tv_star_path(delta, SIGMA, "geometric", m, P, L_geo)[-1]),
                     "mean_plug_in": float(B.gaussian_tv_of_kl(info)), "current_bound": float(B.tv_bound(info))})
out["same_budget"] = same

# Widening bound: information needed for coverage 0.85 by a method that widens past q1 with probability eps under E0
delta, q1, target, k = 0.6, 1.645 * SIGMA, 0.85, 2.0
wid = []
for eps in (0.2, 0.1, 0.01, 1e-3, 1e-4, 1e-6):
    need = B.information_needed(target, q1, delta, SIGMA, eps)
    a = (target - float(B.kappa(q1, delta, SIGMA))) / (1 - float(B.kappa(q1, delta, SIGMA)))
    row = {"eps": eps, "a_required": a, "info_needed": need, "info_needed_simple": max(a * np.log(1 / eps) - np.log(2), 0.0)}
    for kind, m in (("fixed", 100), ("geometric", 100), ("geometric", 400), ("geometric", 1600)):
        row[f"L_{kind}_{m}"] = B.information_lag(kind, m, P, k, nats=need) if need > 0 else 0
    wid.append(row)
out["widening_requirement"] = wid
# Which is stronger at a given round for a concentrated method? Coverage bound by each route at m = 400.
cmp_rows = []
star = B.tv_star_path(delta, SIGMA, "geometric", 400, P, 200)
lam = k * B.expected_arrivals("geometric", 400, P, 200)
kb, kq = float(B.kappa_envelope(q1, delta, SIGMA)), float(B.kappa(q1, delta, SIGMA))
for L in (10, 20, 30, 40, 50, 60, 80, 100, 150, 200):
    r = {"L": L, "tv_current": float(min(1, kb + B.tv_bound(lam[L]))), "tv_star": float(min(1, kb + star[L]))}
    for eps in (0.1, 0.01, 1e-4):
        r[f"widening_eps{eps}"] = float(kq + (1 - kq) * B.widening_probability_bound(lam[L], eps))
    cmp_rows.append(r)
out["coverage_bound_m400"] = cmp_rows
out["kappa_q0"], out["kappa_bar_q0"] = kq, kb
q0_valid = B.q_for_coverage(0.9, 0, SIGMA)
mw = []
for c in (0.85, 0.89, 0.9, 0.95, 0.99):
    info, q1 = B.mean_width_information(q0_valid, c, 0.6, SIGMA)
    mw.append({"target": c, "info": info, "q1": q1, "L_m100": B.information_lag("geometric", 100, P, 2.0, nats=info) if info > 0 else None,
               "L_m400": B.information_lag("geometric", 400, P, 2.0, nats=info) if info > 0 else None})
out["mean_width_corollary"] = mw
json.dump(out, open(RESULTS_DIR / "summaries" / "theory_tvstar.json", "w"), indent=1)
print(mw)
for r in rows:
    print(r["K"], r["kind"], r["m"], "need", round(r["tv_needed"], 3), "L cur", r["L_current"], "star", r["L_star"], "first label", r["L_first_label"])
for r in same:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
for r in wid:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
for r in cmp_rows:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
