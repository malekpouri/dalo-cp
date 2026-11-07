"""Numerical checks of the theory claims (status 'checked-numerically' only). Output: results/summaries/theory_checks.json.

C1: pathwise regional bounds, any delays, including Y-dependent. C3: pathwise window identity and bound.
C2: coverage upper bound on the exact Gaussian instance; detection-lag scaling. C2c: global threshold failure.
"""
import json
import math
import sys
from multiprocessing import Pool

import numpy as np

from delayed_local_cp.delays.models import sample_delays
from delayed_local_cp.methods.base import RoundContext
from delayed_local_cp.methods.baselines import build_method
from delayed_local_cp.streams.delay_queue import DelayQueue
from delayed_local_cp.theory import bounds as B
from delayed_local_cp.utils.paths import RESULTS_DIR

ALPHA, SIGMA, DELTA, P_SLOW = 0.1, 0.3, 0.6, 0.5
N, T0 = 8000, 4000
TAIL_LAGS, TAIL_W, TAIL_Q = (50, 100, 200, 400), 25, (0.6, 0.85)  # draft check of the widening bound; q1 values fixed before running


def simulate(cfg, x, reg, y, d, k, scale):
    """Runs a method. Returns q, err, arrival order, c trajectory per round, pending-after-issue per region."""
    n = len(y)
    m = build_method(cfg, ALPHA, k, {"score_scale": scale})
    q = DelayQueue()
    arr = np.arange(n) + d
    qs = np.zeros(n)
    pend = np.zeros((n, k), dtype=np.int64)
    c = np.zeros((n, k))
    for t in range(n):
        for i, r in q.pop_due(t):
            m.on_arrival(i, x[i], r, 0.0, float(y[i]), t)
        if hasattr(m, "c"):
            c[t] = m.c
        qs[t] = m.issue(t, x[t], int(reg[t]), 0.0, RoundContext(t, q.pending_counts(k), q.n_pending))
        q.push(t, int(arr[t]), int(reg[t]))
        pend[t] = q.pending_counts(k)
    return qs, c, pend, arr, m


C1_CFGS = [
    ("base", {"type": "dalo_cp", "eta_rel": 0.05, "window": 100}),
    ("age_arrival_borrow_guard", {"type": "dalo_cp", "eta_rel": 0.05, "window": 100, "age_tau": 25, "arrival_ipw": True,
                                  "borrow": {"on": True, "lambda": 1.0, "cap_rel": 1.0}, "guard": {"on": True, "kappa_rel": 0.5}}),
    ("large_step", {"type": "dalo_cp", "eta_rel": 1.0, "window": 50}),
]
C1_DELAYS = [{"type": "fixed", "value": 40}, {"type": "geometric", "per_region": [1, 60, 5, 200], "d_max": 800},
             {"type": "pareto", "scale": 3, "shape": 1.2, "d_max": 2000},
             {"type": "y_dependent", "base": {"type": "geometric", "per_region": [2, 50, 10, 100]}, "y_coef": 1.0, "d_max": 2000}]


def c1_one(args):
    seed, (name, cfg), dcfg = args
    rng = np.random.default_rng(seed)
    n, k = 3000, 4
    reg = rng.integers(0, k, n)
    x = ((reg + rng.random(n)) / k)[:, None]
    # heavy tails and an abrupt adversarial-looking change in one region
    y = rng.standard_t(3, n) * SIGMA + 2.0 * ((reg == 1) & (np.arange(n) > 1500))
    d = sample_delays(dcfg, x, reg, y, rng)
    scale = 0.3 * 0.6745
    qs, c, pend, arr, m = simulate(cfg, x, reg, y, d, k, scale)
    s = np.abs(y)
    err = (s > qs).astype(float)
    Bmax = float(s.max())
    G = (float((cfg.get("borrow") or {}).get("cap_rel", 0)) + float((cfg.get("guard") or {}).get("kappa_rel", 0))) * scale
    eta = m.eta
    worst_ratio, viol = 0.0, 0
    win_ratio, win_viol = 0.0, 0
    for r in range(k):
        idx = np.flatnonzero(reg == r)
        P = int(pend[:, r].max())
        for T in np.linspace(200, n - 1, 25).astype(int):
            A = idx[arr[idx] <= T]
            I = idx[idx <= T]
            if len(A):
                v = err[A].mean() - ALPHA
                hi, lo = Bmax / (eta * len(A)) + (1 - ALPHA) * P / len(A), (Bmax + G) / (eta * len(A)) + ALPHA * P / len(A)
                viol += int(v > hi + 1e-9 or -v > lo + 1e-9)
                worst_ratio = max(worst_ratio, v / hi if v > 0 else -v / lo)
            if len(I):
                v = err[I].mean() - ALPHA
                hi, lo = Bmax / (eta * len(I)) + 2 * (1 - ALPHA) * P / len(I), (Bmax + G) / (eta * len(I)) + 2 * ALPHA * P / len(I)
                viol += int(v > hi + 1e-9 or -v > lo + 1e-9)
        # window bound: |sum_I (err - alpha)| <= |c_e - c_s| / eta + P
        for _ in range(40):
            a, b = sorted(rng.integers(0, n - 1, 2))
            if b <= a:
                continue
            I = idx[(idx >= a) & (idx <= b)]
            if not len(I):
                continue
            total = abs(float((err[I] - ALPHA).sum()))
            rhs = abs(c[b, r] - c[a, r]) / eta + P
            win_viol += int(total > rhs + 1e-6)
            win_ratio = max(win_ratio, total / rhs)
    return {"cfg": name, "delay": dcfg["type"], "violations": viol, "max_ratio": worst_ratio, "window_violations": win_viol, "window_max_ratio": win_ratio}


def instance(seed, delay, delta):
    rng = np.random.default_rng([seed, 11])
    reg = (rng.random(N) < P_SLOW).astype(np.int64)
    x = ((reg + rng.random(N)) / 2)[:, None]
    y = SIGMA * rng.standard_normal(N) + delta * ((reg == 1) & (np.arange(N) >= T0))
    d = np.where(reg == 1, sample_delays(delay, x, np.zeros(N, dtype=int), y, rng), 1)
    return x, reg, y, d


C2_METHODS = {
    "regional_aci": {"type": "regional_aci", "gamma": 0.05, "window": 250, "project": True},
    "dalo_cp": {"type": "dalo_cp", "eta_rel": 0.05, "window": 250, "age_tau": 25, "arrival_ipw": True},
    "tracker_e0.02": {"type": "dalo_cp", "eta_rel": 0.02, "window": 250},
    "tracker_e0.1": {"type": "dalo_cp", "eta_rel": 0.1, "window": 250},
    "tracker_e0.5": {"type": "dalo_cp", "eta_rel": 0.5, "window": 250},
}
BIN = 10
WINDOWS = (25, 50, 100, 200, 400)


def c2_one(args):
    seed, kind, mean, name = args
    dcfg = {"type": "fixed", "value": mean} if kind == "fixed" else {"type": "geometric", "mean": mean, "d_max": 20 * mean}
    scale = SIGMA * 0.6745
    out = {}
    for tag, delta in (("e0", 0.0), ("e1", DELTA)):
        x, reg, y, d = instance(seed, dcfg, delta)
        qs, *_ = simulate(C2_METHODS[name], x, reg, y, d, 2, scale)
        idx = np.flatnonzero((reg == 1) & (np.arange(N) >= T0))
        lag = idx - T0
        err = (np.abs(y[idx]) > qs[idx]).astype(float)
        out[tag] = {"lag": lag, "err": err, "q": qs[idx]}
    res = {}
    for L in WINDOWS:
        m1 = out["e1"]["lag"] < L
        m0 = out["e0"]["lag"] < L
        # Counts, pooled over seeds later: a mean of per-seed ratios is biased at order 1/(pL) (see notation.md, coverage criteria).
        res[f"hit_{L}"], res[f"n1_{L}"] = float(np.sum(1 - out["e1"]["err"][m1])), int(m1.sum())
        res[f"q0sum_{L}"], res[f"n0_{L}"] = float(np.minimum(out["e0"]["q"][m0], 50).sum()), int(m0.sum())
    # Draft check of the widening bound: P_1(q > q1) against P_0(q > q1), over a short window ending at each lag.
    for L in TAIL_LAGS:
        w1 = (out["e1"]["lag"] >= L - TAIL_W) & (out["e1"]["lag"] < L)
        w0 = (out["e0"]["lag"] >= L - TAIL_W) & (out["e0"]["lag"] < L)
        for q1 in TAIL_Q:
            res[f"a_{L}_{q1}"], res[f"e_{L}_{q1}"] = int(np.sum(out["e1"]["q"][w1] > q1)), int(np.sum(out["e0"]["q"][w0] > q1))
        res[f"na_{L}"], res[f"ne_{L}"] = int(w1.sum()), int(w0.sum())
    bins = (out["e1"]["lag"] // BIN).astype(int)
    n_b = (N - T0) // BIN + 1
    res["bin_err"] = np.bincount(bins, weights=out["e1"]["err"], minlength=n_b)
    res["bin_n"] = np.bincount(bins, minlength=n_b)
    # stability with no shift: spread of the threshold in the stationary environment E0
    res["q0_sd"] = float(np.std(np.minimum(out["e0"]["q"], 50)))
    res["e0_width"] = float(2 * np.minimum(out["e0"]["q"], 50).mean())
    return {"seed": seed, "kind": kind, "mean": mean, "method": name, **res}


def c2c_one(seed):
    x, reg, y, d = instance(seed, {"type": "fixed", "value": 1}, DELTA)
    qs, *_ = simulate({"type": "aci", "gamma": 0.02, "window": 500, "project": True}, x, reg, y, d, 2, SIGMA * 0.6745)
    tail = np.arange(N) >= T0 + 1500
    cov = np.abs(y) <= qs
    return float(cov[tail & (reg == 1)].mean()), float(cov[tail & (reg == 0)].mean()), float(cov[tail].mean())


def main(seeds=200):
    out = {}
    with Pool(10) as pool:
        jobs = [(s, c, d) for s in range(30) for c in C1_CFGS for d in C1_DELAYS]
        r1 = pool.map(c1_one, jobs, chunksize=4)
        out["c1_c3"] = {"instances": len(r1), "violations": sum(r["violations"] for r in r1), "window_violations": sum(r["window_violations"] for r in r1),
                        "max_ratio": max(r["max_ratio"] for r in r1), "window_max_ratio": max(r["window_max_ratio"] for r in r1)}
        print("C1/C3", out["c1_c3"], flush=True)
        means = (1, 5, 20, 50, 100, 200, 400)
        jobs = [(s, kind, m, name) for s in range(seeds) for kind in ("fixed", "geometric") for m in means for name in C2_METHODS]
        r2 = pool.map(c2_one, jobs, chunksize=8)
        r3 = pool.map(c2c_one, range(100), chunksize=4)
    k = B.gaussian_kl(DELTA, SIGMA)
    table, tail, viol, viol_s = [], [], 0, 0
    no_delay_lag = {}
    for name in C2_METHODS:
        for kind in ("fixed", "geometric"):
            for m in means:
                rs = [r for r in r2 if r["kind"] == kind and r["mean"] == m and r["method"] == name]
                err = np.sum([r["bin_err"] for r in rs], axis=0)
                cnt = np.sum([r["bin_n"] for r in rs], axis=0)
                rate = err / np.maximum(cnt, 1)
                cum = np.cumsum(err - ALPHA * cnt) / len(rs)       # mean cumulative excess miscoverage (labels per run)
                below = np.flatnonzero(rate <= ALPHA + 0.05)
                row = {"kind": kind, "mean": m, "method": name, "peak_excess": float(cum.max()),
                       "lag_to_target": float((below[0] + 1) * BIN) if len(below) else float("nan"),
                       "q0_sd": float(np.mean([r["q0_sd"] for r in rs])), "e0_width": float(np.mean([r["e0_width"] for r in rs])),
                       "info_lag_1nat": B.information_lag(kind, m, P_SLOW, k)}
                for L in WINDOWS:
                    # Empirical analogue of the theorem's quantity: pooled over seeds, on a fixed window (never the worst window).
                    hit, n1 = np.array([r[f"hit_{L}"] for r in rs]), np.array([r[f"n1_{L}"] for r in rs], dtype=float)
                    cov = float(hit.sum() / n1.sum())
                    se = float(np.std(hit - cov * n1, ddof=1) / n1.mean() / np.sqrt(len(rs)))     # ratio estimator, linearised
                    q0 = float(sum(r[f"q0sum_{L}"] for r in rs) / sum(r[f"n0_{L}"] for r in rs))
                    ub = B.window_coverage_upper_bound(q0, DELTA, SIGMA, kind, m, P_SLOW, L)
                    ub_s = B.window_coverage_upper_bound_star(q0, DELTA, SIGMA, kind, m, P_SLOW, L)
                    row[f"cov_{L}"], row[f"ub_{L}"], row[f"ubstar_{L}"], row[f"se_{L}"] = cov, ub, ub_s, se
                    viol += int(cov - 3 * se > ub)
                    viol_s += int(cov - 3 * se > ub_s)
                # Mixture over the window's rounds: the bound applies with the window's mean information (extensions draft, Prop. E.2).
                lam = B.expected_arrivals(kind, m, P_SLOW, max(TAIL_LAGS))
                for L in TAIL_LAGS:
                    info = k * float(lam[L - TAIL_W:L].mean())
                    for q1 in TAIL_Q:
                        na, ne = sum(r[f"na_{L}"] for r in rs), sum(r[f"ne_{L}"] for r in rs)
                        a, e = sum(r[f"a_{L}_{q1}"] for r in rs) / na, sum(r[f"e_{L}_{q1}"] for r in rs) / ne
                        e_up = min((e * ne + 3 * math.sqrt(e * ne + 1) + 3) / ne, 0.999)  # conservative: a zero count must not give a zero bound
                        bound = float(B.widening_probability_bound(info, e_up))
                        tail.append({"kind": kind, "mean": m, "method": name, "lag": L, "q1": q1, "a": a, "e": e, "e_up": e_up, "info": info, "bound": bound})
                table.append(row)
                if m == 1:
                    no_delay_lag[name] = row["lag_to_target"]
    # Parameter-free prediction: a method that needs n labels without delay (n = p * lag) has them at Lambda^{-1}(n) with delay.
    for r in table:
        n_lab = P_SLOW * no_delay_lag[r["method"]]
        r["predicted_lag"] = B.lag_for_labels(r["kind"], r["mean"], P_SLOW, n_lab)
    density = 2 * float(B.stats.norm.pdf(1.6449)) / SIGMA
    for r in table:
        if r["method"].startswith("tracker"):
            eta = float(r["method"].split("_e")[1]) * SIGMA * 0.6745
            r["eta_over_stability_limit"] = eta / B.stability_limit_step(density, P_SLOW * r["mean"])
    out["c2"] = {"table": table, "bound_violations_beyond_3se": viol, "tv_star_violations_beyond_3se": viol_s, "n_comparisons": len(table) * len(WINDOWS), "widening": tail}
    q = B.brentq(lambda q: P_SLOW * float(B.kappa(q, DELTA, SIGMA)) + (1 - P_SLOW) * float(B.kappa(q, 0, SIGMA)) - 0.9, 0.01, 3)
    a = np.array(r3)
    out["c2c"] = {"q_marginal_90": q, "predicted_slow": float(B.kappa(q, DELTA, SIGMA)), "predicted_fast": float(B.kappa(q, 0, SIGMA)),
                  "simulated_slow": float(a[:, 0].mean()), "simulated_fast": float(a[:, 1].mean()), "simulated_marginal": float(a[:, 2].mean())}
    q0 = B.q_for_coverage(0.9, 0, SIGMA)
    out["c4_blind_coverage"] = [{"delta_r": dr, "transfer": tr, "coverage": float(B.kappa(q0 + tr, dr, SIGMA)), "half_width_added": tr}
                                for dr in (0.6, 0.3, 0.0) for tr in (0.0, 0.15, 0.3, 0.6)]
    json.dump(out, open(RESULTS_DIR / "summaries" / "theory_checks.json", "w"), indent=1)
    print("C2 bound violations", viol, "of", out["c2"]["n_comparisons"])
    print("C2c", out["c2c"])
    for r in table:
        print(f"{r['method']:14s} {r['kind']:9s} m={r['mean']:3d} lag {r['lag_to_target']:6.0f} pred {r['predicted_lag']:5d} info {r['info_lag_1nat']:4d} peak {r['peak_excess']:6.1f} "
              f"q0sd {r['q0_sd']:.3f} w0 {r['e0_width']:.3f} stab {r.get('eta_over_stability_limit', float('nan')):5.2f} | L50 {r['cov_50']:.3f}/{r['ub_50']:.3f} L200 {r['cov_200']:.3f}/{r['ub_200']:.3f}")
    for r in out["c4_blind_coverage"]:
        print(r)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
