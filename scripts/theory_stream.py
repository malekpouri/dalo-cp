"""Draft checks for theory/proofs/claim2_stream_draft.md. Output: results/summaries/theory_stream.json. Evidence only; none of this is a proof."""
import json
import math
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, "scripts")
from theory_checks import C2_METHODS, simulate  # noqa: E402

from delayed_local_cp.delays.models import sample_delays  # noqa: E402
from delayed_local_cp.theory import bounds as B  # noqa: E402
from delayed_local_cp.utils.paths import RESULTS_DIR  # noqa: E402

SIGMA, DELTA, P = 0.3, 0.6, 0.5
K = B.gaussian_kl(DELTA, SIGMA)
N, T_LO, T_HI = 8000, 2500, 6500
LAGS = (25, 50, 100, 200, 400, 800)
Q1 = (0.6, 0.7, 0.85)
DELAYS = [("fixed", 20), ("fixed", 100), ("fixed", 400), ("geometric", 20), ("geometric", 100), ("geometric", 400)]
Q_NARROW, Q_WIDE = 1.645 * SIGMA, 1.2


def dcfg(kind, m):
    return {"type": "fixed", "value": m} if kind == "fixed" else {"type": "geometric", "mean": m, "d_max": 20 * m}


def instance(seed, kind, m, t0):
    """t0 = None gives E0. Inputs and delays do not depend on t0 (assumption A)."""
    rng = np.random.default_rng([seed, 23, 0 if kind == "fixed" else 1, m])
    reg = (rng.random(N) < P).astype(np.int64)
    x = ((reg + rng.random(N)) / 2)[:, None]
    z = SIGMA * rng.standard_normal(N)
    d = np.where(reg == 1, sample_delays(dcfg(kind, m), x, np.zeros(N, dtype=int), z, rng), 1)
    y = z + (DELTA * ((reg == 1) & (np.arange(N) >= t0)) if t0 is not None else 0.0)
    return x, reg, y, d


def method_one(args):
    seed, kind, m, name = args
    t0 = int(np.random.default_rng([seed, 5]).integers(T_LO, T_HI))
    scale = SIGMA * 0.6745
    x, reg, y, d = instance(seed, kind, m, None)
    q0, *_ = simulate(C2_METHODS[name], x, reg, y, d, 2, scale)
    x, reg, y, d = instance(seed, kind, m, t0)
    q1s, *_ = simulate(C2_METHODS[name], x, reg, y, d, 2, scale)
    idx = np.flatnonzero(reg == 1)
    res = {"seed": seed, "kind": kind, "m": m, "method": name}
    prev = idx[np.searchsorted(idx, t0) - 1]
    win = idx[(idx >= T_LO) & (idx < T_HI + max(LAGS))]
    for q in Q1:
        wide0 = np.minimum(q0[win], 50) > q
        res[f"up_{q}"] = int(np.sum(wide0[1:] & ~wide0[:-1]))
        res[f"s_{q}"] = int(q0[prev] > q)
        for L in LAGS:
            t = idx[min(np.searchsorted(idx, t0 + L), len(idx) - 1)]
            res[f"a_{q}_{L}"] = int(q1s[t] > q)
    for L in LAGS:
        t = idx[min(np.searchsorted(idx, t0 + L), len(idx) - 1)]
        res[f"lag_{L}"] = int(t - t0)
        res[f"cov_{L}"] = int(abs(y[t]) <= q1s[t])
    return res


def detector_alarm(args):
    """Proposition S.3: first round where some suffix likelihood ratio of arrived labels reaches exp(b); checked at every arrival with a positive increment."""
    seed, kind, m, t0, b, H = args
    _, reg, y, d = instance(seed, kind, m, t0)
    idx = np.flatnonzero(reg == 1)
    arrival = idx + d[idx]
    inc = DELTA * y[idx] / SIGMA ** 2 - K
    order = np.argsort(arrival, kind="stable")
    v = np.zeros(len(idx))
    for pos in order:
        if arrival[pos] >= N:
            break
        v[pos] = inc[pos]
        if inc[pos] > 0:
            top = np.searchsorted(idx, arrival[pos])            # labels of rounds before the current round only
            low = max(top - H, 0)
            if np.cumsum(v[low:top][::-1]).max() >= b:
                return int(arrival[pos])
    return None


def main():
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    out = {"settings": {"sigma": SIGMA, "delta": DELTA, "p": P, "K": K, "q_narrow": Q_NARROW, "q_wide": Q_WIDE, "seeds": seeds, "n": N}}
    # A. Analytic: smallest lag allowed by the lower bound, against the lag the detector guarantees.
    delta_f = 0.05
    k1 = float(B.kappa(Q_NARROW, DELTA, SIGMA))
    grid = np.linspace(0.05, 40, 800)
    rows = []
    for kind, m in (("fixed", 100), ("fixed", 400), ("geometric", 100), ("geometric", 400), ("geometric", 1600)):
        h = 1800 if not (kind == "geometric" and m == 1600) else 900
        tab = B.arrival_power_table(kind, m, P, K, grid, h)
        for T in (10 ** 4, 10 ** 5, 10 ** 6, 10 ** 8):
            for c in (0.85, 0.9):
                power = (c - k1) / (1 - k1)
                # Same narrow level on both sides. After an alarm the detector is infinitely wide; see the proposition's remark for finite width.
                up = B.arrival_power_table(kind, m, P, K, [math.log(T / delta_f)], h)[:, 0]
                hit = np.flatnonzero(up >= power)
                rows.append({"T": T, "kind": kind, "m": m, "target": c, "power": power, "L_lower": B.lower_lag_neyman_pearson(kind, m, P, K, power, delta_f, delta_f, T, grid, tab),
                             "L_lower_kl": B.lower_lag_false_alarm(kind, m, P, K, power, delta_f, T), "L_detector": int(hit[0]) if len(hit) else None, "b": math.log(T / delta_f)})
    out["analytic"] = rows
    with Pool(10) as pool:
        # B. The detector itself. T = N. Monte Carlo of false alarm and of the alarm lag.
        b = math.log(N / delta_f)
        det = {}
        for kind, m in DELAYS:
            fa = pool.map(detector_alarm, [(s, kind, m, None, b, 3000) for s in range(seeds)], chunksize=4)
            al = pool.map(detector_alarm, [(s, kind, m, 4000, b, 3000) for s in range(seeds)], chunksize=4)
            path = B.arrival_power_table(kind, m, P, K, [b], 1000)[:, 0]
            lag = {L: {"observed": float(np.mean([a is not None and 4000 <= a <= 4000 + L for a in al])), "guarantee": float(path[L])} for L in LAGS}
            det[f"{kind}_{m}"] = {"false_alarm": float(np.mean([a is not None for a in fa])), "bound": delta_f, "power": lag,
                                  "pre_change_alarm": float(np.mean([a is not None and a < 4000 for a in al]))}
        out["detector"] = det
        # C. Real methods against Theorem S.1, with candidates spread over [T_LO, T_HI).
        jobs = [(s, kind, m, name) for s in range(seeds) for kind, m in DELAYS for name in C2_METHODS]
        res = pool.map(method_one, jobs, chunksize=8)
    table, viol, viol_np = [], 0, 0
    tabs = {(kind, m): B.arrival_power_table(kind, m, P, K, grid, 1200) for kind, m in DELAYS}
    for kind, m in DELAYS:
        lam = K * B.expected_arrivals(kind, m, P, 1200)
        for name in C2_METHODS:
            rs = [r for r in res if r["kind"] == kind and r["m"] == m and r["method"] == name]
            for q in Q1:
                rho = float(np.mean([r[f"s_{q}"] for r in rs]))
                ups = float(np.mean([r[f"up_{q}"] for r in rs]))
                for L in LAGS:
                    a = float(np.mean([r[f"a_{q}_{L}"] for r in rs]))
                    se = math.sqrt(max(a * (1 - a), 1e-4) / len(rs))
                    span = T_HI - T_LO + max(LAGS)
                    nu = max(ups * (L + 1) / span, 1e-4)          # events per candidate window, averaged over comb offsets
                    # Lag is the first shifted-region round at or after t0 + L: use the 99th percentile of that lag, which favours the method.
                    lag = int(np.percentile([r[f"lag_{L}"] for r in rs], 99))
                    info = float(lam[min(lag, 1200)])
                    need = B.stream_required_information(a - 3 * se, rho, nu / P)
                    viol += int(need > info)
                    ceiling = B.stream_power_ceiling(tabs[(kind, m)][min(lag, 1200)], grid, nu / P)
                    viol_np += int(a - 3 * se - rho > ceiling)
                    table.append({"kind": kind, "m": m, "method": name, "q1": q, "L": L, "a": a, "rho": rho, "nu": nu, "required": need, "I": info, "ceiling": ceiling,
                                  "coverage": float(np.mean([r[f"cov_{L}"] for r in rs]))})
    nontrivial = [x for x in table if x["required"] > 0 and x["I"] > 0]
    out["methods"] = {"comparisons": len(table), "violations": viol, "violations_neyman_pearson": viol_np, "nontrivial_neyman_pearson": sum(x["ceiling"] < 0.999 for x in table), "nontrivial": len(nontrivial), "max_required_over_I": max((x["required"] / x["I"] for x in nontrivial), default=None), "table": table}
    json.dump(out, open(RESULTS_DIR / "summaries" / "theory_stream.json", "w"), indent=1)
    print(json.dumps({"violations_np": viol_np, "nontrivial_np": out["methods"]["nontrivial_neyman_pearson"], "violations": viol, "comparisons": len(table), "nontrivial": len(nontrivial), "max": out["methods"]["max_required_over_I"]}))
    for r in rows:
        print(r["T"], r["kind"], r["m"], r["target"], "power", round(r["power"], 3), "lower", r["L_lower"], "kl", r["L_lower_kl"], "detector", r["L_detector"])
    for k, v in det.items():
        print(k, "FA", v["false_alarm"], "pre", v["pre_change_alarm"], {L: (round(x["observed"], 2), round(x["guarantee"], 2)) for L, x in v["power"].items()})


if __name__ == "__main__":
    main()
