"""Evaluation quantities from PROJECT_BRIEF §7.

All coverage metrics here are computed on *issued* rounds using the true label,
whether or not that label has arrived by the end of the stream. This is the
evaluator's view, not the learner's. ``arrived_mask`` lets a caller restrict
to rounds whose label arrived (the learner's view).

Window conventions (declared before final runs, see reports/decisions.md):

* Region windows are consecutive, non-overlapping blocks of ``w`` issued rounds
  *within one region*, so every window carries the same amount of evidence.
  Incomplete trailing blocks are dropped.
* Rolling coverage at a round is the coverage over the last ``w`` issued rounds
  of the same region (NaN until ``w`` rounds exist).
"""

from __future__ import annotations

import math

import numpy as np


def covered(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return (y >= lo) & (y <= hi)


def marginal_coverage(cov: np.ndarray, mask: np.ndarray | None = None) -> float:
    c = cov if mask is None else cov[mask]
    return float(np.mean(c)) if len(c) else math.nan


def regional_coverage(cov: np.ndarray, regions: np.ndarray, n_regions: int,
                      mask: np.ndarray | None = None) -> list[float]:
    out = []
    for r in range(n_regions):
        m = regions == r if mask is None else (regions == r) & mask
        out.append(float(np.mean(cov[m])) if m.any() else math.nan)
    return out


def region_window_coverage(cov: np.ndarray, regions: np.ndarray, n_regions: int, w: int) -> list[np.ndarray]:
    """Per region, the coverage of each complete block of ``w`` region rounds."""
    out = []
    for r in range(n_regions):
        c = cov[regions == r].astype(float)
        m = len(c) // w
        out.append(c[: m * w].reshape(m, w).mean(axis=1) if m else np.empty(0))
    return out


def worst_region_window_coverage(cov, regions, n_regions, w) -> float:
    blocks = [b for b in region_window_coverage(cov, regions, n_regions, w) if len(b)]
    return float(min(b.min() for b in blocks)) if blocks else math.nan


def rolling_region_coverage(cov: np.ndarray, regions: np.ndarray, n_regions: int, w: int) -> np.ndarray:
    """Array (n,) with the rolling coverage of the round's own region (last w region rounds)."""
    out = np.full(len(cov), np.nan)
    for r in range(n_regions):
        idx = np.flatnonzero(regions == r)
        if len(idx) < w:
            continue
        c = np.concatenate([[0.0], np.cumsum(cov[idx].astype(float))])
        roll = (c[w:] - c[:-w]) / w
        out[idx[w - 1:]] = roll
    return out


def recovery_time(cov: np.ndarray, regions: np.ndarray, region: int, shift_start: int,
                  target: float, w: int) -> float:
    """Rounds from ``shift_start`` until rolling coverage over the last ``w``
    post-shift issued rounds of ``region`` first reaches ``target``.

    Only rounds at or after the shift enter the window. Returns ``inf`` if the
    target is never reached (censored at the end of the stream).
    """
    idx = np.flatnonzero((regions == region) & (np.arange(len(cov)) >= shift_start))
    if len(idx) < w:
        return math.inf
    c = np.concatenate([[0.0], np.cumsum(cov[idx].astype(float))])
    roll = (c[w:] - c[:-w]) / w
    hit = np.flatnonzero(roll >= target - 1e-12)
    if not len(hit):
        return math.inf
    return float(idx[hit[0] + w - 1] - shift_start)


def time_below_target(cov, regions, n_regions, target, w, start: int = 0) -> list[float]:
    """Per region, the fraction of its rounds (from ``start``) whose rolling coverage is below target."""
    roll = rolling_region_coverage(cov, regions, n_regions, w)
    t = np.arange(len(cov))
    out = []
    for r in range(n_regions):
        m = (regions == r) & (t >= start) & ~np.isnan(roll)
        out.append(float(np.mean(roll[m] < target - 1e-12)) if m.any() else math.nan)
    return out


def width_stats(lo: np.ndarray, hi: np.ndarray) -> dict:
    w = hi - lo
    finite = np.isfinite(w)
    return {
        "mean_width": float(np.mean(w[finite])) if finite.any() else math.nan,
        "median_width": float(np.median(w[finite])) if finite.any() else math.nan,
        "frac_infinite": float(1.0 - finite.mean()) if len(w) else math.nan,
    }


def interval_score(y, lo, hi, alpha) -> float:
    """Mean Gneiting-Raftery interval score; inf if any interval is unbounded."""
    s = (hi - lo) + (2.0 / alpha) * np.maximum(lo - y, 0.0) + (2.0 / alpha) * np.maximum(y - hi, 0.0)
    return float(np.mean(s))


def backlog_and_age(regions: np.ndarray, arrival: np.ndarray, n_regions: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-round pending counts and information age, both shape (n, K).

    At round t (before issuing C_t): backlog[t, r] = #{i < t in r: arrival_i > t};
    age[t, r] = t - max{i in r: arrival_i <= t}, or t + 1 if no label of r has arrived.
    """
    n = len(regions)
    backlog = np.zeros((n, n_regions), dtype=np.int64)
    age = np.zeros((n, n_regions), dtype=np.int64)
    for r in range(n_regions):
        idx = np.flatnonzero(regions == r)
        arr = arrival[idx]
        # number issued strictly before t
        issued_before = np.searchsorted(idx, np.arange(n), side="left")
        # number of those that arrived by t: arrivals <= t among them
        order = np.argsort(arr, kind="stable")
        arr_sorted = arr[order]
        arrived_by = np.searchsorted(arr_sorted, np.arange(n), side="right")
        backlog[:, r] = issued_before - arrived_by
        # latest issued index among arrived labels by t
        latest = np.full(n, -1, dtype=np.int64)
        if len(idx):
            run_max = np.maximum.accumulate(idx[order])
            has = arrived_by > 0
            latest[has] = run_max[arrived_by[has] - 1]
        t = np.arange(n)
        age[:, r] = np.where(latest >= 0, t - latest, t + 1)
    return backlog, age


def summarize(y, lo, hi, regions, arrival, n_regions, alpha, w, shift_starts=(), shift_regions=()) -> dict:
    cov = covered(y, lo, hi)
    n = len(y)
    arrived = arrival < n
    target = 1.0 - alpha
    out = {
        "n_rounds": n,
        "alpha": alpha,
        "window": w,
        "marginal_coverage": marginal_coverage(cov),
        "marginal_coverage_arrived": marginal_coverage(cov, arrived),
        "regional_coverage": regional_coverage(cov, regions, n_regions),
        "worst_region_window_coverage": worst_region_window_coverage(cov, regions, n_regions, w),
        "interval_score": interval_score(y, lo, hi, alpha),
        "time_below_target": time_below_target(cov, regions, n_regions, target, w),
        "frac_labels_arrived": float(arrived.mean()),
    }
    out.update(width_stats(lo, hi))
    rec = []
    for t0, regs in zip(shift_starts, shift_regions):
        rlist = range(n_regions) if regs == "all" else regs
        for r in rlist:
            rec.append({"shift_start": int(t0), "region": int(r),
                        "recovery_time": recovery_time(cov, regions, int(r), int(t0), target, w)})
    out["recovery"] = rec
    backlog, age = backlog_and_age(regions, arrival, n_regions)
    out["mean_backlog"] = backlog.sum(axis=1).mean().item()
    out["mean_info_age_by_region"] = age.mean(axis=0).tolist()
    return out
