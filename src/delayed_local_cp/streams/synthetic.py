"""Controlled synthetic streams (PROJECT_BRIEF §10.1).

Inputs: ``x0 ~ U(cell)`` where the cell (region) is drawn from time-varying
weights; extra coordinates are U(0, 1) nuisance features. The declared
partition is ``K`` equal-width cells of ``x0``.

Labels: ``Y = amp * sin(2 pi x0) + mean_shift(t, r) + sigma_r * scale(t, r) * eps``
with ``eps ~ N(0, 1)`` (or Student-t with ``noise_df``).

The base predictor is fitted once on a separate pre-shift training sample and
then frozen, so all methods share it.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from ..delays.models import sample_delays
from ..shifts.schedules import profile, region_mask
from .stream import Stream, apply_cqr


def _region_weights(cfg: dict, t: np.ndarray, k: int) -> np.ndarray:
    w0 = np.asarray(cfg.get("region_weights") or [1.0] * k, dtype=float)
    w = np.tile(w0, (len(t), 1))
    for s in cfg.get("shifts", []):
        if s["kind"] != "covariate":
            continue
        g = profile(t, s)
        cols = np.arange(k) if s.get("regions", "all") == "all" else np.asarray(s["regions"])
        w[:, cols] *= (1.0 + float(s["size"]) * g)[:, None]
    return w / w.sum(axis=1, keepdims=True)


def _sample_inputs(cfg: dict, t: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    k = int(cfg["n_regions"])
    w = _region_weights(cfg, t, k)
    cum = w.cumsum(axis=1)
    u = rng.random(len(t))
    regions = np.minimum((u[:, None] > cum).sum(axis=1), k - 1).astype(np.int64)
    x0 = (regions + rng.random(len(t))) / k
    d_extra = int(cfg.get("n_nuisance", 0))
    x = np.column_stack([x0] + [rng.random(len(t)) for _ in range(d_extra)])
    return x, regions


def _labels(cfg: dict, t: np.ndarray, x: np.ndarray, regions: np.ndarray,
            rng: np.random.Generator, apply_shifts: bool = True) -> np.ndarray:
    k = int(cfg["n_regions"])
    amp = float(cfg.get("amplitude", 1.0))
    sigma = np.asarray(cfg.get("noise_per_region") or [cfg.get("noise", 0.3)] * k, dtype=float)[regions]
    mean = amp * np.sin(2 * np.pi * x[:, 0])
    scale = np.ones(len(t))
    if apply_shifts:
        for s in cfg.get("shifts", []):
            if s["kind"] == "covariate":
                continue
            g = profile(t, s) * region_mask(regions, s)
            if s["kind"] == "mean":
                size = np.asarray(s["size_per_region"], dtype=float)[regions] if "size_per_region" in s else float(s["size"])
                mean = mean + size * g
            elif s["kind"] == "variance":
                scale = scale * (1.0 + float(s["size"]) * g)
            else:
                raise ValueError(f"unknown shift kind {s['kind']!r}")
    df = cfg.get("noise_df")
    eps = rng.standard_t(float(df), size=len(t)) if df else rng.standard_normal(len(t))
    return mean + sigma * scale * eps


def make_synthetic_stream(dataset_cfg: dict, delay_cfg: dict, shifts: list[dict], seed: int) -> Stream:
    cfg = dict(dataset_cfg)
    cfg["shifts"] = list(shifts or [])
    n = int(cfg["n_rounds"])
    ss = np.random.SeedSequence([int(seed), 20261104])
    rng_x, rng_y, rng_d, rng_train = (np.random.default_rng(s) for s in ss.spawn(4))

    t = np.arange(n)
    x, regions = _sample_inputs(cfg, t, rng_x)
    y = _labels(cfg, t, x, regions, rng_y)
    delay = sample_delays(delay_cfg, x, regions, y, rng_d)

    # Frozen base predictor trained on a separate pre-shift sample.
    n_train = int(cfg.get("n_train", 2000))
    t_tr = np.zeros(n_train)
    no_shift = {k: v for k, v in cfg.items() if k != "shifts"}
    x_tr, r_tr = _sample_inputs(no_shift, t_tr, rng_train)
    y_tr = _labels(no_shift, t_tr, x_tr, r_tr, rng_train, apply_shifts=False)
    model = HistGradientBoostingRegressor(max_iter=int(cfg.get("model_iters", 100)), random_state=0)
    model.fit(x_tr, y_tr)
    yhat = model.predict(x)
    # Score scale from a further pre-shift holdout (known before round 0); the draws come last,
    # so the stream, delays and fitted model are unchanged by this step.
    x_ho, r_ho = _sample_inputs(no_shift, np.zeros(1000), rng_train)
    y_ho = _labels(no_shift, np.zeros(1000), x_ho, r_ho, rng_train, apply_shifts=False)
    score_scale = float(np.median(np.abs(y_ho - model.predict(x_ho))))

    shift_starts = sorted({int(s["start"]) for s in cfg["shifts"]})
    meta = {"name": cfg.get("name", "synthetic"), "shift_starts": shift_starts,
            "shift_regions": [s.get("regions", "all") for s in cfg["shifts"]],
            "score_scale": score_scale}
    stream = Stream(x=x, regions=regions, y=y, delay=delay, yhat=yhat, n_regions=int(cfg["n_regions"]), meta=meta)
    if cfg.get("cqr"):
        a = 1 - float(cfg.get("cqr_coverage", 0.9))
        q = {}
        for name, level in (("lo", a / 2), ("hi", 1 - a / 2)):
            q[name] = HistGradientBoostingRegressor(loss="quantile", quantile=level, max_iter=int(cfg.get("model_iters", 100)), random_state=0).fit(x_tr, y_tr)
        return apply_cqr(stream, q["lo"].predict(x), q["hi"].predict(x), score_scale)
    return stream
