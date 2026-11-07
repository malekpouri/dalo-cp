"""Delay mechanisms. Every sampler returns integer delays D >= 1.

Config keys (``type`` selects the mechanism):

* ``fixed``        : ``value`` (scalar) or ``per_region`` (list) of delays.
* ``geometric``    : mean delay ``mean`` or ``per_region`` means; D ~ 1 + Geom.
* ``pareto``       : heavy-tailed; ``scale`` (or ``per_region``) and ``shape``.
* ``x_continuous`` : geometric with mean ``min_mean + (max_mean - min_mean) * x0**power``
                     where ``x0`` is the first input coordinate in [0, 1].
* ``y_dependent``  : robustness case; geometric with mean that also grows with
                     the standardized label: ``base`` delay config times
                     ``exp(y_coef * z(y))``.

``d_max`` (optional) truncates delays to ``[1, d_max]``.
"""

from __future__ import annotations

import numpy as np


def _per_region(cfg: dict, key: str, regions: np.ndarray) -> np.ndarray:
    if "per_region" in cfg:
        vals = np.asarray(cfg["per_region"], dtype=float)
        if regions.max(initial=0) >= len(vals):
            raise ValueError("per_region list shorter than number of regions")
        return vals[regions]
    return np.full(len(regions), float(cfg[key]))


def _geometric_with_mean(rng: np.random.Generator, mean: np.ndarray) -> np.ndarray:
    mean = np.maximum(np.asarray(mean, dtype=float), 1.0)
    return rng.geometric(1.0 / mean).astype(np.int64)  # support {1, 2, ...}, E = mean


def sample_delays(cfg: dict, x: np.ndarray, regions: np.ndarray, y: np.ndarray,
                  rng: np.random.Generator) -> np.ndarray:
    kind = cfg["type"]
    n = len(regions)
    if kind == "fixed":
        d = np.rint(_per_region(cfg, "value", regions)).astype(np.int64)
    elif kind == "geometric":
        d = _geometric_with_mean(rng, _per_region(cfg, "mean", regions))
    elif kind == "pareto":
        scale = _per_region(cfg, "scale", regions)
        d = np.ceil(scale * (1.0 + rng.pareto(float(cfg.get("shape", 1.5)), size=n))).astype(np.int64)
    elif kind == "x_continuous":
        x0 = np.clip(x[:, 0], 0.0, 1.0)
        mean = cfg["min_mean"] + (cfg["max_mean"] - cfg["min_mean"]) * x0 ** float(cfg.get("power", 1.0))
        d = _geometric_with_mean(rng, mean)
    elif kind == "y_dependent":
        base = dict(cfg["base"])
        if base["type"] != "geometric":
            raise ValueError("y_dependent supports a geometric base")
        z = (y - y.mean()) / (y.std() + 1e-12)
        mean = _per_region(base, "mean", regions) * np.exp(float(cfg["y_coef"]) * z)
        d = _geometric_with_mean(rng, mean)
    else:
        raise ValueError(f"unknown delay type {kind!r}")
    d = np.maximum(d, 1)
    if cfg.get("d_max") is not None:
        d = np.minimum(d, int(cfg["d_max"]))
    return d
