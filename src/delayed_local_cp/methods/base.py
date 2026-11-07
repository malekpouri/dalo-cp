"""Common interface for online conformal methods under delayed feedback.

A method sees, in order, for every round ``t``:

1. ``on_arrival(...)`` for each label that arrives at the start of round ``t``.
   It receives the half-width the method itself issued at that label's round,
   so updates use ``err_i = 1{Y_i not in C_i(X_i)}`` as the brief requires.
2. ``issue(t, x_t, region_t, yhat_t, ctx)`` which returns the half-width ``q_t``
   of the interval ``[yhat_t - q_t, yhat_t + q_t]``.

Methods never receive a label any other way. Scores are absolute residuals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class RoundContext:
    """Information available at prediction time besides arrived labels."""

    t: int
    pending_counts: np.ndarray   # pending labels per region (inputs are known)
    n_pending: int


def conformal_quantile(scores: np.ndarray, level: float) -> float:
    """Split-conformal quantile: the ceil((n+1)*level)-th smallest score.

    Returns +inf when that rank exceeds n (including n = 0) and 0 when the
    level is <= 0 (ACI's empty-set case is represented as a zero-width interval).
    """
    n = len(scores)
    if level <= 0:
        return 0.0
    k = math.ceil((n + 1) * level)
    if k > n:
        return math.inf
    return float(np.partition(np.asarray(scores, dtype=float), k - 1)[k - 1])


def projected_quantile(scores: np.ndarray, level: float) -> float:
    """conformal_quantile, except that a rank above n gives the largest score (finite once a label has arrived)."""
    q = conformal_quantile(scores, level)
    return float(np.max(scores)) if math.isinf(q) and len(scores) else q


class OnlineMethod:
    name = "base"

    def __init__(self, cfg: dict, alpha: float, n_regions: int, meta: dict | None = None):
        self.cfg = dict(cfg)
        self.meta = dict(meta or {})   # stream facts fixed before round 0 (for example score_scale)
        self.alpha = float(alpha)
        self.n_regions = int(n_regions)
        self._issued: dict[int, tuple[float, int]] = {}
        self.quantile = projected_quantile if self.cfg.get("project") else conformal_quantile

    # Public API used by the experiment loop -------------------------------
    def issue(self, t: int, x: np.ndarray, region: int, yhat: float, ctx: RoundContext) -> float:
        q = float(self.threshold(t, x, region, yhat, ctx))
        self._issued[t] = (q, int(region))
        return q

    def on_arrival(self, i: int, x: np.ndarray, region: int, yhat: float, y: float, t: int) -> None:
        q_i, _ = self._issued.pop(i)
        score = abs(y - yhat)
        err = 1.0 if score > q_i else 0.0
        self.update(i, x, int(region), score, err, t)

    # To implement -----------------------------------------------------------
    def threshold(self, t: int, x: np.ndarray, region: int, yhat: float, ctx: RoundContext) -> float:
        raise NotImplementedError

    def update(self, i: int, x: np.ndarray, region: int, score: float, err: float, t: int) -> None:
        raise NotImplementedError

    def diagnostics(self) -> dict:
        return {}
