"""Remaining baselines from PROJECT_BRIEF §11. All receive feedback only when labels arrive.

Sources and adaptations (details and reasons in reports/decisions.md):

* ``delayed_aci``: El Halabi & Brandt, arXiv:2609.07251, Algorithm 2: the new level is built from the level
  *issued* at the label's round, ``alpha* = alpha_i + gamma (alpha - err_i)``. They treat a fixed delay; with
  random delays the same rule is applied at each arrival.
* ``dtaci``: Gibbs & Candes, JMLR 2024, Algorithm 2 with their gamma grid, sigma = 1/(2*500) and eta formula.
  Rank statistic beta_i and the experts' errors use the score window as it was at round i.
* ``olcp``: Lai & Raskutti, arXiv:2605.05497, Algorithm 1 with their exponential kernel, per-window
  standardization, Silverman-style h0 and projection of the level onto [0, 1].
* ``static_lcp``: fixed-level localized CP on a frozen calibration set (first ``n_cal`` arrived labels), weighted
  quantile with a point mass at +inf. Guan's recalibrated level is not implemented.
* ``ipw_aci``: I-ACI update as described in Wang, Zecchin & Simeone, arXiv:2503.10345, eq. (6). A label is treated
  as observed if it arrives within ``deadline`` rounds and otherwise as missing; p is P(D <= deadline | region),
  estimated online from matured rounds.
"""

from __future__ import annotations

import heapq
import math
from collections import deque

import numpy as np

from .base import OnlineMethod, conformal_quantile


def weighted_quantile(scores: np.ndarray, weights: np.ndarray, level: float, inf_mass: float = 0.0) -> float:
    """Smallest s with sum(w_i 1{s_i <= s}) >= level * (sum w_i + inf_mass); +inf if no such score."""
    if level <= 0:
        return 0.0
    if len(scores) == 0:
        return math.inf
    order = np.argsort(scores, kind="stable")
    cum = np.cumsum(weights[order])
    total = cum[-1] + inf_mass
    if total <= 0:
        return math.inf
    idx = int(np.searchsorted(cum, level * total - 1e-12, side="left"))
    if idx >= len(scores):
        return math.inf
    return float(scores[order[idx]])


class DelayedACI(OnlineMethod):
    name = "delayed_aci"

    def __init__(self, cfg, alpha, n_regions, meta=None):
        super().__init__(cfg, alpha, n_regions, meta)
        self.gamma = float(cfg.get("gamma", 0.005))
        self.scores: deque = deque(maxlen=int(cfg.get("window", 500)))
        self.alpha_star = self.alpha
        self.level_at: dict[int, float] = {}

    def threshold(self, t, x, region, yhat, ctx):
        self.level_at[t] = self.alpha_star
        return self.quantile(np.fromiter(self.scores, float), 1.0 - self.alpha_star)

    def update(self, i, x, region, score, err, t):
        self.alpha_star = self.level_at.pop(i) + self.gamma * (self.alpha - err)
        self.scores.append(score)


class DtACI(OnlineMethod):
    name = "dtaci"
    GRID = (0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128)

    def __init__(self, cfg, alpha, n_regions, meta=None):
        super().__init__(cfg, alpha, n_regions, meta)
        self.gammas = np.asarray(cfg.get("gammas", self.GRID), dtype=float)
        k = len(self.gammas)
        interval = int(cfg.get("interval", 500))
        self.sigma = 1.0 / (2 * interval)
        a = self.alpha
        self.eta = math.sqrt(3 / interval) * math.sqrt((math.log(2 * k * interval) + 1) / ((1 - a) ** 2 * a ** 2))
        self.w = np.ones(k)
        self.levels = np.full(k, a)
        self.scores: deque = deque(maxlen=int(cfg.get("window", 500)))
        self.snap: dict[int, np.ndarray] = {}

    def threshold(self, t, x, region, yhat, ctx):
        arr = np.fromiter(self.scores, float)
        self.snap[t] = arr
        p = self.w / self.w.sum()
        return self.quantile(arr, 1.0 - float(p @ self.levels))

    def update(self, i, x, region, score, err, t):
        arr = self.snap.pop(i)
        beta = float(np.mean(arr >= score)) if len(arr) else 1.0
        d = beta - self.levels
        loss = self.alpha * d - np.minimum(0.0, d)
        wbar = self.w * np.exp(-self.eta * loss)
        total = wbar.sum()
        if not np.isfinite(total) or total <= 0:   # numerical underflow: restart the weights
            wbar, total = np.ones_like(wbar), float(len(wbar))
        self.w = ((1 - self.sigma) * wbar + total * self.sigma / len(wbar)) / total
        errs = np.array([1.0 if score > self.quantile(arr, 1.0 - a) else 0.0 for a in self.levels])
        self.levels = self.levels + self.gammas * (self.alpha - errs)
        self.scores.append(score)


class OLCP(OnlineMethod):
    name = "olcp"

    def __init__(self, cfg, alpha, n_regions, meta=None):
        super().__init__(cfg, alpha, n_regions, meta)
        self.gamma = float(cfg.get("gamma", 0.005))
        self.R = int(cfg.get("window", 200))
        self.bw_mult = float(cfg.get("bandwidth_mult", 1.0))
        self.scores: deque = deque(maxlen=self.R)
        self.xs: deque = deque(maxlen=self.R)
        self.alpha_t = self.alpha

    def _h0(self, d: int) -> float:
        return (4 / (d + 2)) ** (1 / (d + 4)) * self.R ** (-1 / (d + 4)) * math.sqrt(d)

    def threshold(self, t, x, region, yhat, ctx):
        if not self.scores:
            return math.inf
        xs = np.asarray(self.xs, dtype=float)
        s = np.fromiter(self.scores, float)
        mu, sd = xs.mean(axis=0), xs.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        dist = np.linalg.norm((xs - mu) / sd - (np.asarray(x) - mu) / sd, axis=1)
        w = np.exp(-dist / (self.bw_mult * self._h0(xs.shape[1])))
        if w.sum() <= 1e-300:   # the paper's fallback to uniform weights
            w = np.ones_like(w)
        return weighted_quantile(s, w, 1.0 - self.alpha_t)

    def update(self, i, x, region, score, err, t):
        self.alpha_t = min(1.0, max(0.0, self.alpha_t + self.gamma * (self.alpha - err)))
        self.scores.append(score)
        self.xs.append(np.asarray(x, dtype=float))

    def diagnostics(self):
        return {"alpha_t": self.alpha_t}


class StaticLCP(OnlineMethod):
    name = "static_lcp"

    def __init__(self, cfg, alpha, n_regions, meta=None):
        super().__init__(cfg, alpha, n_regions, meta)
        self.n_cal = int(cfg.get("n_cal", 500))
        self.h = float(cfg.get("bandwidth", 0.2))
        self.xs: list = []
        self.scores: list = []

    def threshold(self, t, x, region, yhat, ctx):
        if not self.scores:
            return math.inf
        xs = np.asarray(self.xs, dtype=float)
        dist = np.linalg.norm(xs - np.asarray(x), axis=1)
        w = np.exp(-dist / self.h)
        return weighted_quantile(np.asarray(self.scores), w, 1.0 - self.alpha, inf_mass=1.0)

    def update(self, i, x, region, score, err, t):
        if len(self.scores) < self.n_cal:
            self.scores.append(score)
            self.xs.append(np.asarray(x, dtype=float))


class IPWACI(OnlineMethod):
    name = "ipw_aci"

    def __init__(self, cfg, alpha, n_regions, meta=None):
        super().__init__(cfg, alpha, n_regions, meta)
        scale = float(self.meta.get("score_scale", 1.0))
        self.eta = float(cfg.get("eta_rel", 0.05)) * scale
        self.deadline = int(cfg.get("deadline", 10))
        self.p_floor = float(cfg.get("p_floor", 0.05))
        self.r = 2.44 * scale   # 90% quantile of |N(0, s)| expressed through its median
        self.issued_q: deque = deque()   # (mature round, region) for every issued round
        self.hits_q: list = []           # (mature round, region) for labels arriving by the deadline
        self.n_mature = np.zeros(n_regions)
        self.n_hit = np.zeros(n_regions)

    def _mature(self, t):
        while self.issued_q and self.issued_q[0][0] <= t:
            _, r = self.issued_q.popleft()
            self.n_mature[r] += 1
        while self.hits_q and self.hits_q[0][0] <= t:
            _, r = heapq.heappop(self.hits_q)
            self.n_hit[r] += 1

    def threshold(self, t, x, region, yhat, ctx):
        self._mature(t)
        self.issued_q.append((t + self.deadline, region))
        return max(self.r, 0.0)

    def update(self, i, x, region, score, err, t):
        if t - i > self.deadline:
            return   # treated as missing
        heapq.heappush(self.hits_q, (i + self.deadline, region))
        self._mature(t)
        p = self.n_hit[region] / self.n_mature[region] if self.n_mature[region] >= 20 else 1.0
        self.r += self.eta * (err - self.alpha) / max(p, self.p_floor)


class WeightedCP(OnlineMethod):
    """Weighted CP for covariate shift (Tibshirani et al., 2019) with an estimated likelihood ratio.

    A logistic model separates arrived calibration inputs from the most recent issued inputs (known even when labels are
    pending) and is refitted every ``refit_every`` rounds. The ratio is clipped to [1/clip, clip] for stability.
    """

    name = "weighted_cp"

    def __init__(self, cfg, alpha, n_regions, meta=None):
        super().__init__(cfg, alpha, n_regions, meta)
        self.W = int(cfg.get("window", 500))
        self.recent_n = int(cfg.get("recent", 500))
        self.refit_every = int(cfg.get("refit_every", 200))
        self.clip = float(cfg.get("clip", 10.0))
        self.scores: deque = deque(maxlen=self.W)
        self.xs: deque = deque(maxlen=self.W)
        self.recent: deque = deque(maxlen=self.recent_n)
        self.coef = None
        self.intercept = 0.0
        self.mu = 0.0
        self.sd = 1.0

    def _ratio(self, x: np.ndarray) -> np.ndarray:
        if self.coef is None:
            return np.ones(len(x))
        logit = ((x - self.mu) / self.sd) @ self.coef + self.intercept
        return np.clip(np.exp(logit), 1 / self.clip, self.clip)

    def _refit(self) -> None:
        from sklearn.linear_model import LogisticRegression

        a, b = np.asarray(self.xs, dtype=float), np.asarray(self.recent, dtype=float)
        if len(a) < 50 or len(b) < 50:
            return
        z = np.vstack([a, b])
        self.mu, sd = z.mean(0), z.std(0)
        self.sd = np.where(sd > 0, sd, 1.0)
        y = np.r_[np.zeros(len(a)), np.ones(len(b))]
        m = LogisticRegression(C=1.0, max_iter=200).fit((z - self.mu) / self.sd, y)
        self.coef, self.intercept = m.coef_[0], float(m.intercept_[0] - np.log(len(b) / len(a)))

    def threshold(self, t, x, region, yhat, ctx):
        self.recent.append(np.asarray(x, dtype=float))
        if t % self.refit_every == 0:
            self._refit()
        if not self.scores:
            return math.inf
        w = self._ratio(np.asarray(self.xs, dtype=float))
        wt = float(self._ratio(np.asarray(x, dtype=float)[None, :])[0])
        return weighted_quantile(np.fromiter(self.scores, float), w, 1.0 - self.alpha, inf_mass=wt)

    def update(self, i, x, region, score, err, t):
        self.scores.append(score)
        self.xs.append(np.asarray(x, dtype=float))
