"""First baselines used by the smoke test and the M2 feasibility study.

All of them update only when a label arrives (arrival-time updates).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .base import OnlineMethod, RoundContext, conformal_quantile


class RollingSplitCP(OnlineMethod):
    """Split CP on the ``window`` most recently arrived scores (global)."""

    name = "split_cp"

    def __init__(self, cfg, alpha, n_regions, meta=None):
        super().__init__(cfg, alpha, n_regions, meta)
        self.scores: deque = deque(maxlen=int(cfg.get("window", 500)))

    def threshold(self, t, x, region, yhat, ctx: RoundContext):
        return conformal_quantile(np.fromiter(self.scores, float), 1.0 - self.alpha)

    def update(self, i, x, region, score, err, t):
        self.scores.append(score)


class ACI(OnlineMethod):
    """Adaptive conformal inference (Gibbs and Candes, 2021) with arrival-time updates.

    ``alpha_{k+1} = alpha_k + gamma * (alpha - err_i)`` is applied once per
    arrived label, where ``err_i`` uses the threshold issued at round ``i``.
    """

    name = "aci"

    def __init__(self, cfg, alpha, n_regions, meta=None):
        super().__init__(cfg, alpha, n_regions, meta)
        self.gamma = float(cfg.get("gamma", 0.005))
        self.scores: deque = deque(maxlen=int(cfg.get("window", 500)))
        self.alpha_t = self.alpha

    def threshold(self, t, x, region, yhat, ctx):
        return self.quantile(np.fromiter(self.scores, float), 1.0 - self.alpha_t)

    def update(self, i, x, region, score, err, t):
        self.alpha_t += self.gamma * (self.alpha - err)
        self.scores.append(score)

    def diagnostics(self):
        return {"alpha_t": self.alpha_t}


class RegionalACI(OnlineMethod):
    """ACI run separately in each declared region, with per-region score windows."""

    name = "regional_aci"

    def __init__(self, cfg, alpha, n_regions, meta=None):
        super().__init__(cfg, alpha, n_regions, meta)
        self.gamma = float(cfg.get("gamma", 0.005))
        w = int(cfg.get("window", 200))
        self.scores = [deque(maxlen=w) for _ in range(n_regions)]
        self.alpha_t = np.full(n_regions, self.alpha)

    def threshold(self, t, x, region, yhat, ctx):
        return self.quantile(np.fromiter(self.scores[region], float), 1.0 - self.alpha_t[region])

    def update(self, i, x, region, score, err, t):
        self.alpha_t[region] += self.gamma * (self.alpha - err)
        self.scores[region].append(score)

    def diagnostics(self):
        return {"alpha_t": self.alpha_t.tolist()}


def _registry() -> dict:
    from .dalo import DaloCP
    from .online import DelayedACI, DtACI, IPWACI, OLCP, StaticLCP, WeightedCP

    return {cls.name: cls for cls in (RollingSplitCP, ACI, RegionalACI, DelayedACI, DtACI, OLCP, StaticLCP, IPWACI, WeightedCP, DaloCP)}


def build_method(cfg: dict, alpha: float, n_regions: int, meta: dict | None = None) -> OnlineMethod:
    methods = _registry()
    try:
        cls = methods[cfg["type"]]
    except KeyError as exc:
        raise ValueError(f"unknown method type {cfg.get('type')!r}; known: {sorted(methods)}") from exc
    return cls(cfg, alpha, n_regions, meta)
