"""DALO-CP: delay-aware localized online conformal prediction (PROJECT_BRIEF §8).

Threshold issued at round t for an input in region r::

    q_t = base_r(t) + c_r(t) + borrow_r(t) + guard_r(t),   clipped at 0

* Regional state and arrival-time updates: ``c_r += eta (err_i - alpha)`` when label i of region r arrives, with
  ``err_i`` computed from the threshold issued at round i. ``eta = eta_rel * score_scale``, fixed in advance.
  ``eta_rel: 0`` switches the correction off.
* Age-aware calibration: ``base_r`` is a weighted (1 - alpha) quantile of the last ``window`` arrived scores of r.
  The weight is ``exp(-age / age_tau)``, with age counted in issued rounds of region r since the label's round.
  ``age_tau: null`` gives uniform weights. If the rank exceeds the mass, the largest arrived score is used. With ``arrival_ipw: true``, weights are divided by the estimated
  probability that a label of that age has arrived, floored at ``ipw_floor``.
* Borrowing (``borrow.on``): adds a bounded, one-sided transfer of the upward change seen in a neighbouring region's
  scores, in proportion to how much fresher the neighbour's feedback is. A region's feedback fraction
  ``phi_r`` is the fresh-label mass actually arrived divided by the mass if every label had arrived.
* Pending guard (``guard.on``): adds ``kappa_rel * score_scale * (1 - phi_r)``, a bounded term computed only from
  arrival timing and known pending inputs.

Everything in q_t is computed from arrived labels and known inputs, so it is predictable.
"""

from __future__ import annotations

import math

import numpy as np

from .online import weighted_quantile
from .base import OnlineMethod


class DaloCP(OnlineMethod):
    name = "dalo_cp"

    def __init__(self, cfg, alpha, n_regions, meta=None):
        super().__init__(cfg, alpha, n_regions, meta)
        self.scale = float(self.meta.get("score_scale", 1.0))
        self.eta = float(cfg.get("eta_rel", 0.05)) * self.scale
        self.W = int(cfg.get("window", 250))
        self.age_tau = cfg.get("age_tau")
        self.arrival_ipw = bool(cfg.get("arrival_ipw", False))
        self.ipw_floor = float(cfg.get("ipw_floor", 0.1))
        self.borrow = dict(cfg.get("borrow") or {})
        self.guard = dict(cfg.get("guard") or {})
        self.tau_fresh = float(cfg.get("tau_fresh", 25))
        k = n_regions
        self.c = np.zeros(k)
        self.n_issued = np.zeros(k, dtype=np.int64)
        self.buf_s = np.zeros((k, self.W))
        self.buf_c = np.zeros((k, self.W), dtype=np.int64)
        self.n_arrived = np.zeros(k, dtype=np.int64)
        self.buf_lag = np.zeros((k, self.W), dtype=np.int64)
        self.ctr_at: dict[int, int] = {}

    # ---- buffers -------------------------------------------------------------
    def _arrays(self, r: int):
        m = min(self.n_arrived[r], self.W)
        return self.buf_s[r, :m], self.n_issued[r] - self.buf_c[r, :m], self.buf_lag[r, :m]

    def _arrival_prob(self, ages: np.ndarray, lags: np.ndarray) -> np.ndarray:
        lags_sorted = np.sort(lags)
        return np.searchsorted(lags_sorted, ages, side="right") / max(len(lags_sorted), 1)

    def _weights(self, ages, lags, tau) -> np.ndarray:
        w = np.ones(len(ages)) if tau is None else np.exp(-(ages - ages.min()) / float(tau))
        if self.arrival_ipw and len(ages):
            w = w / np.maximum(self._arrival_prob(ages, lags), self.ipw_floor)
        return w

    def _base(self, r: int, tau) -> float:
        s, ages, lags = self._arrays(r)
        if len(s) == 0:
            return math.inf
        w = self._weights(ages, lags, tau)
        # The +inf point has age 0, so its mass is the age weight of the freshest arrived label. The arrival
        # correction is left out here: boosting it would make the base infinite whenever fresh labels are rare.
        inf_mass = 1.0
        q = weighted_quantile(s, w, 1.0 - self.alpha, inf_mass=inf_mass)
        # Finite base once any label has arrived; the correction c_r supplies the coverage feedback.
        return q if math.isfinite(q) else float(s.max())

    def _phi(self, r: int) -> float:
        """Share of the fresh-label mass that has arrived, in [0, 1]."""
        _, ages, _ = self._arrays(r)
        if len(ages) == 0:
            return 0.0
        return float(min(1.0, np.exp(-np.maximum(ages, 1) / self.tau_fresh).sum() / self.tau_fresh))

    def _borrow_term(self, r: int) -> float:
        lam = float(self.borrow.get("lambda", 1.0))
        cap = float(self.borrow.get("cap_rel", 1.0)) * self.scale
        phi_r = self._phi(r)
        total, n = 0.0, 0
        for j in (r - 1, r + 1):
            if 0 <= j < self.n_regions:
                n += 1
                gain = self._phi(j) - phi_r
                if gain <= 0:
                    continue
                fresh, ref = self._base(j, self.tau_fresh), self._base(j, None)
                if math.isfinite(fresh) and math.isfinite(ref):
                    total += lam * gain * max(fresh - ref, 0.0)
        return min(total / max(n, 1), cap)

    # ---- interface -----------------------------------------------------------
    def threshold(self, t, x, region, yhat, ctx):
        r = int(region)
        self.ctr_at[t] = int(self.n_issued[r])
        self.n_issued[r] += 1
        base = self._base(r, self.age_tau)
        if not math.isfinite(base):
            return math.inf
        q = base + self.c[r]
        if self.borrow.get("on"):
            q += self._borrow_term(r)
        if self.guard.get("on"):
            q += float(self.guard.get("kappa_rel", 0.5)) * self.scale * (1.0 - self._phi(r))
        return max(q, 0.0)

    def update(self, i, x, region, score, err, t):
        r = int(region)
        ctr = self.ctr_at.pop(i)
        pos = self.n_arrived[r] % self.W
        self.buf_s[r, pos] = score
        self.buf_c[r, pos] = ctr
        self.buf_lag[r, pos] = self.n_issued[r] - ctr
        self.n_arrived[r] += 1
        self.c[r] += self.eta * (err - self.alpha)

    def diagnostics(self):
        return {"c": self.c.tolist()}
