"""Quantities in the theory claims (theory/proofs/). Instance for Claim 2: Gaussian residuals, frozen base predictor."""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import brentq


def kappa(q, delta, sigma):
    """Coverage P(|Z + delta| <= q) of a half-width q interval when residuals are N(delta, sigma^2)."""
    q = np.asarray(q, dtype=float)
    return stats.norm.cdf((q - delta) / sigma) - stats.norm.cdf((-q - delta) / sigma)


def q_for_coverage(level, delta, sigma):
    return brentq(lambda q: float(kappa(q, delta, sigma)) - level, 0.0, abs(delta) + 20 * sigma)


def kappa_envelope(q, delta, sigma, grid_max=None, n=4001):
    """Least concave majorant of q -> kappa(q, delta, sigma) on [0, grid_max], evaluated at q."""
    grid_max = grid_max or abs(delta) + 12 * sigma
    g = np.linspace(0.0, grid_max, n)
    v = kappa(g, delta, sigma)
    hull = [0]
    for i in range(1, n):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            if (v[b] - v[a]) * (g[i] - g[a]) <= (v[i] - v[a]) * (g[b] - g[a]):
                hull.pop()
            else:
                break
        hull.append(i)
    return np.interp(q, g[hull], v[hull])


def gaussian_kl(delta, sigma):
    return delta ** 2 / (2 * sigma ** 2)


def delay_cdf(kind: str, mean: float, a: np.ndarray) -> np.ndarray:
    """P(D <= a) for a = 1, 2, ... under fixed or geometric delay with support {1, 2, ...}."""
    a = np.asarray(a, dtype=float)
    if kind == "fixed":
        return (a >= mean).astype(float)
    if kind == "geometric":
        return 1.0 - (1.0 - 1.0 / mean) ** a
    raise ValueError(kind)


def expected_arrivals(kind: str, mean: float, p: float, horizon: int) -> np.ndarray:
    """Lambda(L) = p * sum_{a=1}^{L} F(a): expected post-shift labels of the region arrived L rounds after the shift.

    Element L of the returned array (L = 0, ..., horizon) applies to the round t0 + L.
    """
    a = np.arange(1, horizon + 1)
    return np.concatenate([[0.0], p * np.cumsum(delay_cdf(kind, mean, a))])


def tv_bound(kl):
    """min(Pinsker, Bretagnolle-Huber) upper bound on total variation from a KL divergence."""
    kl = np.asarray(kl, dtype=float)
    return np.minimum(np.minimum(np.sqrt(kl / 2), np.sqrt(1 - np.exp(-kl))), 1.0)


def tv_path(delta, sigma, kind, mean, p, horizon):
    """TV bound at t0 + L, L = 0..horizon, from KL = K * Lambda(L)."""
    return tv_bound(gaussian_kl(delta, sigma) * expected_arrivals(kind, mean, p, horizon))


def window_coverage_upper_bound(q_bar, delta, sigma, kind, mean, p, window):
    """Corollary to Claim 2: coverage averaged over rounds t0, ..., t0 + window - 1, given mean half-width q_bar under E0.

    Jensen applies to the average, so the clip at 1 comes after averaging the TV terms.
    """
    tv = tv_path(delta, sigma, kind, mean, p, window - 1).mean()
    return float(min(1.0, float(kappa_envelope(q_bar, delta, sigma)) + float(tv)))


def information_lag(kind: str, mean: float, p: float, kl: float, nats: float = 1.0, horizon: int = 200000) -> int:
    """Smallest L with KL * Lambda(L) >= nats: rounds until the labels can carry that much information about the shift."""
    lam = expected_arrivals(kind, mean, p, horizon)
    return int(np.searchsorted(kl * lam, nats, side="left"))


def lag_for_labels(kind: str, mean: float, p: float, n_labels: float, horizon: int = 200000) -> int:
    """Lambda^{-1}: smallest L with Lambda(L) >= n_labels, the rounds until a method needing n_labels has them in expectation."""
    return int(np.searchsorted(expected_arrivals(kind, mean, p, horizon), n_labels, side="left"))


def stability_limit_step(density: float, delay_region_rounds: float) -> float:
    """Linearised stability limit for c_{k+1} = c_k - a c_{k-D}: a < 2 sin(pi / (2 (2 D + 1))), with a = eta * density."""
    d = float(delay_region_rounds)
    return 2 * np.sin(np.pi / (2 * (2 * d + 1))) / density


# --- Draft extensions (theory/proofs/claim2_extensions_draft.md); status: draft, not reviewed ---------------------------

def gaussian_tv_of_kl(w):
    """Exact TV between N(0, I) and a shifted N(mu, I) with |mu|^2 / 2 = w."""
    return 2 * stats.norm.cdf(np.sqrt(np.asarray(w, dtype=float) / 2)) - 1


def arrival_count_law(kind: str, mean: float, p: float, horizon: int) -> list:
    """Law of N(L), post-shift labels of the shifted region arrived by t0 + L, L = 0..horizon: a sum of independent Bernoulli(p F(a))."""
    laws, dist = [np.array([1.0])], np.array([1.0])
    for a in range(1, horizon + 1):
        s = p * float(delay_cdf(kind, mean, np.array([a]))[0])
        dist = np.convolve(dist, [1 - s, s])
        laws.append(dist)
    return laws


def tv_star_path(delta, sigma, kind, mean, p, horizon):
    """TV*(L) for one shifted region, L = 0..horizon: the exact TV between the laws of the learner's information."""
    k = gaussian_kl(delta, sigma)
    return np.array([float(np.sum(d * gaussian_tv_of_kl(k * np.arange(len(d))))) for d in arrival_count_law(kind, mean, p, horizon)])


def window_coverage_upper_bound_star(q_bar, delta, sigma, kind, mean, p, window):
    tv = tv_star_path(delta, sigma, kind, mean, p, window - 1).mean()
    return float(min(1.0, float(kappa_envelope(q_bar, delta, sigma)) + float(tv)))


def binary_kl(a, e):
    a, e = np.clip(a, 1e-300, 1 - 1e-16), np.clip(e, 1e-300, 1 - 1e-16)
    return a * np.log(a / e) + (1 - a) * np.log((1 - a) / (1 - e))


def widening_probability_bound(info, eps):
    """Upper bound on P_1(q_t > q1) given P_0(q_t > q1) = eps and information `info`, from kl(a||eps) >= a log(1/eps) - log 2."""
    return np.minimum((np.asarray(info, dtype=float) + np.log(2)) / np.log(1 / np.asarray(eps, dtype=float)), 1.0)


def information_needed(target_coverage, q1, delta, sigma, eps):
    """Smallest information consistent with coverage target_coverage at round t, for a method with P_0(q_t > q1) = eps (exact binary kl)."""
    k1 = float(kappa(q1, delta, sigma))
    if target_coverage <= k1:
        return 0.0
    a = (target_coverage - k1) / (1 - k1)
    return float(binary_kl(a, eps)) if a > eps else 0.0


def mean_width_information(q_bar, target_coverage, delta, sigma, grid=None):
    """Corollary E.4: the largest kl(a_c || q_bar / q1) over q1, or (0.0, None) if the hypothesis holds for no q1."""
    best, best_q = 0.0, None
    for q1 in (np.linspace(0.05, 3.0, 2951) if grid is None else grid):
        k1 = float(kappa(q1, delta, sigma))
        if target_coverage <= k1:
            continue
        a = (target_coverage - k1) / (1 - k1)
        if q_bar / q1 < a:
            v = float(binary_kl(a, q_bar / q1))
            if v > best:
                best, best_q = v, float(q1)
    return best, best_q



# --- Stream-level drafts (theory/proofs/claim2_stream_draft.md); status: draft, not reviewed --------------------------

def stream_required_information(power, rho, nu_over_p):
    """Theorem S.1: I(L) >= kl(power - rho || nu / p), or 0 when the hypothesis power - rho > nu / p fails."""
    a = power - rho
    return float(binary_kl(a, nu_over_p)) if a > nu_over_p > 0 else 0.0


def lower_lag_false_alarm(kind, mean, p, kl, power, delta_f, horizon, max_lag=20000):
    """Smallest L consistent with Corollary S.2, with J = floor(horizon / (L + 1)) candidates."""
    lam = kl * expected_arrivals(kind, mean, p, max_lag)
    for lag in range(max_lag + 1):
        j = horizon // (lag + 1)
        if j >= 1 and lam[lag] >= stream_required_information(power, delta_f, delta_f / (p * j)):
            return lag
    return None


def detector_power(kind, mean, p, kl, b, horizon_lag):
    """Proposition S.3: lower bound on the probability of an alarm by lag L = 0..horizon_lag, from the arrival-count law."""
    out = []
    for d in arrival_count_law(kind, mean, p, horizon_lag):
        w = kl * np.arange(len(d))
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(w > 0, (w - b) / np.sqrt(2 * np.maximum(w, 1e-300)), -np.inf)
        out.append(float(np.sum(d * stats.norm.cdf(z))))
    return np.array(out)


def arrival_power_table(kind, mean, p, kl, lambdas, horizon_lag):
    """pi_L(lambda) = E[Phi((W_L - lambda) / sqrt(2 W_L))] for L = 0..horizon_lag (rows) and each lambda > 0 (columns)."""
    lambdas = np.asarray(lambdas, dtype=float)
    out = np.zeros((horizon_lag + 1, len(lambdas)))
    for lag, d in enumerate(arrival_count_law(kind, mean, p, horizon_lag)):
        w = kl * np.arange(1, len(d))
        if len(w):
            out[lag] = np.sum(d[1:, None] * stats.norm.cdf((w[:, None] - lambdas) / np.sqrt(2 * w[:, None])), axis=0)
    return out


def stream_power_ceiling(pi_row, lambdas, nu_over_p):
    """Theorem S.1, right-hand side: the smallest pi_L(lambda) + exp(lambda) nu / p over the grid."""
    return float(np.min(pi_row + np.exp(lambdas) * nu_over_p))


def lower_lag_neyman_pearson(kind, mean, p, kl, power, rho, events, horizon, lambdas, table):
    """Smallest L at which Theorem S.1 allows mean widening probability `power`, with J = floor(horizon / (L + 1)) candidates."""
    for lag in range(table.shape[0]):
        j = horizon // (lag + 1)
        if j >= 1 and stream_power_ceiling(table[lag], lambdas, events / (p * j)) >= power - rho:
            return lag
    return None
