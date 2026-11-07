import numpy as np

from delayed_local_cp.theory import bounds as B


def test_kappa_values():
    # sigma 0.3, q at 90% coverage: 1.645 sigma
    q0 = B.q_for_coverage(0.9, 0.0, 0.3)
    assert abs(q0 - 1.6449 * 0.3) < 1e-3
    assert abs(float(B.kappa(q0, 0.6, 0.3)) - 0.3615) < 2e-3


def test_envelope_is_concave_majorant():
    g = np.linspace(0, 2, 400)
    e = B.kappa_envelope(g, 0.6, 0.3)
    assert np.all(e >= B.kappa(g, 0.6, 0.3) - 1e-6)   # hull is built on a finite grid
    assert np.all(np.diff(e, 2) <= 1e-5)


def test_expected_arrivals_fixed_delay_is_blind_before_d():
    lam = B.expected_arrivals("fixed", 10, 0.5, 20)
    assert np.all(lam[:10] == 0.0)
    assert lam[10] == 0.5 and lam[12] == 1.5


def test_expected_arrivals_geometric_hand_value():
    lam = B.expected_arrivals("geometric", 2, 1.0, 2)   # F(1) = 1/2, F(2) = 3/4
    np.testing.assert_allclose(lam, [0, 0.5, 1.25])


def test_information_lag_scaling():
    k = B.gaussian_kl(0.6, 0.3)
    assert k == 2.0
    # fixed: blind until D, then 1/(p k) more rounds; geometric: about sqrt(2 m / (p k))
    assert B.information_lag("fixed", 100, 0.5, k) == 100 + 0
    assert abs(B.information_lag("geometric", 400, 0.5, k) - np.sqrt(2 * 400 / (0.5 * 2))) < 3


def test_tv_bound_never_vacuous():
    assert np.all(B.tv_bound(np.array([0.0, 0.1, 5.0, 100.0])) <= 1.0)
    assert B.tv_bound(np.array([0.0]))[0] == 0.0


def test_window_bound_fixed_delay_blind_equals_envelope():
    q0 = B.q_for_coverage(0.9, 0.0, 0.3)
    ub = B.window_coverage_upper_bound(q0, 0.6, 0.3, "fixed", 100, 0.5, 50)
    assert abs(ub - float(B.kappa_envelope(q0, 0.6, 0.3))) < 1e-12   # no label has arrived: TV = 0 throughout
    assert B.window_coverage_upper_bound(q0, 0.6, 0.3, "geometric", 100, 0.5, 50) > ub


def _brute_force_tv(delta, sigma, kind, mean, p, L, n_grid=4001):
    """TV between the two laws of the arrived residuals, by enumerating arrival patterns over a short horizon."""
    from itertools import product

    from scipy import stats

    from delayed_local_cp.theory.bounds import delay_cdf

    s = [p * float(delay_cdf(kind, mean, np.array([a]))[0]) for a in range(1, L + 1)]
    total = 0.0
    for pat in product((0, 1), repeat=L):
        w = np.prod([x if b else 1 - x for x, b in zip(s, pat)])
        # Given the pattern, TV between N(0, I_n) and N(delta, I_n): the sufficient statistic is the mean.
        n = sum(pat)
        if n == 0:
            continue
        g = np.linspace(-12, 12 + delta * np.sqrt(n) / sigma, n_grid)
        f0, f1 = stats.norm.pdf(g), stats.norm.pdf(g - delta * np.sqrt(n) / sigma)
        total += w * 0.5 * np.trapezoid(np.abs(f0 - f1), g)
    return total


def test_tv_star_matches_brute_force_and_is_never_weaker():
    import pytest

    from delayed_local_cp.theory.bounds import tv_path, tv_star_path

    for kind, m in (("geometric", 5), ("fixed", 3)):
        star = tv_star_path(0.3, 0.3, kind, m, 0.5, 8)
        assert star[8] == pytest.approx(_brute_force_tv(0.3, 0.3, kind, m, 0.5, 8), abs=1e-5)
        assert np.all(star <= tv_path(0.3, 0.3, kind, m, 0.5, 8) + 1e-12)
        assert star[0] == 0


def test_tv_star_is_at_most_the_mean_plug_in():
    """Jensen: g(w) = 2 Phi(sqrt(w/2)) - 1 is concave, so E g(W) <= g(E W)."""
    from delayed_local_cp.theory.bounds import expected_arrivals, gaussian_kl, gaussian_tv_of_kl, tv_star_path

    lam = expected_arrivals("geometric", 50, 0.5, 40)
    assert np.all(tv_star_path(0.6, 0.3, "geometric", 50, 0.5, 40) <= gaussian_tv_of_kl(gaussian_kl(0.6, 0.3) * lam) + 1e-12)


def test_widening_bound_holds_for_the_exact_binary_kl():
    from delayed_local_cp.theory.bounds import binary_kl, widening_probability_bound

    for a in (0.1, 0.5, 0.9):
        for e in (1e-4, 0.01, 0.2):
            info = float(binary_kl(a, e))
            assert a <= float(widening_probability_bound(info, e)) + 1e-12


def test_mean_width_corollary_is_consistent_with_its_proof():
    from delayed_local_cp.theory.bounds import binary_kl, mean_width_information, q_for_coverage

    q0 = q_for_coverage(0.9, 0, 0.3)
    assert mean_width_information(q0, 0.85, 0.6, 0.3) == (0.0, None)   # no content below about 0.89
    info, q1 = mean_width_information(q0, 0.95, 0.6, 0.3)
    assert info > 0 and q1 > q0
    # Any eps at or below q_bar / q1 needs at least that information
    k1 = float(B.kappa(q1, 0.6, 0.3))
    a = (0.95 - k1) / (1 - k1)
    for eps in (q0 / q1, q0 / q1 / 2, q0 / q1 / 10):
        assert float(binary_kl(a, eps)) >= info - 1e-9



def test_stream_bound_helpers():
    assert B.stream_required_information(0.5, 0.6, 0.01) == 0.0
    v = B.stream_required_information(0.8, 0.05, 0.001)
    assert v > 0 and v >= 0.75 * np.log(1 / 0.001) - np.log(2) - 1e-9
    # Without delay and with more candidates, the required lag does not fall; with delay it is at least the blind period.
    a = B.lower_lag_false_alarm("fixed", 1, 0.5, 2.0, 0.8, 0.05, 10 ** 4)
    b = B.lower_lag_false_alarm("fixed", 1, 0.5, 2.0, 0.8, 0.05, 10 ** 8)
    assert b >= a >= 1
    assert B.lower_lag_false_alarm("fixed", 100, 0.5, 2.0, 0.8, 0.05, 10 ** 4) >= 100
    # Detector power is monotone in lag and zero before any label can arrive.
    path = B.detector_power("fixed", 50, 0.5, 2.0, 5.0, 120)
    assert path[49] == 0 and np.all(np.diff(path) >= -1e-9)


def test_power_table_matches_simulation_and_the_detector_form():
    lam = np.array([3.0, 8.0])
    t = B.arrival_power_table("geometric", 20, 0.5, 2.0, lam, 30)
    assert t[0].tolist() == [0.0, 0.0] and np.all(np.diff(t, axis=0) >= -1e-9) and np.all(t[:, 0] >= t[:, 1])
    assert abs(t[30, 0] - B.detector_power("geometric", 20, 0.5, 2.0, 3.0, 30)[30]) < 1e-12
    rng = np.random.default_rng(0)
    F = 1 - (1 - 1 / 20) ** np.arange(1, 31)
    n = (rng.random((200000, 30)) < 0.5 * F).sum(axis=1)
    w = 2.0 * n
    log_ratio = w + np.sqrt(2 * w) * rng.standard_normal(200000)
    assert abs(float(np.mean(log_ratio >= 3.0)) - t[30, 0]) < 0.005


def test_neyman_pearson_bound_on_an_exact_two_point_test():
    """b <= P1(Lambda >= lam) + exp(lam) e for the likelihood-ratio test at any threshold, with fixed information."""
    from scipy import stats

    w = 3.0
    for thr in (0.5, 2.0, 4.0):
        b = stats.norm.sf((thr - w) / np.sqrt(2 * w))
        e = stats.norm.sf((thr + w) / np.sqrt(2 * w))
        for lam in (0.1, 1.0, 3.0, 6.0):
            assert b <= stats.norm.sf((lam - w) / np.sqrt(2 * w)) + np.exp(lam) * e + 1e-12


def test_first_wide_round_decomposition_with_rewidening_paths():
    """Corollary S.2: wide at t_j implies H_j or E_j; at most one E_j holds, however often the path re-widens."""
    rng = np.random.default_rng(0)
    T, L = 400, 9
    c = np.arange(20, T - L, L + 1)
    for _ in range(300):
        wide = rng.random(T) < rng.choice([0.0, 0.02, 0.3])
        hits = np.flatnonzero(wide)
        tau = hits[0] if len(hits) else T + 1
        E = (c <= tau) & (tau <= c + L)
        H = tau < c
        assert E.sum() <= (tau <= T)
        assert np.all(~wide[c + L] | H | E)
    # The reviewer's example: a path re-widening in all J windows has J upcrossings, but only one window holds its first wide round.
    J = len(c)
    wide = np.zeros(T, dtype=bool)
    wide[c] = True
    tau = np.flatnonzero(wide)[0]
    assert int(np.sum((c <= tau) & (tau <= c + L))) == 1
    assert int(np.sum(wide[1:] & ~wide[:-1])) + int(wide[0]) == J


def test_fixed_budget_bounds_upcrossings():
    rng = np.random.default_rng(1)
    q_a, q1 = 0.5, 0.7
    for _ in range(200):
        q = q_a + rng.exponential(0.1, 500) * (rng.random(500) < 0.2)
        wide = q > q1
        n = int(np.sum(wide[1:] & ~wide[:-1])) + int(wide[0])
        assert n * (q1 - q_a) <= np.sum(np.maximum(q - q_a, 0)) + 1e-12


def test_fixed_budget_bounds_rho_on_random_and_adversarial_paths():
    """Proposition R.1: averaged wide-before-candidate rate against budget and empty-window term."""
    rng = np.random.default_rng(2)
    T, L, p, q_a, q1 = 3000, 19, 0.3, 0.5, 0.7
    c = np.arange(L + 1, T - L, L + 1)
    J = len(c)
    rate, excess, wide_runs = 0.0, 0.0, 400
    for _ in range(wide_runs):
        reg = rng.random(T) < p
        idx = np.flatnonzero(reg)
        q = np.full(T, q_a)
        # Adversary: spend the budget exactly on the last region round before every candidate.
        prev = idx[np.searchsorted(idx, c) - 1]
        q[prev[np.searchsorted(idx, c) > 0]] = q1 + 0.01
        excess += float(np.sum(np.maximum(q - q_a, 0)))
        s = np.array([q[idx[k - 1]] > q1 if k > 0 else False for k in np.searchsorted(idx, c)])
        rate += float(s.mean())
    rate, budget = rate / wide_runs, excess / wide_runs
    assert rate <= budget / (J * (q1 - q_a)) + (1 - p) ** (L + 1) + 1e-9

