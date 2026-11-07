"""Metric checks on small hand-calculated examples."""

import math

import numpy as np

from delayed_local_cp.evaluation import metrics as M
from delayed_local_cp.methods.base import conformal_quantile


def test_conformal_quantile_ranks():
    s = np.array([5.0, 1.0, 3.0, 2.0, 4.0])          # sorted: 1 2 3 4 5
    assert conformal_quantile(s, 0.5) == 3.0          # ceil(6*0.5)=3 -> 3rd smallest
    assert conformal_quantile(s, 0.8) == 5.0          # ceil(4.8)=5
    assert math.isinf(conformal_quantile(s, 0.9))     # ceil(5.4)=6 > 5
    assert math.isinf(conformal_quantile(np.array([]), 0.9))
    assert conformal_quantile(s, 0.0) == 0.0


def test_coverage_and_region_windows():
    y = np.array([0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    lo, hi = -np.ones(8), np.ones(8)
    reg = np.array([0, 0, 1, 0, 1, 1, 0, 1])
    cov = M.covered(y, lo, hi)                        # misses at rounds 1 and 4
    assert M.marginal_coverage(cov) == 6 / 8
    assert M.regional_coverage(cov, reg, 2) == [3 / 4, 3 / 4]
    blocks = M.region_window_coverage(cov, reg, 2, 2)
    # region 0 rounds 0,1,3,6 -> cov 1,0,1,1 -> blocks [0.5, 1]; region 1 rounds 2,4,5,7 -> 1,0,1,1
    np.testing.assert_allclose(blocks[0], [0.5, 1.0])
    np.testing.assert_allclose(blocks[1], [0.5, 1.0])
    assert M.worst_region_window_coverage(cov, reg, 2, 2) == 0.5
    assert math.isnan(M.worst_region_window_coverage(cov, reg, 2, 5))


def test_recovery_time_and_time_below():
    # one region; shift at round 4; coverage after shift: 0 0 1 1 1
    cov = np.array([1, 1, 1, 1, 0, 0, 1, 1, 1], dtype=bool)
    reg = np.zeros(9, dtype=int)
    # window 2 over post-shift rounds: (4,5)=0 (5,6)=.5 (6,7)=1 -> first hit ends at round 7
    assert M.recovery_time(cov, reg, 0, 4, 0.9, 2) == 3.0
    assert math.isinf(M.recovery_time(np.zeros(9, bool), reg, 0, 4, 0.9, 2))
    # rolling (w=2) at rounds 1..8: 1 1 1 .5 0 .5 1 1 -> below 0.9 at 3 of 8
    assert M.time_below_target(cov, reg, 1, 0.9, 2) == [3 / 8]


def test_interval_score_and_width():
    y = np.array([0.0, 3.0])
    lo, hi = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    # widths 2, 2; second misses above by 2 -> penalty 2/0.1*2 = 40
    assert M.interval_score(y, lo, hi, 0.1) == (2 + 2 + 40) / 2
    ws = M.width_stats(np.array([-1.0, -np.inf]), np.array([1.0, np.inf]))
    assert ws["mean_width"] == 2.0 and ws["frac_infinite"] == 0.5


def test_backlog_and_information_age():
    reg = np.array([0, 1, 0, 1])
    arrival = np.array([3, 2, 4, 10])   # delays 3, 1, 2, 7
    backlog, age = M.backlog_and_age(reg, arrival, 2)
    # region 0: issued 0 (arr 3), 2 (arr 4)
    assert backlog[:, 0].tolist() == [0, 1, 1, 1]       # t=3: round 0 arrived, round 2 pending
    assert age[:, 0].tolist() == [1, 2, 3, 3]           # t=3: latest arrived index 0 -> age 3
    # region 1: issued 1 (arr 2), 3 (arr 10)
    assert backlog[:, 1].tolist() == [0, 0, 0, 0]
    assert age[:, 1].tolist() == [1, 2, 1, 2]
