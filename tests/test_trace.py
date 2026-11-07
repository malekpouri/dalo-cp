"""TRACE helpers: timing of labels, target encoding, the time-unit check, and the stream loop on a small file."""
import numpy as np
import pytest

from delayed_local_cp.data import trace


def test_delay_is_first_round_at_or_after_purchase_and_at_least_one():
    click = np.array([0, 10, 20, 30, 40])
    pay = np.array([5, 10, 35, 100, 41])
    # purchase at 5 -> round 1; purchase at click time of own round -> round 2 (D=1); 35 -> round 4; 100 -> after the stream
    assert trace.delays_from_times(click, pay).tolist() == [1, 1, 2, 2, 1]


def test_delay_never_reveals_a_label_before_its_purchase():
    rng = np.random.default_rng(0)
    click = np.sort(rng.integers(0, 10_000, 2000))
    pay = click + rng.integers(0, 3000, 2000)
    d = trace.delays_from_times(click, pay)
    arr = np.arange(2000) + d
    inside = arr < 2000
    assert np.all(d >= 1) and np.all(click[arr[inside]] >= pay[inside])


def test_target_encoding_smooths_and_handles_unseen_categories():
    u, v = trace.te_map(np.array([1, 1, 2]), np.array([1.0, 3.0, 5.0]), prior=3.0, smooth=2.0)
    assert u.tolist() == [1, 2]
    assert v.tolist() == pytest.approx([(4 + 6) / 4, (5 + 6) / 3])
    out = trace.te_apply(u, v, 3.0, np.array([2, 9, 0]))
    assert out[0] == pytest.approx(11 / 3) and out[1] == 3.0 and out[2] == 3.0


def test_out_of_fold_encoding_does_not_see_its_own_row():
    # every category occurs once, so an out-of-fold encoding can only return the prior
    cats = np.arange(100).reshape(-1, 1)
    target = np.random.default_rng(0).normal(size=100)
    enc = trace._encode(cats, np.arange(100), target)
    assert np.allclose(enc[:, 0], target.mean(), atol=1e-5)


def test_time_unit_refuses_unknown_spans():
    assert trace.time_unit_per_second(82 * 86400) == 1
    assert trace.time_unit_per_second(82 * 86400 * 1000) == 1000
    with pytest.raises(ValueError):
        trace.time_unit_per_second(5 * 86400)
