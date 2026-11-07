import numpy as np
import pytest

from delayed_local_cp.streams.delay_queue import DelayQueue, LabelOracle, LeakageError


def test_out_of_order_arrivals_are_ordered_by_arrival_then_issue():
    q = DelayQueue()
    q.push(0, 5, region=1)   # long delay
    q.push(1, 2, region=0)   # arrives before round 0's label
    q.push(2, 3, region=0)
    q.push(3, 5, region=0)   # same arrival round as label 0
    assert q.pop_due(1) == []
    assert q.pop_due(2) == [(1, 0)]
    assert q.pop_due(4) == [(2, 0)]
    assert q.pop_due(5) == [(0, 1), (3, 0)]
    assert q.n_pending == 0


def test_pop_due_catches_up_on_skipped_rounds():
    q = DelayQueue()
    for i, a in enumerate([3, 2, 4]):
        q.push(i, a, region=0)
    assert [i for i, _ in q.pop_due(10)] == [1, 0, 2]


def test_pending_counts_by_region():
    q = DelayQueue()
    q.push(0, 10, 0)
    q.push(1, 3, 1)
    q.push(2, 10, 1)
    assert q.pending_counts(3).tolist() == [1, 2, 0]
    q.pop_due(3)
    assert q.pending_counts(3).tolist() == [1, 1, 0]
    assert q.pending_in(1) == 1


def test_zero_delay_and_unordered_push_rejected():
    q = DelayQueue()
    with pytest.raises(ValueError):
        q.push(0, 0, 0)
    q.push(1, 2, 0)
    with pytest.raises(ValueError):
        q.push(1, 3, 0)


def test_oracle_refuses_labels_before_arrival():
    o = LabelOracle(np.array([1.0, 2.0]), np.array([3, 1]))
    assert o.reveal(1, 1) == 2.0
    with pytest.raises(LeakageError):
        o.reveal(0, 2)
    assert o.reveal(0, 3) == 1.0
