"""Pending-label queue and label oracle.

Timing convention (used everywhere in the project):

* At round ``t`` the learner first processes every label whose arrival round
  ``a_i = i + D_i`` satisfies ``a_i <= t``, then observes ``X_t`` and issues
  ``C_t(X_t)``.
* Delays satisfy ``D_i >= 1``, so the label of round ``t`` is never available
  when ``C_t`` is issued. ``D_i = 1`` is the classical no-delay online setting.
* Pending *inputs* (and their regions) are known to the learner; pending
  *labels* are not.
"""

from __future__ import annotations

import heapq
from collections import Counter
from dataclasses import dataclass, field

import numpy as np


class LeakageError(RuntimeError):
    """Raised when code asks for a label before its arrival round."""


class LabelOracle:
    """Holds the labels privately and reveals one only once it has arrived."""

    def __init__(self, y: np.ndarray, arrival: np.ndarray):
        self._y = np.asarray(y, dtype=float)
        self._arrival = np.asarray(arrival, dtype=np.int64)

    def reveal(self, i: int, t: int) -> float:
        if self._arrival[i] > t:
            raise LeakageError(f"label {i} arrives at round {self._arrival[i]}, requested at round {t}")
        return float(self._y[i])


@dataclass
class DelayQueue:
    """Min-heap of pending labels ordered by (arrival round, issue round)."""

    _heap: list = field(default_factory=list)
    _pending_by_region: Counter = field(default_factory=Counter)
    _last_round: int = -1

    def push(self, i: int, arrival: int, region: int) -> None:
        if arrival <= i:
            raise ValueError(f"delay must be >= 1 (round {i}, arrival {arrival})")
        if i <= self._last_round:
            raise ValueError("rounds must be pushed in increasing order")
        self._last_round = i
        heapq.heappush(self._heap, (int(arrival), int(i), int(region)))
        self._pending_by_region[int(region)] += 1

    def pop_due(self, t: int) -> list[tuple[int, int]]:
        """Remove and return ``(i, region)`` for all labels with arrival <= t.

        Labels come out ordered by arrival round, then by issue round, so
        out-of-order arrivals (a later round with a shorter delay) are handled
        deterministically.
        """
        out = []
        while self._heap and self._heap[0][0] <= t:
            _, i, r = heapq.heappop(self._heap)
            self._pending_by_region[r] -= 1
            if self._pending_by_region[r] == 0:
                del self._pending_by_region[r]
            out.append((i, r))
        return out

    @property
    def n_pending(self) -> int:
        return len(self._heap)

    def pending_in(self, region: int) -> int:
        return self._pending_by_region.get(int(region), 0)

    def pending_counts(self, n_regions: int) -> np.ndarray:
        return np.array([self._pending_by_region.get(r, 0) for r in range(n_regions)], dtype=np.int64)
