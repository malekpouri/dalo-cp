"""Shift schedules for synthetic streams.

Each shift is a dict:

``{kind: mean|variance|covariate, regions: [..]|all, start: t0, size: s,
   profile: abrupt|gradual|recurring, duration: L, period: P}``

* ``mean``      adds ``size * g(t)`` to Y in the listed regions (local concept shift).
* ``variance``  multiplies the noise scale by ``1 + size * g(t)``.
* ``covariate`` moves sampling mass: region weights become
                ``w * (1 + size * g(t))`` for the listed regions, renormalised.

``g(t)`` is the profile in [0, 1]: a step at ``start`` (abrupt), a linear ramp
over ``duration`` rounds (gradual), or a square wave with period ``period``
that is on for the first half of each period after ``start`` (recurring).
"""

from __future__ import annotations

import numpy as np


def profile(t: np.ndarray, shift: dict) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    t0 = float(shift["start"])
    kind = shift.get("profile", "abrupt")
    if kind == "abrupt":
        return (t >= t0).astype(float)
    if kind == "gradual":
        dur = max(float(shift.get("duration", 1)), 1.0)
        return np.clip((t - t0) / dur, 0.0, 1.0)
    if kind == "recurring":
        period = float(shift["period"])
        phase = np.mod(t - t0, period)
        return ((t >= t0) & (phase < period / 2)).astype(float)
    raise ValueError(f"unknown profile {kind!r}")


def region_mask(regions: np.ndarray, shift: dict) -> np.ndarray:
    target = shift.get("regions", "all")
    if target == "all":
        return np.ones(len(regions), dtype=bool)
    return np.isin(regions, np.asarray(target))
