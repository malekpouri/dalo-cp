"""The data structure every experiment consumes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Stream:
    """A finite regression stream with delays and a fixed base predictor.

    ``regions`` is the declared partition cell of each input (a function of X
    fixed before the stream starts), so methods may use it at prediction time.
    ``y`` must only reach a method through ``LabelOracle``.
    """

    x: np.ndarray            # (n, d) inputs
    regions: np.ndarray      # (n,) declared partition cell in {0..K-1}
    y: np.ndarray            # (n,) labels
    delay: np.ndarray        # (n,) integer delays >= 1
    yhat: np.ndarray         # (n,) base-predictor outputs
    n_regions: int
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.y)
        for name in ("regions", "delay", "yhat"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has wrong length")
        if len(self.x) != n:
            raise ValueError("x has wrong length")
        if np.any(self.delay < 1):
            raise ValueError("delays must be >= 1")

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def arrival(self) -> np.ndarray:
        return np.arange(self.n, dtype=np.int64) + self.delay.astype(np.int64)

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for arr in (self.x, self.regions, self.y, self.delay, self.yhat):
            h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()[:16]

    def with_delay(self, delay: np.ndarray) -> "Stream":
        return Stream(self.x, self.regions, self.y, np.asarray(delay, dtype=np.int64), self.yhat,
                      self.n_regions, dict(self.meta))


def apply_cqr(stream: Stream, lo: np.ndarray, hi: np.ndarray, scale: float) -> Stream:
    """Robustness check with conformalized quantile regression (Romano et al., 2019).

    The CQR score is ``|y - m| - h``, with ``m, h`` the centre and half-range of the quantile predictions. It can be negative, so the
    stream is rewritten with ``y' = sign(y - m) (|y - m| - h + C)``, ``yhat' = 0``, ``C = max(h)``. Then ``|y'| <= q'`` exactly when
    ``|y - m| <= h + q' - C``, so every method runs unchanged. ``C`` shifts scores by a constant and carries no label information.
    The true half-width is ``q' + width_offset`` with ``width_offset = h - C``, used by the evaluation.
    """
    m, h = (hi + lo) / 2, np.maximum((hi - lo) / 2, 0.0)
    c = float(h.max())
    y = np.sign(stream.y - m) * (np.abs(stream.y - m) - h + c)
    meta = dict(stream.meta, score_scale=scale, width_offset=h - c)
    return Stream(stream.x, stream.regions, y, stream.delay, np.zeros(stream.n), stream.n_regions, meta)
