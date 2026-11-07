"""Dataset dispatch. Real datasets (ELEC2, TRACE) are added in later milestones."""

from __future__ import annotations

from ..streams.stream import Stream
from ..streams.synthetic import make_synthetic_stream


def build_stream(cfg: dict) -> Stream:
    ds = cfg["dataset"]
    kind = ds.get("kind", "synthetic")
    if kind == "synthetic":
        stream = make_synthetic_stream(ds, cfg["delay"], cfg.get("shifts", []), int(cfg["seed"]))
    elif kind == "elec2":
        from .elec2 import make_elec2_stream

        stream = make_elec2_stream(ds, cfg["delay"], int(cfg["seed"]))
    elif kind == "trace":
        from .trace import make_trace_stream

        stream = make_trace_stream(ds, int(cfg["seed"]))
    else:
        raise NotImplementedError(f"dataset kind {kind!r} is not implemented yet")
    if cfg.get("oracle_no_delay", False):
        # Reference only: every label arrives one round later (classical online setting).
        stream = stream.with_delay(stream.delay * 0 + 1)
    return stream
