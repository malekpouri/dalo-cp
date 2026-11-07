"""Single-run execution with checkpointing and resume (PROJECT_BRIEF §5, §17.1).

Run directory ``runs/<run_id>/``:

* ``config.yaml``     resolved config (written once)
* ``run.log``         human-readable log (appended across resumes)
* ``metrics.jsonl``   periodic live metrics, fsynced per record
* ``checkpoint.pkl``  full loop state; removed when the run completes
* ``rounds.npz``      per-round arrays (written at the end)
* ``summary.json``    final metrics; written last, so it marks completion

A resumed run truncates ``metrics.jsonl`` to the byte offset saved in the
checkpoint and replays from the checkpointed round, so its outputs equal those
of an uninterrupted run (wall-clock fields under ``wall`` excepted).
"""

from __future__ import annotations

import io
import logging
import math
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from ..data.registry import build_stream
from ..evaluation.metrics import summarize
from ..methods.base import RoundContext
from ..methods.baselines import build_method
from ..streams.delay_queue import DelayQueue, LabelOracle
from ..tracking import signals
from ..tracking.registry import HEARTBEAT_EVERY_S, Registry
from ..utils.gitinfo import git_commit
from ..utils.io import (JsonlWriter, atomic_write_bytes, atomic_write_json, atomic_write_pickle,
                        atomic_write_yaml, config_hash, read_pickle)
from ..utils.paths import RUNS_DIR

CHECKPOINT_VERSION = 1


@dataclass
class LoopState:
    t: int
    method: object
    queue: DelayQueue
    q: np.ndarray
    latest_arrived: np.ndarray          # per region: latest issued index whose label arrived
    roll_region: list                    # per region: deque of covered flags (last w rounds)
    roll_all: deque                      # last w covered flags, all regions
    roll_width: deque
    arrived_cov_sum: float = 0.0
    arrived_n: int = 0
    jsonl_offset: int = 0
    stream_fp: str = ""
    version: int = CHECKPOINT_VERSION
    extra: dict = field(default_factory=dict)


def run_dir_for(cfg: dict) -> Path:
    return RUNS_DIR / cfg["run_id"]


def _hash_cfg(cfg: dict) -> str:
    return config_hash({k: v for k, v in cfg.items() if k not in ("run_id",)})


def _logger(run_dir: Path, run_id: str) -> logging.Logger:
    log = logging.getLogger(f"run.{run_id}")
    log.setLevel(logging.INFO)
    log.propagate = False
    if not log.handlers:
        h = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(h)
    return log


def _close_logger(log: logging.Logger) -> None:
    for h in list(log.handlers):
        h.close()
        log.removeHandler(h)


def write_config(cfg: dict) -> Path:
    rd = run_dir_for(cfg)
    rd.mkdir(parents=True, exist_ok=True)
    path = rd / "config.yaml"
    if not path.exists():
        atomic_write_yaml(path, cfg)
    return path


def load_config(run_dir: Path) -> dict:
    with open(run_dir / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def run_single(cfg: dict, registry: Registry | None = None, stop_after: int | None = None) -> str:
    """Execute (or resume) one run. Returns the final status.

    ``stop_after`` simulates a shutdown after that many rounds (tests only).
    """
    registry = registry or Registry()
    run_id = cfg["run_id"]
    rd = run_dir_for(cfg)
    if (rd / "summary.json").exists():
        return "completed"
    write_config(cfg)
    log = _logger(rd, run_id)
    commit = git_commit()
    chash = _hash_cfg(cfg)
    registry.register(run_id, experiment=cfg.get("experiment", ""), method=cfg["method"].get("label", cfg["method"]["type"]),
                      dataset=cfg["dataset"].get("name", ""), seed=int(cfg["seed"]), config_hash=chash,
                      git_commit=commit, run_dir=str(rd))
    registry.set_running(run_id, commit)
    try:
        status = _execute(cfg, rd, registry, log, stop_after)
    except Exception as exc:  # recorded, never hidden
        log.error("run failed: %s\n%s", exc, traceback.format_exc())
        registry.finish(run_id, "failed", error=f"{type(exc).__name__}: {exc}")
        _close_logger(log)
        raise
    _close_logger(log)
    return status


def _execute(cfg: dict, rd: Path, registry: Registry, log: logging.Logger, stop_after: int | None) -> str:
    run_id = cfg["run_id"]
    alpha = float(cfg["alpha"])
    rt = cfg.get("runtime", {})
    log_every = int(rt.get("log_every", 100))
    ckpt_rounds = int(rt.get("checkpoint_every_rounds", 5000))
    ckpt_seconds = float(rt.get("checkpoint_every_s", 300))
    w = int(cfg.get("eval", {}).get("window", 100))

    stream = build_stream(cfg)
    n, k = stream.n, stream.n_regions
    arrival = stream.arrival
    oracle = LabelOracle(stream.y, arrival)
    fp = stream.fingerprint()
    ckpt_path = rd / "checkpoint.pkl"

    if ckpt_path.exists():
        st: LoopState = read_pickle(ckpt_path)
        if st.version != CHECKPOINT_VERSION or st.stream_fp != fp:
            raise RuntimeError("checkpoint does not match this stream/config; refusing to resume")
        writer = JsonlWriter(rd / "metrics.jsonl", truncate_to=st.jsonl_offset)
        log.info("resuming from round %d of %d", st.t, n)
    else:
        method = build_method(cfg["method"], alpha, k, stream.meta)
        st = LoopState(t=0, method=method, queue=DelayQueue(), q=np.full(n, np.nan),
                       latest_arrived=np.full(k, -1, dtype=np.int64),
                       roll_region=[deque(maxlen=w) for _ in range(k)], roll_all=deque(maxlen=w),
                       roll_width=deque(maxlen=w), stream_fp=fp)
        writer = JsonlWriter(rd / "metrics.jsonl", truncate_to=0)
        log.info("starting run %s: n=%d regions=%d stream=%s", run_id, n, k, fp)

    def save_checkpoint() -> None:
        st.jsonl_offset = writer.offset
        atomic_write_pickle(ckpt_path, st)

    t_wall0 = time.time()
    last_ckpt_t, last_ckpt_wall, last_hb = st.t, time.time(), 0.0
    method = st.method
    x, regions, yhat, y = stream.x, stream.regions, stream.yhat, stream.y

    for t in range(st.t, n):
        # 1) labels arriving at the start of round t
        for i, r in st.queue.pop_due(t):
            yi = oracle.reveal(i, t)
            q_i = st.q[i]
            st.arrived_cov_sum += float(abs(yi - yhat[i]) <= q_i)
            st.arrived_n += 1
            method.on_arrival(i, x[i], r, yhat[i], yi, t)
            if i > st.latest_arrived[r]:
                st.latest_arrived[r] = i
        # 2) issue C_t using only arrived labels and known pending inputs
        r_t = int(regions[t])
        ctx = RoundContext(t=t, pending_counts=st.queue.pending_counts(k), n_pending=st.queue.n_pending)
        q_t = method.issue(t, x[t], r_t, float(yhat[t]), ctx)
        st.q[t] = q_t
        st.queue.push(t, int(arrival[t]), r_t)
        # 3) evaluator-side bookkeeping (uses the true label; never passed to the method)
        cov_t = float(abs(y[t] - yhat[t]) <= q_t)
        st.roll_region[r_t].append(cov_t)
        st.roll_all.append(cov_t)
        st.roll_width.append(2 * q_t)
        st.t = t + 1

        if st.t % log_every == 0 or st.t == n:
            widths = np.array(st.roll_width)
            fin = np.isfinite(widths)
            writer.write({
                "t": st.t,
                "n_rounds": n,
                "rolling_coverage": float(np.mean(st.roll_all)),
                "rolling_region_coverage": [float(np.mean(d)) if len(d) == w else None for d in st.roll_region],
                "rolling_width": float(widths[fin].mean()) if fin.any() else None,
                "frac_infinite": float(1 - fin.mean()),
                "arrived_coverage": st.arrived_cov_sum / st.arrived_n if st.arrived_n else None,
                "backlog": int(st.queue.n_pending),
                "backlog_by_region": st.queue.pending_counts(k).tolist(),
                "info_age_by_region": [int(t - a) if a >= 0 else int(t + 1) for a in st.latest_arrived],
                "method": method.diagnostics(),
                "wall": {"elapsed_s": round(time.time() - t_wall0, 3)},
            })
        now = time.time()
        if now - last_hb >= HEARTBEAT_EVERY_S:
            registry.heartbeat(run_id, st.t / n)
            last_hb = now
        stop = signals.stop_requested() or (stop_after is not None and st.t >= stop_after)
        if stop and st.t < n:
            save_checkpoint()
            writer.close()
            registry.finish(run_id, "interrupted", error="stop requested; checkpoint saved")
            log.info("interrupted at round %d; checkpoint saved", st.t)
            return "interrupted"
        if st.t < n and (st.t - last_ckpt_t >= ckpt_rounds or now - last_ckpt_wall >= ckpt_seconds):
            save_checkpoint()
            last_ckpt_t, last_ckpt_wall = st.t, now

    writer.close()
    lo, hi = yhat - st.q, yhat + st.q
    summary = summarize(y, lo, hi, regions, arrival, k, alpha, w,
                        stream.meta.get("shift_starts", []), stream.meta.get("shift_regions", []))
    summary.update({"run_id": run_id, "stream_fingerprint": fp, "config_hash": _hash_cfg(cfg),
                    "method": cfg["method"].get("label", cfg["method"]["type"]), "seed": int(cfg["seed"])})
    buf = io.BytesIO()
    extra = {"width_offset": stream.meta["width_offset"]} if "width_offset" in stream.meta else {}
    np.savez_compressed(buf, t=np.arange(n), region=regions, y=y, yhat=yhat, q=st.q, delay=stream.delay, **extra)
    atomic_write_bytes(rd / "rounds.npz", buf.getvalue())
    atomic_write_json(rd / "summary.json", _clean(summary))
    if ckpt_path.exists():
        ckpt_path.unlink()
    registry.finish(run_id, "completed", extra={"marginal_coverage": summary["marginal_coverage"],
                                                 "worst_region_window_coverage": summary["worst_region_window_coverage"],
                                                 "mean_width": summary["mean_width"]})
    log.info("completed: marginal=%.4f worst-region-window=%.4f width=%.4f",
             summary["marginal_coverage"], summary["worst_region_window_coverage"], summary["mean_width"])
    return "completed"


def _clean(obj):
    """Make floats JSON-safe (inf/nan become strings)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return "inf" if obj > 0 else ("-inf" if obj < 0 else "nan")
    return obj
