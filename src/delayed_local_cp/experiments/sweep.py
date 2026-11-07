"""Experiment configs, sweep expansion, and resumable parallel execution.

Experiment YAML::

    name: smoke
    alpha: 0.1
    dataset: {use: synthetic_two_region, n_rounds: 4000}   # 'use' loads configs/datasets/<name>.yaml
    delay:   {use: two_region_geometric}
    shifts:  {use: slow_region_mean_jump}                  # file holds a list under 'shifts'
    eval:    {window: 100}
    runtime: {log_every: 100}
    methods: [{use: aci}, {use: regional_aci}, {use: regional_aci, oracle_no_delay: true, label: regional_aci_oracle}]
    seeds:   {start: 0, count: 2}                           # or an explicit list
    grid:    {"delay.per_region": [[1, 5], [1, 20]]}       # optional, dotted paths

Each (method, grid point, seed) becomes one run with id
``<name>__<label>[__g<k>]__s<seed>__<hash10>``.
"""

from __future__ import annotations

import copy
import itertools
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

from ..tracking import signals
from ..tracking.registry import Registry
from ..utils.gitinfo import git_commit
from ..utils.io import config_hash
from ..utils.paths import CONFIGS_DIR
from .runner import _hash_cfg, load_config, run_dir_for, run_single, write_config

log = logging.getLogger(__name__)

KIND_DIRS = {"dataset": "datasets", "delay": "delays", "shifts": "shifts", "method": "methods"}


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve(kind: str, spec, configs_dir: Path = CONFIGS_DIR):
    """Resolve ``{use: name, **overrides}`` against configs/<kind>/<name>.yaml."""
    if spec is None:
        return [] if kind == "shifts" else None
    if isinstance(spec, str):
        spec = {"use": spec}
    if isinstance(spec, list):
        return copy.deepcopy(spec)
    spec = dict(spec)
    name = spec.pop("use", None)
    if name is None:
        return spec
    with open(configs_dir / KIND_DIRS[kind] / f"{name}.yaml", encoding="utf-8") as fh:
        base = yaml.safe_load(fh)
    if kind == "shifts":
        return copy.deepcopy(base["shifts"])
    base.setdefault("name", name)
    return _deep_merge(base, spec)


def _set_dotted(cfg: dict, path: str, value) -> None:
    keys = path.split(".")
    cur = cfg
    for key in keys[:-1]:
        cur = cur.setdefault(key, {})
    cur[keys[-1]] = copy.deepcopy(value)


def _seeds(spec) -> list[int]:
    if isinstance(spec, dict):
        return list(range(int(spec.get("start", 0)), int(spec.get("start", 0)) + int(spec["count"])))
    return [int(s) for s in spec]


def expand(exp: dict, configs_dir: Path = CONFIGS_DIR, name_suffix: str = "") -> list[dict]:
    name = exp["name"] + name_suffix
    base = {
        "experiment": name,
        "alpha": float(exp.get("alpha", 0.1)),
        "dataset": resolve("dataset", exp["dataset"], configs_dir),
        "delay": resolve("delay", exp["delay"], configs_dir),
        "shifts": resolve("shifts", exp.get("shifts"), configs_dir),
        "eval": dict(exp.get("eval", {"window": 100})),
        "runtime": dict(exp.get("runtime", {})),
    }
    grid = exp.get("grid") or {}
    keys = list(grid)
    points = list(itertools.product(*[grid[k] for k in keys])) if keys else [()]
    runs = []
    for m_spec in exp["methods"]:
        m_spec = dict(m_spec) if isinstance(m_spec, dict) else {"use": m_spec}
        oracle = bool(m_spec.pop("oracle_no_delay", False))
        method = resolve("method", m_spec, configs_dir)
        method.setdefault("label", method.get("name", method["type"]) + ("_oracle" if oracle else ""))
        for gi, point in enumerate(points):
            for seed in _seeds(exp.get("seeds", [0])):
                cfg = copy.deepcopy(base)
                cfg["method"] = method
                cfg["oracle_no_delay"] = oracle
                cfg["seed"] = seed
                for k, v in zip(keys, point):
                    _set_dotted(cfg, k, v)
                if keys:
                    cfg["grid_point"] = dict(zip(keys, point))
                h = config_hash(cfg)[:10]
                g = f"__g{gi}" if keys else ""
                cfg["run_id"] = f"{name}__{method['label']}{g}__s{seed}__{h}"
                runs.append(cfg)
    ids = [c["run_id"] for c in runs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate run ids; give methods distinct labels")
    return runs


def load_experiment(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _worker(cfg: dict) -> tuple[str, str]:
    signals.install()
    if signals.stop_requested():  # shutdown in progress: leave the run queued
        return cfg["run_id"], "not started (stop requested)"
    try:
        return cfg["run_id"], run_single(cfg)
    except Exception as exc:  # already recorded as failed in the registry
        return cfg["run_id"], f"failed: {type(exc).__name__}: {exc}"


def register_all(cfgs: list[dict], registry: Registry) -> None:
    commit = git_commit()
    for cfg in cfgs:
        write_config(cfg)
        registry.register(cfg["run_id"], experiment=cfg["experiment"],
                          method=cfg["method"].get("label", cfg["method"]["type"]),
                          dataset=cfg["dataset"].get("name", ""), seed=int(cfg["seed"]),
                          config_hash=_hash_cfg(cfg), git_commit=commit, run_dir=str(run_dir_for(cfg)))


def execute(cfgs: list[dict], jobs: int = 1, registry: Registry | None = None) -> dict[str, str]:
    """Run every config whose summary.json does not exist yet."""
    registry = registry or Registry()
    register_all(cfgs, registry)
    todo = [c for c in cfgs if not (run_dir_for(c) / "summary.json").exists()]
    results = {c["run_id"]: "completed" for c in cfgs if c not in todo}
    signals.install()
    if not todo:
        return results
    if jobs <= 1:
        for cfg in todo:
            if signals.stop_requested():
                break
            rid, status = _worker(cfg)
            results[rid] = status
            print(f"[{status}] {rid}", flush=True)
        return results
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futs = {pool.submit(_worker, c): c["run_id"] for c in todo}
        for fut in as_completed(futs):
            rid, status = fut.result()
            results[rid] = status
            print(f"[{status}] {rid}", flush=True)
            if signals.stop_requested():
                pool.shutdown(wait=True, cancel_futures=True)
                break
    return results


def pending_from_registry(registry: Registry) -> list[dict]:
    """Configs of runs that are queued or interrupted (after marking stale runs)."""
    registry.mark_stale()
    cfgs = []
    for row in registry.list():
        if row["status"] in ("queued", "interrupted"):
            rd = run_dir_for({"run_id": row["run_id"]})
            if (rd / "config.yaml").exists():
                cfgs.append(load_config(rd))
    return cfgs
