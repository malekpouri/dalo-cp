"""Crash-safe file helpers: every write is atomic or append-and-fsync."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import yaml


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Write to a temporary file in the same directory, fsync, then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n")


def atomic_write_yaml(path: str | Path, obj: Any) -> None:
    atomic_write_text(path, yaml.safe_dump(obj, sort_keys=False))


def atomic_write_pickle(path: str | Path, obj: Any) -> None:
    atomic_write_bytes(path, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


def read_pickle(path: str | Path) -> Any:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _json_default(o: Any) -> Any:
    import numpy as np

    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


class JsonlWriter:
    """Append-only JSONL writer that flushes and fsyncs every record.

    ``offset`` is the byte length after the last complete record; a checkpoint
    stores it so a resumed run can truncate records written after the checkpoint.
    """

    def __init__(self, path: str | Path, truncate_to: int | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "ab")
        if truncate_to is not None:
            self._fh.truncate(truncate_to)
            self._fh.seek(truncate_to)
        self._fh.seek(0, os.SEEK_END)

    @property
    def offset(self) -> int:
        return self._fh.tell()

    def write(self, record: dict) -> None:
        line = json.dumps(record, sort_keys=True, default=_json_default) + "\n"
        self._fh.write(line.encode("utf-8"))
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


def read_jsonl(path: str | Path) -> list[dict]:
    """Read complete records, ignoring a trailing partial line from a crash."""
    out = []
    path = Path(path)
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.endswith("\n"):
                break
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return out


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def config_hash(cfg: dict) -> str:
    """Stable SHA-256 of a config (key order does not matter)."""
    return hashlib.sha256(canonical_json(cfg).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()
