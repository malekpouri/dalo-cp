from __future__ import annotations

import subprocess

from .paths import PROJECT_ROOT


def git_commit() -> str:
    """Current commit hash, with '-dirty' if tracked files have changes."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def recent_commits(n: int = 15) -> list[dict]:
    try:
        out = subprocess.run(
            ["git", "log", f"-{n}", "--pretty=format:%h%x09%ad%x09%s", "--date=format:%Y-%m-%d %H:%M"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    rows = []
    for line in out.splitlines():
        parts = line.split("\t", 2)
        if len(parts) == 3:
            rows.append({"commit": parts[0], "date": parts[1], "message": parts[2]})
    return rows
