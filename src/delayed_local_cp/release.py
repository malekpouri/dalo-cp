"""Builds the curated public folder from an allowlist and checks it (PROJECT_BRIEF §13.1). Never commits or pushes."""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import sys
from pathlib import Path

import yaml

from .utils.paths import PROJECT_ROOT

OUT = PROJECT_ROOT / "github_repo"
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".ruff_cache", ".mypy_cache", "*.egg-info")
SECRET = re.compile(r"ghp_[A-Za-z0-9]{20,}|github_pat_\w{20,}|sk-[A-Za-z0-9-]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KE[Y]-----|hf_[A-Za-z0-9]{30,}")
HOST_PATH = re.compile(r"/home/[a-z]\w+|/mnt/stor" r"age|/Users/[A-Za-z]\w+")  # split so this file does not match itself
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]*[A-Za-z]{2,}")
# Addresses in fixtures of third-party text formats and in the licence are not personal data about the authors or subjects.
IGNORED_EMAIL = re.compile(r"@(example\.(com|org)|users\.noreply\.github\.com)$")


def _copy(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, ignore=SKIP, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def build(root: Path, out: Path, spec: dict, source_commit: str, source_tag: str) -> None:
    if not (out / ".git").is_dir():
        raise SystemExit("github_repo/ must be a clone of the remote (it keeps the remote's history); `make release` clones it first")
    for p in out.iterdir():
        if p.name != ".git":
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    for rel in spec["include"]:
        _copy(root / rel, out / rel)
    for src, dst in spec.get("map", {}).items():
        _copy(root / src, out / dst)
    summaries = root / "results" / "summaries"
    for f in sorted(summaries.iterdir()):
        if any(fnmatch.fnmatch(f.name, pat) for pat in spec.get("summaries", [])):
            _copy(f, out / "results" / "summaries" / f.name)
    (out / "RELEASE.md").write_text(f"# Release\n\nBuilt by `make release` from source commit `{source_commit}`, tag `{source_tag}`.\n"
                                    "Files here are generated: change them in the source repository and release again.\n", encoding="utf-8")


def check(out: Path, spec: dict) -> list[str]:
    problems = []
    limit = int(spec.get("max_file_bytes", 10_000_000))
    forbidden = spec.get("forbidden", [])
    for p in sorted(out.rglob("*")):
        rel = p.relative_to(out).as_posix()
        if rel == ".git" or rel.startswith(".git/") or not p.is_file():
            continue
        for pat in forbidden:
            hit = rel.startswith(pat) if pat.endswith("/") else (fnmatch.fnmatch(p.name, pat) or rel == pat)
            if hit:
                problems.append(f"forbidden path: {rel}")
        size = p.stat().st_size
        if size > limit:
            problems.append(f"file over {limit} bytes: {rel} ({size})")
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if SECRET.search(text):
            problems.append(f"possible secret: {rel}")
        m = HOST_PATH.search(text)
        if m:
            problems.append(f"host path {m.group(0)!r}: {rel}")
        if rel not in spec.get("allowed_email_files", []):
            for addr in EMAIL.findall(text):
                if not IGNORED_EMAIL.search(addr):
                    problems.append(f"e-mail address {addr!r}: {rel}")
    return problems


def main() -> int:
    spec = yaml.safe_load((PROJECT_ROOT / "configs" / "release.yaml").read_text(encoding="utf-8"))
    build(PROJECT_ROOT, OUT, spec, os.environ.get("RELEASE_COMMIT", "unknown"), os.environ.get("RELEASE_TAG", "none"))
    problems = check(OUT, spec)
    for line in problems:
        print("RELEASE CHECK FAILED:", line)
    print(f"{len(problems)} problems in {sum(1 for p in OUT.rglob('*') if p.is_file() and '.git' not in p.parts)} files")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
