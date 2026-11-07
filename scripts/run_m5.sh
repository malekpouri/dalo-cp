#!/usr/bin/env bash
# Runs every M5 sweep in order. Safe to re-run after a shutdown: completed runs are skipped.
set -u
mkdir -p runs/_tmp
for f in configs/experiments/m5_*.yaml; do
  n=$(basename "$f" .yaml)
  case "$n" in m5_trace*) continue;; esac
  make experiments EXP="$f" JOBS=10 > "runs/_tmp/$n.out" 2>&1
  echo "$n exit=$? $(tail -1 "runs/_tmp/$n.out")"
done
