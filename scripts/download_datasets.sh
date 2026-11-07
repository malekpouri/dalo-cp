#!/usr/bin/env bash
# Fetches the public datasets from their sources and checks them. No data is redistributed in this repository.
# Usage: scripts/download_datasets.sh [DATASET_DIR]   (default: ./datasets; files go under dalo_cp/raw/)
set -euo pipefail
ROOT="${1:-datasets}/dalo_cp/raw"
mkdir -p "$ROOT/elec2" "$ROOT/trace"

fetch() { # url path sha256
  if [ -f "$2" ] && echo "$3  $2" | sha256sum -c --status; then echo "ok: $2"; return; fi
  curl -fL --retry 5 -C - -o "$2.partial" "$1"
  echo "$3  $2.partial" | sha256sum -c -
  mv "$2.partial" "$2"
  echo "downloaded: $2"
}

# ELEC2, OpenML dataset 151.
fetch "https://openml.org/data/v1/download/2419/electricity.arff" "$ROOT/elec2/electricity.arff" \
  2d86fbc74c69a5c04bcde0119bce280dd6d8efa0115987edc75b77ccf024d7f2

# TRACE (arXiv:2601.20307), about 4.9 GB, Apache-2.0. Only data/trace_encrypted.txt is read, as text.
# The archive also holds pickle files. This code never loads them, and neither should you: pickle can execute code.
fetch "https://huggingface.co/datasets/alimamaTech/TRACE/resolve/main/data.zip" "$ROOT/trace/data.zip" \
  21cc87fd8cebc912a1cded19e4f681097d6b4902f5b9c8df8bb7f39ccf46a9c1
