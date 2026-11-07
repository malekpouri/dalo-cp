# Reproducing the results

All steps run in Docker from the repository root. Seeds and settings are in `configs/`.

```bash
make env && make build && make test
scripts/download_datasets.sh "$PWD/datasets"          # about 5 GB; set DATASET_DIR in .env to this folder
make prepare NAME=elec2
make prepare NAME=trace
make prepare NAME=trace_models
make analyze NAME=trace                               # TRACE diagnostics
bash scripts/run_m5.sh                                # about 41,800 runs; safe to re-run after a stop
make experiments EXP=configs/experiments/m5_trace_p.yaml JOBS=4
make experiments EXP=configs/experiments/m5_trace_d.yaml JOBS=4
```

Then, in the runner container (`make shell`):

```bash
python scripts/m5_analysis.py          # robustness criterion, TRACE summary
python scripts/m5_sensitivity.py       # delay, settings, region count, ablation
python scripts/theory_checks.py        # numerical checks of the bounds
python scripts/theory_summary.py
python scripts/theory_stream.py 200    # stream-level bound and detector
python -m delayed_local_cp.cli figures # figures, tables and generated numbers
```

The main sweeps took about two hours on 12 CPU cores, and they use about 13 GB under `runs/`. Interrupted runs resume from checkpoints: `make down` stops them safely, and `make resume` restarts them.

The result files behind the paper are in `results/summaries/`, so figures can be checked without rerunning everything, except F3 and F4, which read per-round run outputs.
