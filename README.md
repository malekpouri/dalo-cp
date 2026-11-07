# DALO-CP

Code for *Conformal Prediction with Local Guarantees under Distribution Shift and Covariate-Dependent Label Delay* (manuscript in preparation).

Project page: `docs/index.html`. How to reproduce the results: [`docs/reproduce.md`](docs/reproduce.md). Pre-registered protocols: [`docs/protocol.md`](docs/protocol.md).

## The short version

When labels arrive late, and how late depends on the input, a prediction interval can keep its overall coverage while a slow part of the input space has shifted and stays under-covered. This repository has:

- **Theory:** the numerical checks for an explicit limit on the coverage of any threshold method under a general delay law.
- **A method:** DALO-CP, a delay-aware localized online conformal method, with nine baseline families.
- **Experiments:** synthetic streams, ELEC2 with simulated delays, and TRACE with real delays, all from pinned Docker images.

## What the results say

- The pre-registered robustness criterion **passed** on 13 designs with simulated delays or shifts.
- A pre-specified superiority test (gain of at least 0.03) **failed**.
- On real delays (TRACE), DALO-CP **fell behind** several baselines, and a simpler variant did better.

So we do not recommend DALO-CP under strongly skewed real delays without further work. The theoretical limits, and the evidence that delay costs real coverage, are the firmer findings.

## Quick start

```bash
make env       # writes .env; set DATASET_DIR if needed
make build     # pinned image
make test      # test suite
make smoke     # small end-to-end run, about 1 minute
make up        # local dashboard at http://127.0.0.1:8501
```

Requires Docker with Compose v2. An NVIDIA GPU is optional.

## Layout

| Path | What |
|---|---|
| `src/delayed_local_cp/` | Methods, streams, delays, evaluation, theory bounds, figures |
| `configs/` | Every experiment |
| `scripts/` | Config generators, analyses and the dataset download |
| `results/summaries/` | Frozen result summaries behind the paper |
| `dashboard/` | Local run dashboard |
| `tests/` | Test suite |

Some docstrings refer to internal project notes that are not published; the protocols in them are in `docs/protocol.md`.

## Data

Datasets are not redistributed. `scripts/download_datasets.sh` fetches them from their sources and checks SHA-256. TRACE clicks are conditional on conversion, so results do not describe clicks without a purchase. The TRACE archive holds pickle files, which this code never loads.

## Status and AI use

The manuscript has not been peer reviewed. Proofs were reviewed by the authors, not by an outside reader. Claude Code (Anthropic) helped write software, run experiments, and draft text and proofs, and ChatGPT (OpenAI) assisted in rewriting text, under the authors' oversight and final approval.

## Citation and licence

See `CITATION.cff`. MIT licence.
