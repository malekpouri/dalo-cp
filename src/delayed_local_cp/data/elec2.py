"""ELEC2 as a real stream with simulated (semi-synthetic) label delays. See reports/decisions.md (M3 data).

* Target: NSW price at round t. Inputs: day, period and the previous half hour's NSW demand, Victoria price, Victoria
  demand and transfer. Lags of the target are excluded on purpose: past prices are themselves labels, so using them
  as inputs would reveal pending labels.
* Rows before 17424 are dropped (Victoria fields and transfer are constant there). Original row order is kept. The
  ``date`` field is not used: it decreases at 5 rows although the period sequence is regular throughout.
* Split in time: 40% base-model training, 10% validation (score scale, region cut points, hyperparameters), 40%+10%
  rest cut into 10 consecutive stream segments.
* Regions: three equal parts of the day, from the period field. Tertiles of the base prediction were tried first and
  rejected before any method was run: prices drift, so 72% of the stream fell in one region.
"""

from __future__ import annotations

import io
import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from ..delays.models import sample_delays
from ..streams.stream import Stream, apply_cqr
from ..utils.io import atomic_write_bytes, atomic_write_json, sha256_file
from ..utils.paths import DATA_PROCESSED, DATA_RAW

FIRST_VALID_ROW = 17424
N_SEGMENTS = 10
RAW = "elec2/electricity.arff"
PROCESSED = "elec2/stream_v1.npz"


def read_arff(path) -> np.ndarray:
    rows, on = [], False
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if on and line:
                rows.append([float(v) for v in line.split(",")[:8]])
            elif line.lower().startswith("@data"):
                on = True
    return np.asarray(rows)


def prepare() -> dict:
    raw = DATA_RAW / RAW
    a = read_arff(raw)
    day, period, price, demand, vicprice, vicdemand, transfer = (a[:, i] for i in range(1, 8))
    x = np.column_stack([day[1:], period[1:], demand[:-1], vicprice[:-1], vicdemand[:-1], transfer[:-1]])
    y = price[1:]
    x, y = x[FIRST_VALID_ROW:], y[FIRST_VALID_ROW:]
    regions = np.minimum((x[:, 1] * 3).astype(np.int64), 2)
    n = len(y)
    n_tr, n_val = int(0.4 * n), int(0.1 * n)
    model = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(x[:n_tr], y[:n_tr])
    yhat = model.predict(x)
    val = np.arange(n_tr, n_tr + n_val)
    scale = float(np.median(np.abs(y[val] - yhat[val])))
    q_lo = HistGradientBoostingRegressor(loss="quantile", quantile=0.05, max_iter=200, random_state=0).fit(x[:n_tr], y[:n_tr]).predict(x)
    q_hi = HistGradientBoostingRegressor(loss="quantile", quantile=0.95, max_iter=200, random_state=0).fit(x[:n_tr], y[:n_tr]).predict(x)
    # standardize inputs with training statistics
    mu, sd = x[:n_tr].mean(0), x[:n_tr].std(0)
    z = (x - mu) / np.where(sd > 0, sd, 1.0)
    buf = io.BytesIO()
    np.savez_compressed(buf, x=z, y=y, yhat=yhat, regions=regions, q_lo=q_lo, q_hi=q_hi)
    out = DATA_PROCESSED / PROCESSED
    atomic_write_bytes(out, buf.getvalue())
    prov = {"raw_sha256": sha256_file(raw), "rows_raw": len(a), "first_valid_row": FIRST_VALID_ROW, "n": n, "n_train": n_tr,
            "n_validation": n_val, "n_segments": N_SEGMENTS, "regions": "thirds of the day (period field)", "score_scale": scale,
            "train_mae": float(np.mean(np.abs(y[:n_tr] - yhat[:n_tr]))),
            "validation_mae": float(np.mean(np.abs(y[val] - yhat[val]))),
            "constant_predictor_validation_mae": float(np.mean(np.abs(y[val] - np.median(y[:n_tr])))),
            "region_share_stream": np.bincount(regions[n_tr + n_val:], minlength=3).tolist()}
    atomic_write_json(DATA_PROCESSED / "elec2" / "provenance.json", prov)
    return prov


def make_elec2_stream(ds: dict, delay_cfg: dict, seed: int) -> Stream:
    d = np.load(DATA_PROCESSED / PROCESSED)
    prov = json.loads((DATA_PROCESSED / "elec2" / "provenance.json").read_text())
    n_tr, n_val = prov["n_train"], prov["n_validation"]
    seg = ds.get("segment", 0)
    if seg == "validation":
        a, b = n_tr, n_tr + n_val
    else:
        edges = np.linspace(n_tr + n_val, prov["n"], N_SEGMENTS + 1).astype(int)
        a, b = edges[int(seg)], edges[int(seg) + 1]
    x, y, yhat, regions = d["x"][a:b], d["y"][a:b], d["yhat"][a:b], d["regions"][a:b]
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 7, 0 if seg == "validation" else int(seg) + 1]))
    delay = sample_delays(delay_cfg, x, regions, y, rng)
    stream = Stream(x=x, regions=regions, y=y, delay=delay, yhat=yhat, n_regions=3,
                    meta={"name": "elec2", "segment": seg, "shift_starts": [], "shift_regions": [],
                          "score_scale": prov["score_scale"]})
    if ds.get("cqr"):
        return apply_cqr(stream, d["q_lo"][a:b], d["q_hi"][a:b], prov["score_scale"])
    return stream
