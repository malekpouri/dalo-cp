"""TRACE (arXiv:2601.20307): real delays from post-click purchases. See reports/decisions.md (M5 data).

Only ``data/trace_encrypted.txt`` is read from the archive, as text. The archive's pickle files are never loaded: they
are untrusted and could execute code. One row is one click that led to at least one purchase, so all analysis is
conditional on conversion.

Target: log1p(first transaction price). Delay: first purchase time minus click time. Times are in the file's integer
unit; the unit is checked in the diagnostics.
"""

from __future__ import annotations

import ast
import json
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..utils.io import atomic_write_json, sha256_file
from ..utils.paths import DATA_PROCESSED, DATA_RAW

ARCHIVE = "trace/data.zip"
MEMBER = "data/trace_encrypted.txt"
PROCESSED = "trace/clicks_v1.parquet"
N_FEAT = 22
BATCH = 200_000


def _flush(writer, cols, schema):
    arrays = [pa.array(cols[n], type=schema.field(n).type) for n in schema.names]
    writer.write_table(pa.Table.from_arrays(arrays, schema=schema))
    for v in cols.values():
        v.clear()


def prepare(limit: int | None = None) -> dict:
    raw = DATA_RAW / ARCHIVE
    names = ["click_time", "first_pay_time", "first_price", "n_purchases"] + [f"f{i}" for i in range(N_FEAT)]
    schema = pa.schema([(n, pa.float32() if n == "first_price" else (pa.int16() if n == "n_purchases" else pa.int64())) for n in names])
    out = DATA_PROCESSED / PROCESSED
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".partial")
    cols = {n: [] for n in names}
    n = bad = unsorted_pay = 0
    with zipfile.ZipFile(raw) as z, z.open(MEMBER) as fh, pq.ParquetWriter(tmp, schema, compression="zstd") as w:
        header = fh.readline().decode().rstrip("\n").split("\t")
        ix = {h: i for i, h in enumerate(header)}
        feat = [ix[f"feature_{i}"] for i in range(N_FEAT)]
        for line in fh:
            parts = line.decode().rstrip("\n").split("\t")
            try:
                pays = ast.literal_eval(parts[ix["pay_time"]])
                amts = ast.literal_eval(parts[ix["dirpay_amt"]])
                click = int(parts[ix["click_time"]])
                pays = [pays] if not isinstance(pays, (list, tuple)) else pays
                amts = [amts] if not isinstance(amts, (list, tuple)) else amts
                order = min(range(len(pays)), key=lambda j: pays[j])
                unsorted_pay += int(order != 0)
                cols["click_time"].append(click)
                cols["first_pay_time"].append(int(pays[order]))
                cols["first_price"].append(float(amts[order]))
                cols["n_purchases"].append(len(pays))
                for i, c in enumerate(feat):
                    # 60 bits of the hash: an identifier for the category, never a numeric value
                    cols[f"f{i}"].append(int(parts[c][:15], 16))
            except (ValueError, SyntaxError, IndexError):
                bad += 1
                continue
            n += 1
            if len(cols["click_time"]) >= BATCH:
                _flush(w, cols, schema)
            if limit and n >= limit:
                break
        if cols["click_time"]:
            _flush(w, cols, schema)
    tmp.replace(out)
    t = pq.read_table(out, columns=["click_time", "first_pay_time", "first_price"]).to_pandas()
    prov = {"archive_sha256": sha256_file(raw), "member": MEMBER, "rows": n, "bad_rows": bad, "first_pay_not_first_in_list": unsorted_pay,
            "click_time_min": int(t.click_time.min()), "click_time_max": int(t.click_time.max()),
            "delay_min": int((t.first_pay_time - t.click_time).min()), "delay_max": int((t.first_pay_time - t.click_time).max()),
            "price_min": float(t.first_price.min()), "price_median": float(t.first_price.median()), "price_max": float(t.first_price.max()),
            "click_time_sorted": bool(t.click_time.is_monotonic_increasing), "limit": limit}
    atomic_write_json(DATA_PROCESSED / "trace" / "provenance.json", prov)
    return prov


# --- Models, diagnostics and stream (protocol: reports/decisions.md, "TRACE protocol") ---------------------------------

FIT_END, HOLD_END, STREAM_START = 41, 51, 57
MODELS = "trace/models_v1.npz"
MODEL_INFO = "trace/models_v1.json"
TE_SMOOTH = 20.0
MAX_FIT_ROWS = 1_000_000
STREAM_ROUNDS = 50_000
SECONDS_PER_DAY = 86_400


def time_unit_per_second(span: float) -> int:
    """Infers the file's time unit from the reported 82-day span. Refuses anything else rather than guessing."""
    for unit in (1, 1000):
        if 70 <= span / unit / SECONDS_PER_DAY <= 95:
            return unit
    raise ValueError(f"click_time span {span} does not match about 82 days in seconds or milliseconds")


def te_map(cat: np.ndarray, target: np.ndarray, prior: float, smooth: float = TE_SMOOTH):
    u, inv = np.unique(cat, return_inverse=True)
    sums, counts = np.bincount(inv, target), np.bincount(inv)
    return u, (sums + smooth * prior) / (counts + smooth)


def te_apply(u: np.ndarray, vals: np.ndarray, prior: float, cat: np.ndarray) -> np.ndarray:
    i = np.minimum(np.searchsorted(u, cat), len(u) - 1)
    return np.where(u[i] == cat, vals[i], prior)


def _encode(cats: np.ndarray, fit: np.ndarray, target: np.ndarray, seed: int = 0) -> np.ndarray:
    """Target encoding: out-of-fold on the fit rows, and fit-rows-only maps everywhere else."""
    prior = float(target.mean())
    fold = np.random.default_rng(seed).integers(0, 5, len(fit))
    out = np.empty(cats.shape, dtype=np.float32)
    rest = np.setdiff1d(np.arange(len(cats)), fit, assume_unique=False)
    for c in range(cats.shape[1]):
        col = cats[:, c]
        for k in range(5):
            tr, te = fit[fold != k], fit[fold == k]
            u, v = te_map(col[tr], target[fold != k], prior)
            out[te, c] = te_apply(u, v, prior, col[te])
        u, v = te_map(col[fit], target, prior)
        out[rest, c] = te_apply(u, v, prior, col[rest])
    return out


def _gbt(x, y, rng):
    from sklearn.ensemble import HistGradientBoostingRegressor

    sub = rng.choice(len(y), min(MAX_FIT_ROWS, len(y)), replace=False)
    return HistGradientBoostingRegressor(max_iter=200, random_state=0, early_stopping=False).fit(x[sub], y[sub])


def build_models() -> dict:
    """Fits on days 0-40 only. Stores predictions for holdout and stream days."""
    t = pq.read_table(DATA_PROCESSED / PROCESSED).to_pandas().sort_values("click_time", kind="stable")
    click, pay = t.click_time.to_numpy(), t.first_pay_time.to_numpy()
    unit = time_unit_per_second(float(click.max() - click.min()))
    t0 = int(click.min())
    day = (click - t0) // (unit * SECONDS_PER_DAY)
    delay_s = (pay - click) / unit
    negative = int((delay_s < 0).sum())
    delay_s = np.maximum(delay_s, 0.0)
    y = np.log1p(t.first_price.to_numpy(dtype=np.float64))
    ld = np.log1p(delay_s)
    hour = (((click - t0) / unit) % SECONDS_PER_DAY) / 3600.0
    cats = t[[f"f{i}" for i in range(N_FEAT)]].to_numpy()
    fit = np.flatnonzero(day < FIT_END)
    rng = np.random.default_rng(0)
    preds = {}
    for name, target in (("yhat", y), ("dhat", ld)):
        x = np.column_stack([hour, _encode(cats, fit, target[fit])]).astype(np.float32)
        model = _gbt(x[fit], target[fit], rng)
        preds[name] = model.predict(x)
        del x, model
    keep = np.flatnonzero((day >= FIT_END) & ((day < HOLD_END) | (day >= STREAM_START)))
    hold = keep[day[keep] < HOLD_END]
    cuts = {"P": np.quantile(preds["yhat"][hold], [1 / 3, 2 / 3]).tolist(), "D": np.quantile(preds["dhat"][hold], [1 / 3, 2 / 3]).tolist()}
    info = {"unit_per_second": unit, "t0": t0, "n_fit": len(fit), "n_holdout": len(hold), "n_stream_days": int((day >= STREAM_START).sum()),
            "days": int(day.max()) + 1, "negative_delays_clipped": negative, "cuts": cuts,
            "score_scale": float(np.median(np.abs(y[hold] - preds["yhat"][hold]))),
            "x_mean": [float(preds[k][hold].mean()) for k in ("yhat", "dhat")], "x_std": [float(preds[k][hold].std()) for k in ("yhat", "dhat")],
            "fit_mae": float(np.mean(np.abs(y[fit] - preds["yhat"][fit]))),
            "holdout_mae": float(np.mean(np.abs(y[hold] - preds["yhat"][hold]))),
            "constant_predictor_holdout_mae": float(np.mean(np.abs(y[hold] - np.median(y[fit])))),
            "max_pay_minus_max_click_days": float((pay.max() - click.max()) / unit / SECONDS_PER_DAY)}
    out = DATA_PROCESSED / MODELS
    with open(out.with_suffix(".partial"), "wb") as fh:  # uncompressed: streams load this file on every run
        np.savez(fh, day=day[keep].astype(np.int16), click=click[keep], pay=pay[keep], y=y[keep], delay_s=delay_s[keep],
                 yhat=preds["yhat"][keep], dhat=preds["dhat"][keep])
    out.with_suffix(".partial").replace(out)
    atomic_write_json(DATA_PROCESSED / MODEL_INFO, info)
    return info


def delays_from_times(click: np.ndarray, pay: np.ndarray) -> np.ndarray:
    """A label is available at the first round whose click time is at or after the purchase time, and never in its own round."""
    return np.maximum(np.searchsorted(click, pay, side="left") - np.arange(len(click)), 1)


def regions_from(partition: str, yhat: np.ndarray, dhat: np.ndarray, cuts: dict) -> np.ndarray:
    return np.searchsorted(cuts[partition], yhat if partition == "P" else dhat).astype(np.int64)


def make_trace_stream(ds: dict, seed: int):
    from ..streams.stream import Stream

    info = json.loads((DATA_PROCESSED / MODEL_INFO).read_text())
    d = np.load(DATA_PROCESSED / MODELS)
    m = np.flatnonzero(d["day"] >= STREAM_START)
    n = int(ds.get("n_rounds", STREAM_ROUNDS))
    ix = m[np.sort(np.random.default_rng(np.random.SeedSequence([int(seed), 11])).choice(len(m), n, replace=False))]
    click, pay, y, yhat, dhat = d["click"][ix], d["pay"][ix], d["y"][ix], d["yhat"][ix], d["dhat"][ix]
    x = (np.column_stack([yhat, dhat]) - np.array(info["x_mean"])) / np.array(info["x_std"])
    partition = ds.get("partition", "P")
    return Stream(x=x, regions=regions_from(partition, yhat, dhat, info["cuts"]), y=y, delay=delays_from_times(click, pay), yhat=yhat, n_regions=3,
                  meta={"name": f"trace_{partition}", "shift_starts": [], "shift_regions": [], "score_scale": info["score_scale"]})


def diagnostics() -> dict:
    """Protocol checks C1-C3. Reads holdout days only, plus stream-day predictions and clock times, never stream delays or prices."""
    from scipy import stats

    info = json.loads((DATA_PROCESSED / MODEL_INFO).read_text())
    d = np.load(DATA_PROCESSED / MODELS)
    day = d["day"]
    h = day < HOLD_END
    days = np.unique(day[h])
    y, yhat, dhat, delay_s = d["y"][h], d["yhat"][h], d["dhat"][h], d["delay_s"][h]
    ld, dh = np.log1p(delay_s), day[h]
    rho1 = [float(stats.spearmanr(dhat[dh == k], ld[dh == k]).statistic) for k in days]
    rho2 = [float(stats.spearmanr((y - yhat)[dh == k], (ld - dhat)[dh == k]).statistic) for k in days]
    c1 = bool(np.mean(rho1) >= 0.10 and min(rho1) > 0)
    s = d["day"] >= STREAM_START
    click_s = d["click"][s]
    rounds_per_s = STREAM_ROUNDS / ((click_s.max() - click_s.min()) / info["unit_per_second"])
    grid_h = [1, 6, 24, 72, 168]
    parts = {}
    for name in ("P", "D"):
        reg = regions_from(name, yhat, dhat, info["cuts"])
        stream_reg = regions_from(name, d["yhat"][s], d["dhat"][s], info["cuts"])
        rows = []
        for r in range(3):
            v = delay_s[reg == r]
            rows.append({"share_holdout": float((reg == r).mean()), "share_stream": float((stream_reg == r).mean()), "mean_h": float(v.mean() / 3600),
                         "median_h": float(np.median(v) / 3600), "p90_h": float(np.quantile(v, 0.9) / 3600),
                         "mean_rounds": float(v.mean() * rounds_per_s),
                         "survival": {str(g): float((v > g * 3600).mean()) for g in grid_h}})
        means = [x["mean_h"] for x in rows]
        ratio = max(means) / min(means)
        slowest_rounds = max(x["mean_rounds"] for x in rows)
        ks = float(stats.ks_2samp(delay_s[reg == int(np.argmax(means))], delay_s[reg == int(np.argmin(means))]).statistic)
        parts[name] = {"regions": rows, "mean_ratio": ratio, "slowest_mean_rounds": slowest_rounds, "ks_slowest_fastest": ks,
                       "C3": bool(ratio >= 2 and slowest_rounds >= 20)}
    res = {"info": info, "holdout_days": [int(days[0]), int(days[-1])], "rounds_per_day_at_thinning": rounds_per_s * SECONDS_PER_DAY,
           "C1": {"per_day": rho1, "mean": float(np.mean(rho1)), "min": float(min(rho1)), "pass": c1},
           "C2": {"per_day": rho2, "mean": float(np.mean(rho2)), "mean_abs": float(np.mean(np.abs(rho2))), "claim2_excluded": bool(np.mean(np.abs(rho2)) >= 0.05)},
           "delay_overall": {"mean_h": float(delay_s.mean() / 3600), "median_h": float(np.median(delay_s) / 3600), "p90_h": float(np.quantile(delay_s, 0.9) / 3600), "max_h": float(delay_s.max() / 3600)},
           "partitions": parts, "primary_partitions": [k for k, v in parts.items() if c1 and v["C3"]]}
    res["outcome"] = "primary" if res["primary_partitions"] else "negative_control"
    from ..utils.paths import RESULTS_DIR

    atomic_write_json(RESULTS_DIR / "summaries" / "trace_diagnostics.json", res)
    return res
