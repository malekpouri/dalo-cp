"""M5 descriptive outputs: gap against delay (F5), DALO-CP settings and region count (F10), ablation. Output: results/summaries/m5_sensitivity.json."""
import glob
import json
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import yaml

from delayed_local_cp.evaluation.m3 import run_metrics
from delayed_local_cp.utils.paths import RESULTS_DIR, RUNS_DIR


def one(rd):
    from pathlib import Path

    p = Path(rd)
    m = run_metrics(p, 500)
    cfg = yaml.safe_load((p / "config.yaml").read_text())
    m["grid"] = (cfg.get("grid_point") or {}).get("delay.per_region")
    return m


def table(prefix):
    dirs = sorted(str(Path_) for Path_ in glob.glob(str(RUNS_DIR / f"{prefix}__*")) if glob.glob(Path_ + "/summary.json"))
    with Pool(10) as pool:
        return pool.map(one, dirs, chunksize=32)


def summ(rows):
    a = {k: np.array([r[k] for r in rows], dtype=float) for k in ("marginal", "worst", "width", "recovery") if k in rows[0]}
    return {k: float(v.mean()) for k, v in a.items()} | {"worst_se": float(a["worst"].std(ddof=1) / np.sqrt(len(rows))), "n": len(rows)}


out = {"f5": {}, "f10": {}}
for kind in ("fixed", "geometric"):
    g = defaultdict(list)
    for r in table(f"m5_f5_{kind}"):
        g[(r["label"], r["grid"][1])].append(r)
    out["f5"][kind] = {f"{l}|{m}": summ(v) for (l, m), v in sorted(g.items(), key=lambda kv: (kv[0][0], kv[0][1]))}
for d in ("a01_local_geometric", "a03_smooth_alternating", "regions2", "regions3", "regions6", "regions10"):
    g = defaultdict(list)
    for r in table(f"m5_f10_{d}"):
        g[r["label"]].append(r)
    out["f10"][d] = {l: summ(v) for l, v in sorted(g.items())}
ABL = ["dalo_agei25_e0.05_b", "abl_base", "abl_age", "abl_age_arrival", "abl_full_guard", "dalo_oracle"]
res = json.load(open(RESULTS_DIR / "summaries" / "m5_results.json"))
out["ablation"] = {d: {l: {k: s["table"][l][k] for k in ("marginal", "worst", "width")} for l in ABL if l in s["table"]} for d, s in res["designs"].items()}
json.dump(out, open(RESULTS_DIR / "summaries" / "m5_sensitivity.json", "w"), indent=1)
for kind, t in out["f5"].items():
    print(kind)
    for k, v in t.items():
        print(" ", k, round(v["marginal"], 3), round(v["worst"], 3), round(v["width"], 3))
for d, t in out["f10"].items():
    print(d)
    for k, v in t.items():
        print(" ", k, round(v["marginal"], 3), round(v["worst"], 3), round(v["width"], 3), round(v.get("recovery", 0)))
print("ablation worst")
for d, t in out["ablation"].items():
    print(" ", d[:12], {l.replace("dalo_agei25_e0.05_b", "DALO").replace("abl_", ""): round(v["worst"], 3) for l, v in t.items()}, "DALO width/base", round(t["dalo_agei25_e0.05_b"]["width"] / t["abl_base"]["width"], 3))
