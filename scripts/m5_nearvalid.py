"""Exploratory, outside the pre-specified protocol: how the robustness margin depends on the near-valid band (0.02 in the protocol).
Output: results/summaries/m5_nearvalid.json."""
import json
import sys

import numpy as np

sys.path.insert(0, "scripts")
import m5_analysis as M  # noqa: E402

from delayed_local_cp.utils.paths import RESULTS_DIR  # noqa: E402

designs = M.SELECTION + M.HELD_OUT + ["elec2"]
data = {d: M.load(d) for d in designs}
out = {}
for band in (0.01, 0.015, 0.02, 0.025, 0.03):
    M.BAND = band
    res = {}
    for tag, ds in (("held_out", M.HELD_OUT), ("all", designs)):
        tab = M.regret_table(data, ds, 0.9)
        mx = {f: M.maxreg(tab, f) for f in tab}
        rng = np.random.default_rng(20261106)
        diffs = []
        for _ in range(M.B):
            t = M.regret_table(data, ds, 0.9, rng)
            diffs.append(min(M.maxreg(t, f) for f in t if f != "dalo_cp") - M.maxreg(t, "dalo_cp"))
        diffs = np.nan_to_num(np.array(diffs), nan=0.0, posinf=1.0, neginf=-1.0)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        best = min((f for f in mx if f != "dalo_cp"), key=lambda f: mx[f])
        res[tag] = {"margin": mx[best] - mx["dalo_cp"], "lo": float(lo), "hi": float(hi), "best_competitor": best, "dalo_max_regret": mx["dalo_cp"],
                    "not_near_valid": {f: [d for d in ds if not np.isfinite(tab[f][d])] for f in tab}}
    out[str(band)] = res
    print(band, {k: (round(v["margin"], 3), round(v["lo"], 3), v["best_competitor"], round(v["dalo_max_regret"], 3)) for k, v in res.items()},
          "DALO not near-valid in:", res["all"]["not_near_valid"]["dalo_cp"])
json.dump(out, open(RESULTS_DIR / "summaries" / "m5_nearvalid.json", "w"), indent=1, default=float)
