"""EXPLORATORY, outside the protocol: DALO-CP's age scale on TRACE. Output: results/summaries/m5x_trace_tau.json."""
import json
import sys

sys.path.insert(0, "scripts")
import m5_analysis as M  # noqa: E402

from delayed_local_cp.utils.paths import RESULTS_DIR  # noqa: E402

out = {}
for part in ("p", "d"):
    import glob
    from collections import defaultdict
    from multiprocessing import Pool

    import numpy as np

    from delayed_local_cp.utils.paths import RUNS_DIR
    dirs = [f.split("/")[-2] for f in glob.glob(str(RUNS_DIR / f"m5x_trace_tau_{part}__*" / "summary.json"))]
    with Pool(10) as pool:
        rows = pool.map(M._one, [(d, 500) for d in dirs], chunksize=4)
    per = defaultdict(list)
    for _, m in rows:
        per[m["label"]].append(m)
    out[part] = {l: {k: float(np.mean([r[k] for r in rs])) for k in ("marginal", "worst", "worst_200", "width")} for l, rs in per.items()}
res = json.load(open(RESULTS_DIR / "summaries" / "m5_results.json"))
for part in ("p", "d"):
    t = res["trace"][f"trace_{part}"]["table"]
    out[part]["_ref_static_lcp"] = {k: t["static_lcp"][k] for k in ("marginal", "worst", "worst_200", "width")}
    out[part]["_ref_regional_aci"] = {k: t["regional_aci_g0.005_p"][k] for k in ("marginal", "worst", "worst_200", "width")}
json.dump(out, open(RESULTS_DIR / "summaries" / "m5x_trace_tau.json", "w"), indent=1)
for part, t in out.items():
    print(part)
    for l, v in sorted(t.items(), key=lambda kv: -kv[1]["worst"]):
        print(f"  {l:24s} marg {v['marginal']:.3f} worst {v['worst']:.3f} w200 {v['worst_200']:.3f} width {v['width']:.3f}")
