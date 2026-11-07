"""Paper figures F1-F11, tables and generated numbers. Everything here reads frozen results; no number is typed by hand (PROJECT_BRIEF §15, §16)."""

from __future__ import annotations

import glob
import json
import subprocess
from multiprocessing import Pool

import numpy as np
import yaml

from ..utils.paths import PAPER_DIR, RESULTS_DIR, RUNS_DIR
from . import style

FIG = PAPER_DIR / "figures"
DALO, ORACLE = "dalo_agei25_e0.05_b", "dalo_oracle"
FAM = ["regional_aci_oracle", "regional_aci", "delayed_aci", "ipw_aci", "dtaci", "olcp", "aci", "split_cp", "static_lcp", "weighted_cp"]
DESIGN_NAME = {"a01_local_geometric": "A1 local, geometric", "a02_local_fixed": "A2 local, fixed", "a03_smooth_alternating": "A3 smooth, alternating", "a04_smooth_gradient": "A4 smooth, gradient",
               "a05_nonsmooth_fast_only": "A5 non-smooth", "a06_gradual": "A6 gradual", "a07_recurring": "A7 recurring", "a08_variance": "A8 variance", "a09_covariate_and_mean": "A9 covariate + mean",
               "a10_y_dependent_delay": "A10 $Y$-dependent delay", "a11_heavy_tailed_delay": "A11 heavy-tailed delay", "a12_continuous_x_delay": "A12 continuous $X$ delay", "elec2": "ELEC2"}


def family(label):
    if label == DALO:
        return "dalo_cp"
    if label == ORACLE:
        return "dalo_oracle"
    for f in FAM:
        if label.startswith(f):
            return f
    return None


def fname(label):
    return style.method_style(family(label) or label)["label"]


def _save(fig, name):
    import matplotlib.pyplot as plt

    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=220)
    plt.close(fig)


def _bins(args):
    rd, t_lo, t_hi, width, region = args
    d = np.load(rd + "/rounds.npz")
    cov = np.abs(d["y"] - d["yhat"]) <= d["q"]
    t, reg = d["t"], d["region"]
    m = (t >= t_lo) & (t < t_hi) & ((reg == region) if region is not None else True)
    b = ((t[m] - t_lo) // width).astype(int)
    n = (t_hi - t_lo) // width
    if region is None:
        k = int(reg.max()) + 1
        hit, cnt = np.zeros((k, n)), np.zeros((k, n))
        np.add.at(hit, (reg[m], b), cov[m])
        np.add.at(cnt, (reg[m], b), 1)
        return hit, cnt
    return np.bincount(b, cov[m], n), np.bincount(b, minlength=n).astype(float)


def _pooled(prefix, label, t_lo, t_hi, width, region):
    dirs = [x[:-13] for x in glob.glob(str(RUNS_DIR / f"{prefix}__{label}__*" / "summary.json"))]
    with Pool(10) as pool:
        res = pool.map(_bins, [(d, t_lo, t_hi, width, region) for d in dirs], chunksize=8)
    hit, cnt = sum(r[0] for r in res), sum(r[1] for r in res)
    return hit / np.maximum(cnt, 1)


def f1():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(style.DOUBLE_COL_IN, 1.8))
    t0 = 20
    rng = np.random.default_rng(3)
    rows = {1: ("Fast region", 3, style.OKABE_ITO["blue"]), 0: ("Slow region", 16, style.OKABE_ITO["vermillion"])}
    for y, (name, mean, col) in rows.items():
        for t in np.arange(2, 46, 3) + (0 if y else 1):
            d = 1 + rng.geometric(1 / mean)
            ax.plot([t], [y], "o", ms=3, color=col, zorder=3)
            if t + d <= 50:
                ax.annotate("", xy=(t + d, y + 0.28), xytext=(t, y + 0.02), arrowprops={"arrowstyle": "-", "color": col, "lw": 0.5, "alpha": 0.55, "connectionstyle": "arc3,rad=-0.25"})
                ax.plot([t + d], [y + 0.28], "|", ms=4, color=col)
        ax.text(-0.5, y, name, ha="right", va="center")
    ax.axvline(t0, color="0.3", lw=0.9, ls="--")
    ax.text(t0 + 0.4, 1.62, "shift in the slow region", fontsize=7, color="0.2")
    ax.axvspan(t0, t0 + 16, ymin=0.02, ymax=0.3, color="0.88", zorder=0)
    ax.text(t0 + 8, -0.33, "post-shift labels not yet arrived", ha="center", fontsize=7, color="0.2")
    ax.set_xlim(-9, 50)
    ax.set_ylim(-0.5, 1.8)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Round (dot: interval issued; tick above: its label arrives)")
    _save(fig, "F1_timeline")


def f2():
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(style.DOUBLE_COL_IN, 2.2))
    boxes = {"x": (0.02, 0.62, "Input $X_t$\nregion $r$"), "q": (0.24, 0.62, "Threshold\n$q_t = b_r + c_r + g_r$"), "c": (0.50, 0.62, "Issue interval\n$\\hat f(X_t) \\pm q_t$"), "p": (0.76, 0.62, "Pending queue\n(input known)"),
             "a": (0.76, 0.12, "Label $i$ arrives\nat $i + D_i$"), "u": (0.50, 0.12, "Update correction\n$c_r \\leftarrow c_r + \\eta(\\mathrm{err}_i - \\alpha)$"), "s": (0.24, 0.12, "Age-weighted\nscore quantile $b_r$"), "b": (0.02, 0.12, "Borrow from\nfresher regions $g_r$")}
    w, h = 0.2, 0.25
    for x, y, t in boxes.values():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.005,rounding_size=0.02", fc="#F4F4F2", ec="0.45", lw=0.7))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7)

    def arrow(a, b, dy=False):
        xa, ya, _ = boxes[a]
        xb, yb, _ = boxes[b]
        if dy:
            start, end = (xa + w / 2, ya + (0 if ya > yb else h)), (xb + w / 2, yb + (h if ya > yb else 0))
        else:
            start, end = (xa + (w if xb > xa else 0), ya + h / 2), (xb + (0 if xb > xa else w), yb + h / 2)
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "-|>", "color": "0.3", "lw": 0.8})

    for a, b in (("x", "q"), ("q", "c"), ("c", "p"), ("a", "u"), ("u", "s"), ("s", "b")):
        arrow(a, b)
    arrow("p", "a", True)
    arrow("s", "q", True)
    arrow("b", "x", False) if False else None
    ax.annotate("", xy=(0.27, 0.62), xytext=(0.10, 0.37), arrowprops={"arrowstyle": "-|>", "color": "0.3", "lw": 0.8})
    ax.annotate("", xy=(0.40, 0.62), xytext=(0.55, 0.37), arrowprops={"arrowstyle": "-|>", "color": "0.3", "lw": 0.8})
    ax.set_xlim(0, 1)
    ax.set_ylim(0.05, 0.92)
    ax.axis("off")
    _save(fig, "F2_workflow")


def f3():
    import matplotlib.pyplot as plt

    labs = [(DALO, 1.6), ("regional_aci_g0.05_p", 1.2), ("olcp_g0.02", 1.2), ("aci_g0.02_p", 1.2), (ORACLE, 1.2)]
    fig, axes = plt.subplots(1, 2, figsize=(style.DOUBLE_COL_IN, 2.5), sharey=True)
    t0, lo, hi, w = 3000, 2500, 5000, 50
    x = (np.arange((hi - lo) // w) + 0.5) * w + lo - t0
    for ax, (pre, title) in zip(axes, (("m5_a01_local_geometric", "A1: geometric delay, mean 100"), ("m5_a02_local_fixed", "A2: fixed delay 100"))):
        for lab, lw in labs:
            st = style.method_style(family(lab))
            ax.plot(x, _pooled(pre, lab, lo, hi, w, 1), color=st["color"], ls=st["ls"], lw=lw, label=st["label"])
        ax.axhline(0.9, color="0.6", lw=0.6)
        ax.axvline(0, color="0.4", lw=0.6, ls=":")
        ax.set_title(title)
        ax.set_xlabel("Rounds since the shift")
    axes[0].set_ylabel("Coverage in the slow region")
    axes[1].legend(frameon=False, loc="lower right")
    _save(fig, "F3_recovery")


def f4():
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    fig, axes = plt.subplots(1, 3, figsize=(style.DOUBLE_COL_IN, 2.3), sharey=True)
    norm = TwoSlopeNorm(vmin=0.55, vcenter=0.9, vmax=1.0)
    lo, hi, w = 5000, 9500, 300
    im = None
    for ax, lab in zip(axes, (DALO, "regional_aci_g0.05_p", "olcp_g0.02")):
        m = _pooled("m5_a03_smooth_alternating", lab, lo, hi, w, None)
        im = ax.imshow(m, aspect="auto", origin="lower", cmap="PuOr", norm=norm, extent=(lo - 6000, hi - 6000, -0.5, 5.5))
        ax.axvline(0, color="k", lw=0.6, ls=":")
        ax.set_title(fname(lab))
        ax.set_xticks([0, 3000])
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("Region (odd: slow)")
    axes[0].set_yticks(range(6))
    fig.supxlabel("Rounds since the shift", y=-0.02, fontsize=7.5)
    fig.colorbar(im, ax=axes, shrink=0.9, label="Coverage (0.90 neutral)")
    _save(fig, "F4_heatmap")


def f5(sens):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(style.DOUBLE_COL_IN, 2.5), sharey=True)
    fams = {"dalo_agei25_e0.05_b": "dalo_cp", "regional_aci_g0.05_p": "regional_aci", "olcp_g0.02": "olcp", "aci_g0.02_p": "aci"}
    ms = (1, 5, 20, 50, 100, 200, 400)
    for ax, kind in zip(axes, ("fixed", "geometric")):
        t = sens["f5"][kind]
        for lab, f in fams.items():
            st = style.method_style(f)
            ax.plot(ms, [t[f"{lab}|{m}"]["worst"] for m in ms], color=st["color"], ls=st["ls"], marker="o", ms=3, label=st["label"], lw=1.6 if f == "dalo_cp" else 1.1)
        ax.axhline(t[f"{ORACLE}|1"]["worst"], color=style.OKABE_ITO["vermillion"], lw=0.7, ls=":")
        ax.text(1.1, t[f"{ORACLE}|1"]["worst"] + 0.035, "DALO-CP, no delay", fontsize=6, color="0.25", ha="left")
        ax.set_xscale("log")
        ax.set_title(f"{kind.capitalize()} delay")
    axes[0].set_ylabel("Worst region-window coverage")
    axes[1].legend(frameon=False, loc="lower left")
    fig.supxlabel("Slow-region mean delay (fast region: 1)", y=-0.02, fontsize=7.5)
    _save(fig, "F5_gap_vs_delay")


def f6(res, sens):
    import matplotlib.pyplot as plt

    var = [(DALO, "Full DALO-CP", style.OKABE_ITO["vermillion"], "o"), ("abl_base", "Base (no age)", "0.45", "s"), ("abl_age", "Age only", style.OKABE_ITO["blue"], "^"),
           ("abl_age_arrival", "Age + arrival", style.OKABE_ITO["green"], "v"), ("abl_full_guard", "Full + guard", style.OKABE_ITO["orange"], "D")]
    tab = dict(sens["ablation"])
    for part, name in (("trace_p", "TRACE, price regions"), ("trace_d", "TRACE, delay regions")):
        tab[name] = {l: res["trace"][part]["table"][l] for l, *_ in var}
    names = list(tab)
    fig, ax = plt.subplots(figsize=(style.DOUBLE_COL_IN, 3.2))
    y = np.arange(len(names))[::-1]
    for i, d in enumerate(names):
        ax.axhline(y[i], color="0.92", lw=0.6, zorder=0)
    for lab, name, col, mk in var:
        ax.scatter([tab[d][lab]["worst"] for d in names], y, s=18, color=col, marker=mk, label=name, zorder=3, edgecolors="white", linewidths=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([DESIGN_NAME.get(d, d) for d in names])
    ax.set_xlabel("Worst region-window coverage")
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.45, 1.08), columnspacing=1, handletextpad=0.2)
    _save(fig, "F6_ablation")


def _tradeoff(ax, table, title):
    pts = [(fname(l), v["width"], v["worst"], family(l)) for l, v in table.items() if family(l) not in (None, "dalo_oracle", "regional_aci_oracle")]
    handles = {}
    for name, x, y, f in pts:
        st = style.method_style(f)
        handles[name] = ax.scatter(x, y, s=34 if f == "dalo_cp" else 15, color=st["color"], marker="D" if f == "dalo_cp" else "o", zorder=4 if f == "dalo_cp" else 3, edgecolors="white", linewidths=0.4)
    ax.set_title(title)
    return handles


def f7(res):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(style.DOUBLE_COL_IN, 4.2))
    panels = [(res["designs"]["a01_local_geometric"]["table"], "A1"), (res["designs"]["a03_smooth_alternating"]["table"], "A3"), (res["designs"]["a08_variance"]["table"], "A8"),
              (res["designs"]["elec2"]["table"], "ELEC2"), (res["trace"]["trace_p"]["table"], "TRACE, price regions"), (res["trace"]["trace_d"]["table"], "TRACE, delay regions")]
    handles = {}
    for ax, (t, title) in zip(axes.flat, panels):
        handles.update(_tradeoff(ax, t, title))
    for ax in axes[1]:
        ax.set_xlabel("Mean width")
    for ax in axes[:, 0]:
        ax.set_ylabel("Worst window coverage")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.legend(handles.values(), handles.keys(), frameon=False, ncol=5, loc="upper center", columnspacing=1, handletextpad=0.2)
    _save(fig, "F7_tradeoff")


def f8(diag):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(style.DOUBLE_COL_IN, 2.4), sharey=True)
    cols = [style.OKABE_ITO["blue"], style.OKABE_ITO["green"], style.OKABE_ITO["vermillion"]]
    names = ("Low", "Middle", "High")
    for ax, (part, title) in zip(axes, (("P", "Regions by predicted price"), ("D", "Regions by predicted delay"))):
        for i, r in enumerate(diag["partitions"][part]["regions"]):
            xs = sorted(int(k) for k in r["survival"])
            ax.plot(xs, [r["survival"][str(k)] for k in xs], marker="o", ms=3, color=cols[i], label=f"{names[i]} (mean {r['mean_h']:.0f} h)")
        ax.set_xscale("log")
        ax.set_xticks([1, 6, 24, 72, 168])
        ax.set_xticklabels(["1 h", "6 h", "1 d", "3 d", "7 d"])
        ax.set_title(title)
        ax.set_xlabel("Time since the click")
        ax.legend(frameon=False, loc="upper right", fontsize=6)
    axes[0].set_ylabel("Share of labels not yet arrived")
    _save(fig, "F8_trace_delay")


def f9(res):
    import matplotlib.pyplot as plt

    panels = [(res["designs"]["elec2"]["table"], "ELEC2 (simulated delays)"), (res["trace"]["trace_p"]["table"], "TRACE, price regions"), (res["trace"]["trace_d"]["table"], "TRACE, delay regions")]
    fig, axes = plt.subplots(1, 3, figsize=(style.DOUBLE_COL_IN, 3.0), sharex=False)
    for ax, (t, title) in zip(axes, panels):
        rows = sorted(((v["worst"], fname(l), family(l)) for l, v in t.items() if family(l) and family(l) != "regional_aci_oracle"), key=lambda r: r[0])
        for i, (w, name, f) in enumerate(rows):
            hi = f in ("dalo_cp", "dalo_oracle")
            ax.plot([min(r[0] for r in rows) - 0.01, w], [i, i], color="0.88", lw=0.6, zorder=1)
            ax.scatter(w, i, s=24 if hi else 13, color=style.OKABE_ITO["vermillion"] if hi else "0.35", marker="o" if f != "dalo_oracle" else "D", zorder=3)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[1] for r in rows], fontsize=6)
        ax.set_title(title, fontsize=7)
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(2))
        ax.tick_params(axis="x", labelsize=6)
    fig.supxlabel("Worst region-window coverage", fontsize=7.5)
    fig.tight_layout()
    _save(fig, "F9_real_data")


def f10(sens):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(style.DOUBLE_COL_IN, 2.3))
    a1, a3 = sens["f10"]["a01_local_geometric"], sens["f10"]["a03_smooth_alternating"]
    cols = {"A1": style.OKABE_ITO["vermillion"], "A3": style.OKABE_ITO["blue"]}
    base = {"eta": 0.05, "tau": 25}
    for ax, key, vals, xl in ((axes[0], "eta", (0.02, 0.05, 0.1, 0.2, 0.5), r"Relative step size $\eta$"), (axes[1], "tau", (10, 25, 50, 100, 200), r"Age scale $\tau$")):
        for name, t in (("A1", a1), ("A3", a3)):
            ys = [t[DALO if v == base[key] else f"f10_{key}{v}"]["worst"] for v in vals]
            ax.plot(vals, ys, marker="o", ms=3, color=cols[name], label=name)
            ax.plot([base[key]], [t[DALO]["worst"]], "o", ms=5, mfc="none", mec="k", mew=0.6)
        ax.set_xscale("log")
        ax.set_xticks(vals)
        ax.set_xticklabels([str(v) for v in vals])
        ax.minorticks_off()
        ax.set_xlabel(xl)
    axes[0].set_ylabel("Worst window coverage")
    axes[0].legend(frameon=False, title="Circle: frozen setting")
    ks = (2, 3, 6, 10)
    for lab, f in ((DALO, "dalo_cp"), ("regional_aci_g0.05_p", "regional_aci"), ("olcp_g0.02", "olcp")):
        st = style.method_style(f)
        axes[2].plot(ks, [sens["f10"][f"regions{k}"][lab]["worst"] for k in ks], marker="o", ms=3, color=st["color"], ls=st["ls"], label=st["label"])
    axes[2].set_xlabel("Number of regions")
    axes[2].legend(frameon=False)
    fig.tight_layout()
    _save(fig, "F10_sensitivity")


def f11(stream):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(style.DOUBLE_COL_IN, 2.6))
    rows = [r for r in stream["analytic"] if r["target"] == 0.85]
    cols = {100: style.OKABE_ITO["blue"], 400: style.OKABE_ITO["vermillion"], 1600: style.OKABE_ITO["green"]}
    for ax, (kind, ms) in zip(axes, (("geometric", (100, 400, 1600)), ("fixed", (100, 400)))):
        for m in ms:
            rr = sorted((r for r in rows if r["kind"] == kind and r["m"] == m), key=lambda r: r["T"])
            ts = [r["T"] for r in rr]
            ax.plot(ts, [r["L_detector"] for r in rr], color=cols[m], marker="o", ms=3)
            ax.plot(ts, [r["L_lower"] for r in rr], color=cols[m], ls="--", marker="s", ms=3)
        ax.set_xscale("log")
        ax.set_xticks([10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7, 10 ** 8])
        ax.minorticks_off()
        ax.set_xlabel("Horizon $T$")
        ax.set_title(f"{kind.capitalize()} delay")
    axes[0].set_ylabel("Lag $L$ for coverage 0.85")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    hs = [Line2D([], [], color=c, label=f"$m = {m}$") for m, c in cols.items()] + [Line2D([], [], color="0.3", marker="o", ms=3, label="Detector"), Line2D([], [], color="0.3", ls="--", marker="s", ms=3, label="Lower bound")]
    fig.legend(handles=hs, frameon=False, ncol=5, loc="upper center")
    _save(fig, "F11_stream_bound")


def _tex(v):
    return str(v).replace("_", r"\_")


def tables_and_numbers(res, sens, diag, stream, theory, tv):
    n = {}

    def put(k, v):
        n[k] = v

    g = res["robustness_gate"]
    r = res["robustness"]
    put("RoneMargin", f"{r['held_out']['margin']:.3f}")
    put("RoneLo", f"{r['held_out']['margin_lo']:.3f}")
    put("RoneHi", f"{r['held_out']['margin_hi']:.3f}")
    put("RtwoMargin", f"{r['all']['margin']:.3f}")
    put("RtwoLo", f"{r['all']['margin_lo']:.3f}")
    put("RtwoHi", f"{r['all']['margin_hi']:.3f}")
    put("RfourMean", f"{g['R4_mean_width_ratio']:.3f}")
    put("MaxRegretDalo", f"{r['all']['max_regret']['dalo_cp']:.3f}")
    put("MaxRegretRegional", f"{r['all']['max_regret']['regional_aci']:.3f}")
    put("MaxRegretOlcp", f"{r['all']['max_regret']['olcp']:.3f}")
    put("LooMin", f"{min(r['all']['margin_leave_one_design_out'].values()):.3f}")
    d = res["designs"]
    for k, key in (("AOne", "a01_local_geometric"), ("AThree", "a03_smooth_alternating"), ("AFour", "a04_smooth_gradient"), ("Elec", "elec2")):
        put(f"Gain{k}", f"{d[key]['gain']:+.3f}".replace("-", "$-$").replace("+", "$+$"))
    put("ElecWidth", f"{d['elec2']['width_ratio']:.3f}")
    tp, td = res["trace"]["trace_p"], res["trace"]["trace_d"]
    for k, t in (("P", tp), ("D", td)):
        put(f"Regret{k}", f"{t['regret']['dalo_cp']:.3f}")
        put(f"Gain{k}", f"{t['gain']:.3f}".replace("-", "$-$"))
        put(f"GainLo{k}", f"{t['gain_lo']:.3f}".replace("-", "$-$"))
        put(f"GainHi{k}", f"{t['gain_hi']:.3f}".replace("-", "$-$"))
        put(f"WidthRatio{k}", f"{t['width_ratio']:.3f}")
        put(f"GapDalo{k}", f"{t['gap_to_no_delay'][DALO]['mean']:.3f}")
        put(f"GapRegional{k}", f"{t['gap_to_no_delay']['regional_aci_g0.005_p']['mean']:.3f}")
        put(f"Base{k}", f"{t['table']['abl_base']['worst']:.3f}")
        put(f"Age{k}", f"{t['table']['abl_age']['worst']:.3f}")
        put(f"Full{k}", f"{t['table'][DALO]['worst']:.3f}")
        put(f"RatioDelay{k}", f"{diag['partitions'][k]['mean_ratio']:.2f}" if k == "D" else f"{diag['partitions'][k]['mean_ratio']:.3f}")
    put("TraceRhoOne", f"{diag['C1']['mean']:.2f}")
    put("TraceRhoOneMin", f"{diag['C1']['min']:.2f}")
    put("TraceRhoTwo", f"{diag['C2']['mean_abs']:.3f}")
    put("TraceMedianMin", f"{diag['delay_overall']['median_h'] * 60:.0f}")
    put("TraceMeanH", f"{diag['delay_overall']['mean_h']:.0f}")
    put("TraceMae", f"{diag['info']['holdout_mae']:.3f}")
    put("TraceMaeConst", f"{diag['info']['constant_predictor_holdout_mae']:.3f}")
    put("TraceFit", f"{diag['info']['n_fit']:,}".replace(",", "{,}"))
    from ..utils.paths import DATA_PROCESSED

    prov = json.loads((DATA_PROCESSED / "trace" / "provenance.json").read_text())
    elec = json.loads((DATA_PROCESSED / "elec2" / "provenance.json").read_text())
    put("TraceRows", f"{prov['rows']:,}".replace(",", "{,}"))
    put("ElecN", f"{elec['n']:,}".replace(",", "{,}"))
    put("ElecTrain", f"{elec['n_train']:,}".replace(",", "{,}"))
    put("ElecVal", f"{elec['n_validation']:,}".replace(",", "{,}"))
    put("TraceStream", f"{diag['info']['n_stream_days']:,}".replace(",", "{,}"))
    b = theory["bound"]
    put("ChkN", str(b["comparisons"]))
    put("ChkNontrivial", str(b["nontrivial"]))
    put("ChkGeo", str(b["nontrivial_geometric"]))
    put("ChkStarNontrivial", str(theory["tv_star"]["nontrivial"]))
    m = stream["methods"]
    put("StreamN", str(m["comparisons"]))
    put("StreamNontrivial", str(m["nontrivial_neyman_pearson"]))
    put("StreamMaxPct", f"{100 * m['max_required_over_I']:.0f}")
    f5 = sens["f5"]
    for k, lab in (("Dalo", DALO), ("Regional", "regional_aci_g0.05_p"), ("Olcp", "olcp_g0.02")):
        put(f"FiveFixed{k}", f"{f5['fixed'][f'{lab}|400']['worst']:.3f}")
        put(f"FiveGeo{k}", f"{f5['geometric'][f'{lab}|400']['worst']:.3f}")
        put(f"FiveOne{k}", f"{f5['geometric'][f'{lab}|1']['worst']:.3f}")
    put("FiveFixedRegionalMarginal", f"{f5['fixed']['regional_aci_g0.05_p|400']['marginal']:.3f}")
    gaps = [sens["f10"][f"regions{k}"]["olcp_g0.02"]["worst"] - sens["f10"][f"regions{k}"][DALO]["worst"] for k in (2, 3, 6, 10)]
    put("RegionsGapLo", f"{min(gaps):.3f}")
    put("RegionsGapHi", f"{max(gaps):.3f}")
    # Example of Theorem 2.2
    from ..theory import bounds as B

    q0 = B.q_for_coverage(0.9, 0, 0.3)
    put("ExKappa", f"{float(B.kappa(q0, 0.6, 0.3)):.3f}")
    put("ExKappaBar", f"{float(B.kappa_envelope(q0, 0.6, 0.3)):.3f}")
    put("ExWidth", f"{B.q_for_coverage(0.9, 0.6, 0.3):.3f}")
    mx = max(max(v.get("frac_inf", 0) for v in d[k]["table"].values()) for k in d)
    put("MaxInfPct", f"{100 * mx:.2f}")
    nv = json.load(open(RESULTS_DIR / "summaries" / "m5_nearvalid.json"))
    rows = []
    for band, v in nv.items():
        h, a = v["held_out"], v["all"]
        rows.append(f"{float(band):.3f} & {h['margin']:.3f} & [{h['lo']:.3f}, {h['hi']:.3f}] & {a['margin']:.3f} & [{a['lo']:.3f}, {a['hi']:.3f}] \\\\")
    (PAPER_DIR / "tables" / "nearvalid.tex").write_text("\\begin{tabular}{rrcrc}\n\\toprule\nBand & \\multicolumn{2}{c}{Held-out designs (R1)} & \\multicolumn{2}{c}{All designs (R2)}\\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n& Margin & 95\\% interval & Margin & 95\\% interval\\\\\n\\midrule\n" + "\n".join(rows).replace("-", "$-$") + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    put("NVoneAllLo", f"{nv['0.01']['all']['lo']:.3f}".replace("-", "$-$"))
    put("NVElecMarg", f"{d['elec2']['table'][DALO]['marginal']:.3f}")
    tt = json.load(open(RESULTS_DIR / "summaries" / "m5x_trace_tau.json"))
    order = [("x_tau5", "5"), ("x_tau10", "10"), (DALO, "25 (frozen)"), ("x_tau50", "50"), ("x_tau100", "100"), ("x_tau200", "200"), ("x_tau1000", "1000"), ("x_tau_none", "none (uniform)"),
             ("x_no_ipw", "25, no arrival weights"), ("_ref_regional_aci", "Regional ACI (reference)"), ("_ref_static_lcp", "Static LCP (reference)")]
    rows = [f"{name} & {tt['p'][l]['worst']:.3f} & {tt['p'][l]['width']:.3f} & {tt['d'][l]['worst']:.3f} & {tt['d'][l]['width']:.3f} \\\\" for l, name in order]
    tau_tab = ("\\begin{tabular}{lrrrr}\n\\toprule\nAge scale $\\tau$ & \\multicolumn{2}{c}{Price regions} & \\multicolumn{2}{c}{Delay regions}\\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
               "& Worst & Width & Worst & Width\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    best = {pp: max((l for l, _ in order if not l.startswith("_")), key=lambda l: tt[pp][l]["worst"]) for pp in ("p", "d")}
    gen0 = PAPER_DIR / "generated"
    gen0.mkdir(exist_ok=True)
    (gen0 / "tau_text.tex").write_text(
        "Outside the pre-specified protocol, and requested by a reviewer. DALO-CP is rerun on TRACE with its age scale varied, and with age weighting but no arrival weights; all other settings are frozen, with the same 10 seeds. "
        "Short age scales are harmful, arrival weights matter, and weaker age weighting helps, but no variant reaches static LCP. These are point values over thinnings of one stream.\n"
        "\\begin{table}[!htb]\\centering\\small\n\\caption{Exploratory: worst-window coverage and mean width of DALO-CP on TRACE by age scale.}\n" + tau_tab + "\\end{table}\n", encoding="utf-8")
    nm = dict(order)
    (gen0 / "tau_main.tex").write_text(
        f"worst-window coverage falls to {tt['p']['x_tau5']['worst']:.3f} and {tt['d']['x_tau5']['worst']:.3f} at $\\tau=5$, and rises with weaker age weighting, the best variants reaching "
        f"{tt['p'][best['p']]['worst']:.3f} and {tt['d'][best['d']]['worst']:.3f}, with uniform weights or $\\tau=1000$; without arrival weights it is {tt['p']['x_no_ipw']['worst']:.3f} and {tt['d']['x_no_ipw']['worst']:.3f}. "
        f"No variant reaches static localized conformal ({tt['p']['_ref_static_lcp']['worst']:.3f}, {tt['d']['_ref_static_lcp']['worst']:.3f}). "
        "This is consistent with labels that arrive soon being unrepresentative under strongly skewed delay, so that strong age weighting concentrates on them, but it does not establish the cause (Supplementary Section~S3).\n", encoding="utf-8")
    gen = PAPER_DIR / "generated"
    gen.mkdir(exist_ok=True)
    (gen / "numbers.tex").write_text("% Generated by delayed_local_cp.figures.paper; do not edit.\n" + "".join(f"\\newcommand{{\\v{k}}}{{{v}}}\n" for k, v in n.items()), encoding="utf-8")

    tab = PAPER_DIR / "tables"
    tab.mkdir(exist_ok=True)
    rows, rows_ci = [], []
    short = {"regional_aci": "Reg. ACI", "olcp": "OLCP", "aci": "ACI", "static_lcp": "Static LCP", "dtaci": "DtACI", "split_cp": "Split CP", "weighted_cp": "Weighted CP", "delayed_aci": "Del. ACI", "ipw_aci": "IPW-ACI"}
    for key, name in DESIGN_NAME.items():
        s_ = d[key]
        code = name.split(" ")[0]
        rows.append(f"{code} & {short.get(family(s_['comparator']), fname(s_['comparator']))} & {s_['gain']:+.3f} & {s_['width_ratio']:.3f} & {r['all']['dalo_regret_by_design'][key]:.3f} & {r['competitor_regret_by_design']['regional_aci'][key]:.3f} & {r['competitor_regret_by_design']['olcp'][key]:.3f} \\\\")
        rows_ci.append(f"{name} & {fname(s_['comparator'])} & {s_['gain']:+.3f} & [{s_['gain_lo']:+.3f}, {s_['gain_hi']:+.3f}] \\\\")
    fix = lambda t: t.replace("+0.000", "0.000").replace("-0.000", "0.000").replace("+", "$+$").replace("-0", "$-$0")
    (tab / "main_results.tex").write_text("\\begin{tabular}{llrrrrr}\n\\toprule\n& \\multicolumn{3}{c}{DALO-CP vs best baseline} & \\multicolumn{3}{c}{Regret}\\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\nDesign & Baseline & Gain & Width & DALO & Reg. ACI & OLCP\\\\\n\\midrule\n"
                                          + fix("\n".join(rows)) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    (tab / "main_results_ci.tex").write_text("\\begin{tabular}{llrc}\n\\toprule\nDesign & Best near-valid baseline & Gain & Descriptive 95\\% interval\\\\\n\\midrule\n" + fix("\n".join(rows_ci)) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    rows = []
    for part, name in (("trace_p", "Price regions"), ("trace_d", "Delay regions")):
        t = res["trace"][part]["table"]
        for l in (DALO, "abl_base", "regional_aci_g0.005_p", "olcp_g0.005", "static_lcp", ORACLE):
            lab = {"abl_base": "DALO-CP base variant", ORACLE: "DALO-CP, no delay"}.get(l, fname(l))
            rows.append(f"{name} & {lab} & {t[l]['marginal']:.3f} & {t[l]['worst']:.3f} & {t[l]['worst_200']:.3f} & {t[l]['width']:.3f} \\\\")
        rows.append("\\midrule")
    (tab / "trace.tex").write_text("\\begin{tabular}{llrrrr}\n\\toprule\nRegions & Method & Marginal & Worst (100) & Worst (200) & Width\\\\\n\\midrule\n" + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    rows = []
    names = {"alpha05_a01_local_geometric": ("A1", 0.05), "alpha05_a03_smooth_alternating": ("A3", 0.05), "alpha05_a06_gradual": ("A6", 0.05), "alpha05_elec2": ("ELEC2", 0.05),
             "cqr_a01_local_geometric": ("A1, CQR", 0.1), "cqr_a03_smooth_alternating": ("A3, CQR", 0.1), "cqr_a08_variance": ("A8, CQR", 0.1), "cqr_elec2": ("ELEC2, CQR", 0.1)}
    for key, (nm, al) in names.items():
        t = res["sensitivity"][key]
        rows.append(f"{nm} & {al} & {fname(t['comparator'])} & {t['gain']:+.3f} & {t['width_ratio']:.3f} & {'yes' if t['dalo_near_valid'] else 'no'} \\\\")
    (tab / "sensitivity.tex").write_text("\\begin{tabular}{lrlrrc}\n\\toprule\nDesign & $\\alpha$ & Best near-valid baseline & Gain & Width ratio & Near-valid\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    rows = []
    for k, v in stream["detector"].items():
        kind, m = k.split("_")
        p100, p200 = v["power"]["100"], v["power"]["200"]
        rows.append(f"{kind.capitalize()}, {m} & {v['false_alarm']:.3f} & {p100['observed']:.2f} / {p100['guarantee']:.2f} & {p200['observed']:.2f} / {p200['guarantee']:.2f} \\\\")
    (tab / "detector.tex").write_text("\\begin{tabular}{lrcc}\n\\toprule\nDelay & False alarm & $L=100$ & $L=200$\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    rows = []
    a = {(x["T"], x["kind"], x["m"]): x for x in stream["analytic"] if x["target"] == 0.85}
    for kind, m in (("fixed", 100), ("fixed", 400), ("geometric", 100), ("geometric", 400), ("geometric", 1600)):
        rows.append(f"{kind.capitalize()}, {m} & " + " & ".join(f"{a[(T, kind, m)]['L_lower']} / {a[(T, kind, m)]['L_detector']}" for T in (10 ** 4, 10 ** 5, 10 ** 6, 10 ** 8)) + " \\\\")
    (tab / "stream.tex").write_text("\\begin{tabular}{lcccc}\n\\toprule\nDelay & $T=10^4$ & $T=10^5$ & $T=10^6$ & $T=10^8$\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")


def main() -> int:
    style.apply()
    S = RESULTS_DIR / "summaries"
    res, sens, diag, stream, theory = (json.load(open(S / f)) for f in ("m5_results.json", "m5_sensitivity.json", "trace_diagnostics.json", "theory_stream.json", "theory_summary.json"))
    tv = json.load(open(S / "theory_tvstar.json"))
    FIG.mkdir(exist_ok=True)
    f1(); f2(); f3(); f4(); f5(sens); f6(res, sens); f7(res); f8(diag); f9(res); f10(sens); f11(stream)
    tables_and_numbers(res, sens, diag, stream, theory, tv)
    source = {"F1": "illustration", "F2": "illustration", "F3": "m5_a01, m5_a02 runs", "F4": "m5_a03 runs", "F5": "m5_sensitivity.json", "F6": "m5_results.json, m5_sensitivity.json", "F7": "m5_results.json",
              "F8": "trace_diagnostics.json", "F9": "m5_results.json", "F10": "m5_sensitivity.json", "F11": "theory_stream.json"}
    names = {p.stem.split("_")[0]: p.name for p in FIG.glob("F*.pdf")}
    (FIG / "manifest.yaml").write_text("# One entry per paper figure (PROJECT_BRIEF §16). Script: delayed_local_cp.figures.paper.\nfigures:\n" + yaml.safe_dump([{"id": k, "file": names[k], "data": v, "results_tag": "results-v1"} for k, v in source.items()], sort_keys=False), encoding="utf-8")
    return 0
