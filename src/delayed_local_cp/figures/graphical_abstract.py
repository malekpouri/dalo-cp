"""Graphical abstract for Pattern Recognition: at least 1328 px wide, longest side horizontal, readable at 13 cm.
Numbers come from frozen result files."""

from __future__ import annotations

import json

import numpy as np

from ..utils.paths import PAPER_DIR, RESULTS_DIR
from . import style


def main() -> int:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    style.apply()
    S = RESULTS_DIR / "summaries"
    res = json.load(open(S / "m5_results.json"))
    stream = json.load(open(S / "theory_stream.json"))
    rmax = res["robustness"]["all"]["max_regret"]
    tp, td = res["trace"]["trace_p"]["regret"]["dalo_cp"], res["trace"]["trace_d"]["regret"]["dalo_cp"]
    gains = [res["brief_gate_0.03"][d]["gain"] for d in res["brief_gate_0.03"]]

    W, H = 5.12, 2.3                       # 13 cm wide; 1536 x 690 px at 300 dpi
    fig = plt.figure(figsize=(W, H))
    ink, muted, slow, fast = "#1d2329", "#59636e", style.OKABE_ITO["vermillion"], style.OKABE_ITO["blue"]
    head = dict(fontsize=8, fontweight="bold", color=ink, ha="left", va="top")
    body = dict(fontsize=6.5, color=ink, ha="left", va="top")

    def panel(x, w, title):
        bg = fig.add_axes([x, 0.04, w, 0.92], zorder=-5)
        bg.set_xticks([]); bg.set_yticks([])
        bg.set_facecolor("#f6f4ef")
        for sp in bg.spines.values():
            sp.set_visible(True); sp.set_color("#d9d5cc"); sp.set_linewidth(0.6)
        fig.text(x + 0.015, 0.93, title, **head)

    # 1. The problem
    panel(0.01, 0.32, "1  Labels arrive late")
    ax = fig.add_axes([0.03, 0.30, 0.28, 0.48])
    rng = np.random.default_rng(3)
    for y, mean, col in ((1, 2, fast), (0, 12, slow)):
        for t in np.arange(1, 30, 2.5):
            d = 1 + rng.geometric(1 / mean)
            ax.plot(t, y, "o", ms=2.6, color=col)
            if t + d <= 32:
                ax.annotate("", xy=(t + d, y + 0.3), xytext=(t, y + 0.02), arrowprops={"arrowstyle": "-", "color": col, "lw": 0.5, "alpha": 0.6, "connectionstyle": "arc3,rad=-0.3"})
    ax.axvline(13, color=muted, lw=0.7, ls="--")
    ax.axvspan(13, 25, ymin=0.05, ymax=0.4, color="#e3ded3", zorder=0)
    ax.text(0, 1.45, "fast region", fontsize=6, color=fast)
    ax.text(0, 0.45, "slow region", fontsize=6, color=slow)
    ax.text(13.4, -0.48, "shift unseen", fontsize=6, color=muted)
    ax.set_xlim(-1, 32)
    ax.set_ylim(-0.6, 1.8)
    ax.axis("off")
    fig.text(0.025, 0.24, "Overall coverage looks fine,\nwhile the slow region stays\nunder-covered after a shift.", **body)

    # 2. The limit
    panel(0.345, 0.32, "2  A limit for any method")
    fig.text(0.36, 0.80, r"Label information by lag $L$:", **body)
    fig.text(0.36, 0.71, r"$I(L)=\sum_r K_r\,p_r\sum_{a\leq L}F_r(a)$", fontsize=7.5, color=ink, ha="left", va="top")
    fig.text(0.36, 0.58, "Coverage after the shift is at most\n" r"$\bar\kappa(\bar q,\delta)+\mathrm{TV}(I(L))$", **body)
    ax2 = fig.add_axes([0.375, 0.13, 0.26, 0.24])
    rows = sorted((r for r in stream["analytic"] if r["kind"] == "geometric" and r["m"] == 400 and r["target"] == 0.85), key=lambda r: r["T"])
    ts = [r["T"] for r in rows]
    ax2.plot(ts, [r["L_detector"] for r in rows], color=slow, lw=1.1, label="detector")
    ax2.plot(ts, [r["L_lower"] for r in rows], color=slow, lw=1.1, ls="--", label="lower bound")
    ax2.set_xscale("log")
    ax2.set_xticks([1e4, 1e6, 1e8])
    ax2.minorticks_off()
    ax2.tick_params(labelsize=5.5, length=2)
    ax2.set_xlabel("horizon $T$", fontsize=5.5, labelpad=1)
    ax2.set_ylabel("lag", fontsize=5.5, labelpad=1)
    ax2.set_facecolor("#f6f4ef")
    ax2.set_ylim(60, 215)
    ax2.set_title("lag for coverage 0.85, geometric delay", fontsize=5.3, color=muted, pad=2)
    ax2.legend(frameon=False, fontsize=5.2, loc="upper left", handlelength=1.4, ncol=2)

    # 3. The case study
    panel(0.68, 0.31, "3  A case study")
    items = [("PASSED", style.OKABE_ITO["green"], f"Robust on 13 simulated-delay\ndesigns: max regret {rmax['dalo_cp']:.3f},\nbest baseline {min(v for k, v in rmax.items() if k != 'dalo_cp'):.3f}"),
             ("FAILED", slow, f"Pre-specified superiority test\n(gain $\\geq$ 0.03): largest\ngain {max(gains):+.3f}"),
             ("BEHIND", slow, f"Behind several baselines on\nreal delays (TRACE): regret\n{tp:.3f} and {td:.3f}")]
    y = 0.82
    for tag, col, txt in items:
        fig.text(0.695, y, tag, fontsize=6, fontweight="bold", color=col, ha="left", va="top")
        fig.text(0.695, y - 0.065, txt, fontsize=5.8, color=ink, ha="left", va="top", linespacing=1.15)
        y -= 0.27
    out = PAPER_DIR / "submission"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "graphical_abstract.pdf", facecolor="white")
    fig.savefig(out / "graphical_abstract.tiff", dpi=300, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(out / "graphical_abstract.png", dpi=300, facecolor="white")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    main()
