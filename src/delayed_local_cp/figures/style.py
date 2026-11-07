"""Shared figure style (PROJECT_BRIEF §16): one palette, fixed method styles, journal sizes."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SINGLE_COL_IN = 3.5
DOUBLE_COL_IN = 7.2

# Okabe-Ito colorblind-safe palette.
OKABE_ITO = {
    "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
    "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "black": "#000000",
}

# One entry per method label; add new methods here, never inline in a script.
METHOD_STYLE = {
    "split_cp":            {"color": OKABE_ITO["black"], "ls": ":",  "label": "Rolling split CP"},
    "aci":                 {"color": OKABE_ITO["orange"], "ls": "--", "label": "ACI"},
    "regional_aci":        {"color": OKABE_ITO["blue"], "ls": "-.", "label": "Regional ACI"},
    "regional_aci_oracle": {"color": OKABE_ITO["sky"], "ls": "-",  "label": "Regional ACI (no-delay oracle)"},
    "dalo_cp":             {"color": OKABE_ITO["vermillion"], "ls": "-", "label": "DALO-CP"},
    "dalo_oracle":         {"color": OKABE_ITO["vermillion"], "ls": ":", "label": "DALO-CP (no delay)"},
    "olcp":                {"color": OKABE_ITO["green"], "ls": "--", "label": "OLCP"},
    "static_lcp":          {"color": OKABE_ITO["purple"], "ls": "-.", "label": "Static LCP"},
    "weighted_cp":         {"color": OKABE_ITO["sky"], "ls": "--", "label": "Weighted CP"},
    "dtaci":               {"color": "#7F7F7F", "ls": "-", "label": "DtACI"},
    "delayed_aci":         {"color": OKABE_ITO["yellow"], "ls": "-", "label": "Delayed ACI"},
    "ipw_aci":             {"color": "#A6761D", "ls": "-", "label": "IPW-ACI"},
}


def method_style(label: str) -> dict:
    return METHOD_STYLE.get(label, {"color": OKABE_ITO["purple"], "ls": "-", "label": label})


def apply() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.2,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.bbox": "tight", "savefig.dpi": 200, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
