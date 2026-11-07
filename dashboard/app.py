"""DALO-CP live dashboard (Streamlit). Reads only files under the project; all
history lives in runs/ and reports/, so restarting the dashboard loses nothing."""

from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psutil
import streamlit as st
import yaml

from delayed_local_cp.evaluation.metrics import covered
from delayed_local_cp.figures.style import method_style
from delayed_local_cp.tracking.registry import Registry
from delayed_local_cp.utils.gitinfo import recent_commits
from delayed_local_cp.utils.io import read_jsonl
from delayed_local_cp.utils.paths import PAPER_DIR, REPORTS_DIR, RUNS_DIR

st.set_page_config(page_title="DALO-CP dashboard", layout="wide")
REFRESH_S = 10


# ---------------------------------------------------------------- helpers
def _num(v):
    if isinstance(v, str):
        return {"inf": math.inf, "-inf": -math.inf, "nan": math.nan}.get(v, math.nan)
    return math.nan if v is None else v


@st.cache_data(ttl=REFRESH_S)
def registry_df() -> pd.DataFrame:
    reg = Registry()
    reg.mark_stale()
    df = pd.DataFrame(reg.list())
    if df.empty:
        return df
    for c in ("created_at", "started_at", "ended_at", "heartbeat"):
        df[c] = pd.to_datetime(df[c], unit="s", errors="coerce")
    return df


def gpu_stats() -> list[dict]:
    try:
        import pynvml

        pynvml.nvmlInit()
        out = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            u = pynvml.nvmlDeviceGetUtilizationRates(h)
            m = pynvml.nvmlDeviceGetMemoryInfo(h)
            out.append({"gpu": i, "util_%": u.gpu, "mem_used_GB": m.used / 1e9, "mem_total_GB": m.total / 1e9})
        return out
    except Exception:  # GPU not visible in this container
        return []


def summaries(run_ids) -> pd.DataFrame:
    rows = []
    for rid in run_ids:
        p = RUNS_DIR / rid / "summary.json"
        if p.exists():
            s = json.loads(p.read_text())
            rows.append({"run_id": rid, "method": s["method"], "seed": s["seed"],
                         "marginal_coverage": _num(s["marginal_coverage"]),
                         "worst_region_window_coverage": _num(s["worst_region_window_coverage"]),
                         "mean_width": _num(s["mean_width"]), "interval_score": _num(s["interval_score"]),
                         **{f"cov_region_{i}": _num(c) for i, c in enumerate(s["regional_coverage"])},
                         **{f"recovery_r{r['region']}_t{r['shift_start']}": _num(r["recovery_time"])
                            for r in s.get("recovery", [])}})
    return pd.DataFrame(rows)


def ci95(x: pd.Series) -> str:
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) == 0:
        return "-"
    m = x.mean()
    if len(x) < 2:
        return f"{m:.3f}"
    h = 1.96 * x.std(ddof=1) / np.sqrt(len(x))
    return f"{m:.3f} ± {h:.3f}"


# ---------------------------------------------------------------- layout
st.title("DALO-CP: delay-aware localized online CP")
tabs = st.tabs(["Overview", "Runs", "Live metrics", "Heatmap", "Comparison", "Figures & paper", "Reports", "Git"])


@st.fragment(run_every=REFRESH_S)
def overview():
    ms_path = REPORTS_DIR / "milestones.yaml"
    ms = yaml.safe_load(ms_path.read_text()) if ms_path.exists() else {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current milestone", ms.get("current", "?"))
    cpu = psutil.cpu_percent(interval=0.2)
    vm = psutil.virtual_memory()
    c2.metric("CPU", f"{cpu:.0f}%")
    c3.metric("RAM", f"{vm.used / 1e9:.1f} / {vm.total / 1e9:.1f} GB")
    g = gpu_stats()
    c4.metric("GPU", f"{g[0]['util_%']}% · {g[0]['mem_used_GB']:.1f} GB" if g else "not visible")
    if ms.get("milestones"):
        st.subheader("Milestones and gates")
        st.dataframe(pd.DataFrame(ms["milestones"]), hide_index=True, use_container_width=True)
    df = registry_df()
    st.subheader("Runs")
    if df.empty:
        st.info("No runs registered yet.")
        return
    st.write(df["status"].value_counts().to_dict())
    active = df[df["status"] == "running"].copy()
    if not active.empty:
        now = pd.Timestamp.now()
        el = (now - active["started_at"]).dt.total_seconds()
        prog = active["progress"].fillna(0).clip(1e-6, 1)
        active["eta_min"] = ((el / prog - el) / 60).round(1)
        st.dataframe(active[["run_id", "method", "seed", "progress", "eta_min", "heartbeat"]],
                     hide_index=True, use_container_width=True)
    queued = int((df["status"] == "queued").sum())
    done = df[df["status"] == "completed"]
    if queued and len(done):
        dur = (done["ended_at"] - done["started_at"]).dt.total_seconds().median()
        st.caption(f"{queued} queued runs; median completed run {dur:.0f}s (sequential ETA ≈ {queued * dur / 60:.0f} min).")


def runs_tab():
    df = registry_df()
    if df.empty:
        st.info("No runs yet.")
        return
    exps = ["(all)"] + sorted(df["experiment"].dropna().unique().tolist())
    exp = st.selectbox("Experiment", exps, key="runs_exp")
    view = df if exp == "(all)" else df[df["experiment"] == exp]
    cols = ["run_id", "experiment", "method", "dataset", "seed", "config_hash", "git_commit", "status",
            "progress", "started_at", "ended_at", "error"]
    st.dataframe(view[cols], hide_index=True, use_container_width=True)
    rid = st.selectbox("Show log for run", view["run_id"].tolist(), key="runs_log")
    if rid:
        log = RUNS_DIR / rid / "run.log"
        if log.exists():
            text = log.read_text(encoding="utf-8", errors="replace")
            st.code(text[-20000:], language="text")
            st.download_button("Download run.log", text, file_name=f"{rid}.log")


@st.fragment(run_every=REFRESH_S)
def live_tab():
    df = registry_df()
    if df.empty:
        st.info("No runs yet.")
        return
    exps = sorted(df["experiment"].dropna().unique().tolist())
    exp = st.selectbox("Experiment", exps, index=len(exps) - 1, key="live_exp")
    sub = df[df["experiment"] == exp]
    seeds = sorted(sub["seed"].unique().tolist())
    seed = st.selectbox("Seed", seeds, key="live_seed")
    rows = []
    for _, r in sub[sub["seed"] == seed].iterrows():
        for rec in read_jsonl(RUNS_DIR / r["run_id"] / "metrics.jsonl"):
            base = {"method": r["method"], "t": rec["t"], "rolling_coverage": rec["rolling_coverage"],
                    "rolling_width": rec["rolling_width"], "backlog": rec["backlog"]}
            for i, c in enumerate(rec["rolling_region_coverage"]):
                base[f"region_{i}"] = c
            for i, a in enumerate(rec["info_age_by_region"]):
                base[f"age_{i}"] = a
            rows.append(base)
    if not rows:
        st.info("No metrics yet.")
        return
    m = pd.DataFrame(rows)
    colors = {k: method_style(k)["color"] for k in m["method"].unique()}
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.line(m, x="t", y="rolling_coverage", color="method", color_discrete_map=colors,
                            title="Rolling marginal coverage"), use_container_width=True)
    c2.plotly_chart(px.line(m, x="t", y="rolling_width", color="method", color_discrete_map=colors,
                            title="Rolling mean width"), use_container_width=True)
    reg_cols = [c for c in m.columns if c.startswith("region_")]
    long = m.melt(id_vars=["method", "t"], value_vars=reg_cols, var_name="region", value_name="coverage")
    st.plotly_chart(px.line(long, x="t", y="coverage", color="method", facet_col="region",
                            color_discrete_map=colors, title="Rolling regional coverage"),
                    use_container_width=True)
    first = m[m["method"] == m["method"].iloc[0]]
    age_cols = [c for c in m.columns if c.startswith("age_")]
    c3, c4 = st.columns(2)
    c3.plotly_chart(px.line(first, x="t", y="backlog", title="Label backlog (pending labels)"), use_container_width=True)
    c4.plotly_chart(px.line(first, x="t", y=age_cols, title="Information age by region (rounds)"),
                    use_container_width=True)


def heatmap_tab():
    df = registry_df()
    done = df[df["status"] == "completed"] if not df.empty else df
    if done.empty:
        st.info("No completed runs yet.")
        return
    rid = st.selectbox("Completed run", done["run_id"].tolist()[::-1], key="heat_run")
    bins = st.slider("Time bins", 10, 200, 50, key="heat_bins")
    d = np.load(RUNS_DIR / rid / "rounds.npz")
    cov = covered(d["y"], d["yhat"] - d["q"], d["yhat"] + d["q"]).astype(float)
    k = int(d["region"].max()) + 1
    edges = np.linspace(0, len(cov), bins + 1).astype(int)
    z = np.full((k, bins), np.nan)
    for b in range(bins):
        sl = slice(edges[b], edges[b + 1])
        for r in range(k):
            m = d["region"][sl] == r
            if m.any():
                z[r, b] = cov[sl][m].mean()
    fig = go.Figure(go.Heatmap(z=z, x=edges[:-1], y=[f"region {r}" for r in range(k)], colorscale="Viridis",
                               zmin=0.5, zmax=1.0, colorbar={"title": "coverage"}))
    fig.update_layout(title=f"Coverage by region and time: {rid}", xaxis_title="round")
    st.plotly_chart(fig, use_container_width=True)


def comparison_tab():
    df = registry_df()
    if df.empty:
        st.info("No runs yet.")
        return
    exps = sorted(df["experiment"].dropna().unique().tolist())
    exp = st.selectbox("Experiment", exps, index=len(exps) - 1, key="cmp_exp")
    s = summaries(df[(df["experiment"] == exp) & (df["status"] == "completed")]["run_id"])
    if s.empty:
        st.info("No completed runs in this experiment.")
        return
    metric_cols = [c for c in s.columns if c not in ("run_id", "method", "seed")]
    table = s.groupby("method")[metric_cols].agg(ci95)
    table.insert(0, "n_seeds", s.groupby("method")["seed"].nunique())
    st.caption("Mean ± 1.96·SE across seeds (normal approximation; paired tests and block bootstrap come with M5).")
    st.dataframe(table, use_container_width=True)


def figures_tab():
    figs = sorted(PAPER_DIR.glob("figures/**/*.png"))
    if not figs:
        st.info("No figure previews yet (run `make figures`).")
    for f in figs:
        st.image(str(f), caption=str(f.relative_to(PAPER_DIR)))
    pdf = PAPER_DIR / "main.pdf"
    if pdf.exists():
        st.download_button("Download latest paper PDF", pdf.read_bytes(), file_name="main.pdf",
                           mime="application/pdf")
        st.caption(f"main.pdf updated {datetime.fromtimestamp(pdf.stat().st_mtime):%Y-%m-%d %H:%M}")
    else:
        st.info("paper/main.pdf not built yet (run `make paper` on the host).")


def reports_tab():
    for name in ("STATE.md", "decisions.md", "environment.md"):
        p = REPORTS_DIR / name
        if p.exists():
            with st.expander(name, expanded=(name == "STATE.md")):
                st.markdown(p.read_text(encoding="utf-8"))
    reps = sorted((REPORTS_DIR / "progress").glob("*.md"), reverse=True)
    st.subheader("Progress reports")
    for p in reps[:30]:
        with st.expander(p.name):
            st.markdown(p.read_text(encoding="utf-8"))


def git_tab():
    commits = recent_commits(30)
    st.dataframe(pd.DataFrame(commits), hide_index=True, use_container_width=True) if commits else st.info("No commits.")


with tabs[0]:
    overview()
with tabs[1]:
    runs_tab()
with tabs[2]:
    live_tab()
with tabs[3]:
    heatmap_tab()
with tabs[4]:
    comparison_tab()
with tabs[5]:
    figures_tab()
with tabs[6]:
    reports_tab()
with tabs[7]:
    git_tab()

st.caption(f"Rendered {time.strftime('%Y-%m-%d %H:%M:%S')}. Live panels refresh every {REFRESH_S}s.")
