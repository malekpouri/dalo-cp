# Pre-registered protocols

Copied unchanged from the project's decision log. The protocol commits predate the runs they govern.

## 2026-11-06: M5 protocol (pre-registered before any M5 run)

**Settings.** Frozen from M3; nothing is tuned in M5. DALO-CP is `dalo_agei25_e0.05_b`. Weighted CP is new, with defaults set here and untuned (window 500, recent 500, refit every 200 rounds, ratio clip 10).

**Fresh data.** Synthetic seeds 3000-3099. ELEC2 uses the same 10 stream segments as M3 with fresh delay seeds 3000-3009, so its *prices* have been seen in M3: ELEC2 is not fully fresh and is reported as such.

**Designs.**
- Selection designs (they influenced settings): A1-A5.
- **Held-out designs (they influenced nothing):** A6 gradual shift, A7 recurring shift, A8 local variance shift, A9 covariate shift with mean jump, A10 `Y`-dependent delay, A11 heavy-tailed delay, A12 continuous `X`-dependent delay.
- ELEC2. TRACE is added under its own protocol after its diagnostics, or kept as a negative control.

**Primary criterion: robustness (author decision D12).** Regret of a family in a design is the highest mean worst region-window coverage among near-valid families minus its own. Near-valid means mean marginal coverage within 0.02 of the target. A family that is not near-valid in a design has infinite regret there. Maximum regret is taken over a set of designs. Margin is the smallest maximum regret among baseline families minus DALO-CP's.
- R1: on the held-out designs, the 95% bootstrap interval of the margin lies above 0 (2,000 replicates; units resampled within each design; seed 20261106).
- R2: the same on all designs (A1-A12, ELEC2).
- R3: DALO-CP is near-valid in every design.
- R4: mean over all designs of DALO-CP's width divided by that design's comparator's width is at most 1.10. Per-design ratios are also reported, because an average can hide one bad design.
The criterion passes only if R1 to R4 all hold. If it fails, stop and report.

**Reported in any case: the brief's M3 test.** +0.03 over the best near-valid baseline with Holm correction, width within 10%, and faster recovery by Wilcoxon with Holm, in A1, A3, A4 and ELEC2. M3 failed this. This criterion cannot be reinterpreted by the outcome of R1 to R4.

**Secondary (descriptive).** Alpha 0.05 (A1, A3, A6, ELEC2; near-valid band 0.93-0.97). CQR scores (A1, A3, A8, ELEC2). Worst-window sizes 50 and 200. Gap against delay heterogeneity, fixed and geometric, means 1 to 400 (F5). One-setting-at-a-time sensitivity: step, age scale, borrowing cap, window, region count (F10). Ablation. Block bootstrap for ELEC2 is computed for M6 tables.

**Known limit.** All synthetic designs were written by the same authors as the method. Held-out means "not used to choose settings", not "unknown to the designers".

## 2026-11-06: TRACE protocol (written before any delay, price or model output was examined)

**What had been seen.** File layout, row count and column names, from the M5 data check. The conversion step reports time and price ranges only.

**Split.** Fit days 0-40, holdout days 41-50, gap days 51-56, stream days 57 onward. This differs from the brief's "fit on days 0 to 50": the holdout is needed for the scale, the region cuts and the diagnostics without touching the stream. Every label from days 0-50 has closed by day 57 under the 7-day attribution window.

**Models.** Target: `log1p(first price)`. Delay target: `log1p(delay in seconds)`. Inputs: hour of day and the 22 hashed categories, target-encoded with five-fold out-of-fold encoding on fit days (smoothing 20; unseen categories get the prior), never used as numbers. Gradient-boosted trees, 200 iterations, on at most 1,000,000 fit rows (uniform, seed 0).

**Regions, fixed before the diagnostics.** Tertiles on the holdout of (P) predicted price and (D) predicted delay. Both are functions of `X` alone. P is the brief's prediction-based representation. D asks whether a practitioner who groups by expected delay sees a different result.

**Diagnostics, on holdout days only.**
- C1, delay depends on `X`: Spearman correlation of predicted and observed log delay, computed per day, has mean at least 0.10 and is positive on all 10 days.
- C2, delay and `Y` given `X`: Spearman correlation of the price residual and the log-delay residual, per day. Reported, not a pass/fail check: coverage is still measured on all rounds and Claim 1 still holds, but Claim 2 assumes delay independent of `Y` given region, so if the mean absolute value is at least 0.05 the paper must say Claim 2 does not cover TRACE.
- C3, regional differences matter (per partition): slowest-to-fastest ratio of mean delay at least 2, and slowest region's mean delay at least 20 stream rounds at the planned thinning. Survival curves are reported (figure F8).

**Outcome rule.** TRACE is primary real-delay evidence only for a partition passing C1 and C3. If no partition passes, TRACE is a negative control: partition P is still run and reported, and real-delay evidence is stated as limited.

**Stream.** Seeds 3000-3009 each thin the stream days uniformly to 50,000 rounds (D9); this is an estimated 2% of stream clicks, and delays are real times mapped to rounds, so `D_t` is the first thinned round at or after the purchase time, at least 1. Inputs to methods: the two predictions, standardized on the holdout. Method settings are the frozen ELEC2 list and the score scale is from the holdout; nothing is tuned on TRACE. Evaluation window 100, burn-in 500.

**Reporting.** All M5 metrics plus the gap to the no-delay reference. No injected shift exists, so there is no recovery metric. TRACE is outside R1-R4: it is reported as separate descriptive evidence and does not change the M5 outcome.

**Selection caveat.** Every row is a click that converted. Results do not describe clicks without a purchase.

**TRACE diagnostic outcome (holdout days only; `results/summaries/trace_diagnostics.json`).** C1 passed (mean Spearman 0.42, minimum 0.35). C2 mean absolute correlation 0.084, above 0.05, so Claim 2 does not cover TRACE and the paper must say so. C3 passed on both partitions: P ratio 2.006, only just above the threshold of 2, and D ratio 3.50. Outcome: both partitions run as primary real-delay evidence, with P's margin stated. Thinning is 50,000 of 2,014,835 stream clicks (2.5%; the protocol's estimate was 2%). Reference added for the gap: Regional ACI with no delay, alongside DALO-CP's own no-delay reference.
