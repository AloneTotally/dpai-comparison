# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python (geometry-comparison)
#     language: python
#     name: geometry-comparison
# ---

# %%
# ============================================================================
# ADVANCED ANALYSIS WALKTHROUGH
# Copy-paste this into your notebook AFTER you've already run:
#   - master_pipeline.py (FAMILIES populated, master table built)
#   - preprocessing.py is imported
# ============================================================================

# %%
# %% [Cell 1] Imports and shared setup

# %load_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from master_pipeline import leadership_segments  # reused for family-level timeline too

R_COLORS = plt.get_cmap("tab10").colors  # or your own list — needed by
                                          # get_color()/single_family_timeline

from preprocessing import build_family_entry, fmt_time, rank_at, check_stability
from master_pipeline import FAMILIES, build_master_table, leadership_duration_scores
from single_family_timeline import leadership_timeline_report
 
# Columns that are metrics, not geometry parameters -- everything else in
# `master` is treated as a parameter for auto-generating plot labels below.
METRIC_COLS = {
    "family", "SA", "V", "SA_V", "L_c",
    "n_t2000", "n_t15000", "n_t134900", "n_eq",
    "pct_eq_t15000", "early_leadership_share",
}
 
 
def build_family_colors(families):
    """One consistent color per family, reused across every plot below so
    a family always reads as the same color regardless of which figure
    you're looking at."""
    cmap = plt.get_cmap("tab10")
    return {name: cmap(i % 10) for i, name in enumerate(sorted(families))}
 
 
def geometry_label(row, param_cols):
    """Build a short label like 'cone: R=20, H=35' from whichever
    parameter columns are actually populated for this row -- handles the
    fact that different families have different parameter names."""
    parts = [f"{c}={row[c]:.3g}" for c in param_cols if pd.notna(row.get(c))]
    return f"{row['family']}: " + ", ".join(parts)


# %%
import importlib
import preprocessing

importlib.reload(preprocessing)

from pathlib import Path
from master_pipeline import FAMILIES, build_master_table

data_dir = Path("data")

FAMILIES["cylinder_recessed"] = preprocessing.build_family_entry(
    uptake_txt=(data_dir / "cylinder-recessed-uptake.txt").read_text(),
    volume_csv_txt=(data_dir / "cylinder-recessed-volume.csv").read_text(),
    area_csv_txt=(data_dir / "cylinder-recessed-area.csv").read_text(),
)

FAMILIES["square_recessed"] = preprocessing.build_family_entry(
    uptake_txt=(data_dir / "square-recessed-uptake.txt").read_text(),
    volume_csv_txt=(data_dir / "square-recessed-volume.csv").read_text(),
    area_csv_txt=(data_dir / "square-recessed-area.csv").read_text(),
)

FAMILIES["cone_recessed"] = preprocessing.build_family_entry(
    uptake_txt=(data_dir / "cone-recessed-uptake.txt").read_text(),
    volume_csv_txt=(data_dir / "cone-recessed-volume.csv").read_text(),
    area_csv_txt=(data_dir / "cone-recessed-area.csv").read_text(),
)

FAMILIES["square_pillar"] = preprocessing.build_family_entry(
    uptake_txt=(data_dir / "square-pillars-uptake.txt").read_text(),
    volume_csv_txt=(data_dir / "square-pillars-volume.csv").read_text(),
    area_csv_txt=(data_dir / "square-pillars-area.csv").read_text(),
)


FAMILIES["edges"] = preprocessing.build_family_entry(
    uptake_txt=(data_dir / "edges-uptake.txt").read_text(),
    volume_csv_txt=(data_dir / "edges-volume.csv").read_text(),
    area_csv_txt=(data_dir / "edges-area.csv").read_text(),
)

FAMILIES["cylindrical_pillars"] = preprocessing.build_family_entry(
    uptake_txt=(data_dir / "cylindrical-pillars-uptake.txt").read_text(),
    volume_csv_txt=(data_dir / "cylindrical-pillars-volume.csv").read_text(),
    area_csv_txt=(data_dir / "cylindrical-pillars-area.csv").read_text(),
)

FAMILIES["grooves"] = preprocessing.build_family_entry(
    uptake_txt=(data_dir / "grooves-uptake.txt").read_text(),
    volume_csv_txt=(data_dir / "grooves-volume.csv").read_text(),
    area_csv_txt=(data_dir / "grooves-area.csv").read_text(),
)

master = build_master_table()

# families = FAMILIES  # plots.py's Cell 5 specifically expects this exact name

# %%
master

# %%
import advanced_analysis as aa


# %%
# This merges EVERY family's geometries into one dict keyed by
# (family_name, R, H), so all analyses score on identical cross-family
# rankings instead of each family only competing against itself.
combined_series, labels = aa.build_combined_series(FAMILIES)

print(f"Combined series has {len(combined_series)} total geometries across {len(FAMILIES)} families")
# Example output: Combined series has 150 total geometries across 6 families

# %%
# ANALYSIS 1: RANK TRAJECTORY + RANK VOLATILITY
# Question: When do rankings change, and when do they stabilize?
# Output: One continuous line per geometry showing how its rank evolves
# =========================================================================

# Compute rank at 300 log-spaced time points from t=1s to t=200,000s
rank_df = aa.compute_rank_trajectory(combined_series, n_steps=300)
print(rank_df.shape)  # (300 samples, 150 geometries)

# Plot: continuous rank trajectory for every geometry that was EVER in top-10
fig, ax = plt.subplots(figsize=(12, 6))
aa.plot_rank_trajectory(rank_df, top_n=10, labels=labels, palette=R_COLORS, ax=ax)
fig.tight_layout()
fig.savefig("rank_trajectory.pdf", bbox_inches="tight")
plt.show()

# Compute how many geometries changed rank at each timestep
volatility = aa.compute_rank_volatility(rank_df)
# Find when volatility hits zero and STAYS zero (e.g., 5 consecutive samples)
stab_time = aa.find_stabilization_time(volatility, zero_run_length=5)
# print(f"Rankings stabilize at t ≈ {stab_time:,.0f} s")

# Plot: volatility over time (literally: number of rank swaps per sample)
fig, ax = plt.subplots(figsize=(10, 3))
aa.plot_rank_volatility(volatility, stabilization_time=stab_time, ax=ax)
fig.tight_layout()
fig.savefig("rank_volatility.pdf", bbox_inches="tight")
plt.show()

# What this means:
# - If stabilization_time is ~10,000 s, rankings are DONE changing by t=t½
# - If it's ~100,000 s, they're still shuffling near equilibrium
# - This directly answers "when do geometry rankings become meaningful"

# %%
# ANALYSIS 2: TIME-TO-X% UPTAKE
# Question: Who reaches 10%, 25%, 50%, 75%, 90% FIRST?
# Output: "Milestones" for each geometry (e.g., t50 = 5,000 s for cone A)
# =========================================================================

# Build table: one row per geometry, columns = [t10, t25, t50, t75, t90]
t2f = aa.build_time_to_fraction_table(FAMILIES)
print(t2f.head())
# Output:
#        family       t10           t50           t90
# 0  cone_recessed   632.387   4170.625   15804.859

# Key insight: Compare the same geometry across families at the SAME milestone
# e.g., all cones vs. all pillars at the t50 milestone

# Plot: "race" showing fastest geometries to reach each milestone
fig, ax = plt.subplots(figsize=(8, 5))
aa.plot_time_to_fraction_race(t2f, top_n_per_fraction=5, ax=ax)
fig.tight_layout()
fig.savefig("time_to_fraction_race.pdf", bbox_inches="tight")
plt.show()

# What this means:
# - If cones are fastest to t25 but pillars are fastest to t75,
#   that's concrete evidence of the "early vs. late" trade-off
# - Much richer than just saying "cone is better at t=2000s"

# %%
# ANALYSIS 3: SENSITIVITY MAP
# Question: Does depth (H) actually matter more than radius (R)?
# Output: A bar chart showing how sensitive each family is to each parameter
# =========================================================================

# Compute sensitivity: how much does n_near_eq change per unit change in R vs. H
sens = aa.build_sensitivity_table(master, metric="n_near_eq")
print(sens[["family", "param1", "param2", "dominant_param", "range_normalized_param1", "range_normalized_param2"]])
# Output:
#         family param1 param2 dominant_param  range_normalized_param1  range_normalized_param2
# 0  cylinder_rec      R      H              H                   0.001                   0.008
# 1  cone_recessed     R      H              H                   0.002                   0.010

# Plot: grouped bars showing param1 vs param2 sensitivity per family
fig, ax = plt.subplots(figsize=(8, 5))
aa.plot_sensitivity_bars(sens, ax=ax)
fig.tight_layout()
fig.savefig("sensitivity.pdf", bbox_inches="tight")
plt.show()

# What this means:
# - If H's bar (right) is consistently taller than R's bar (left),
#   the "depth is the dominant lever" claim is CONFIRMED numerically
# - If they're similar, then radius matters too, and the design space
#   has two competing parameters

# %%
# ANALYSIS 4: ROBUSTNESS OF THE OPTIMUM
# Question: Is the #1 geometry meaningfully better, or just barely ahead?
# Output: Gap between winner and runner-up, as absolute and percentage
# =========================================================================

# Rank every geometry by near-equilibrium uptake; show top 5
robustness = aa.compute_optimum_robustness(master, metric="n_near_eq", top_n=5)
print(robustness[["family", "n_near_eq", "gap_to_best_pct"]])
# Output:
#      family  n_near_eq  gap_to_best_pct
# 0  pillar_sq   0.000256         0.000
# 1  cone_cone   0.000254         0.781
# 2  groove_ch   0.000252         1.562
# 3  pillar_sq   0.000248         3.125
# 4  recess_cy   0.000244         4.688

# Plot: bar chart of top performers with their gap annotations
fig, ax = plt.subplots(figsize=(7, 4))
aa.plot_optimum_robustness(robustness, metric="n_near_eq", ax=ax)
fig.tight_layout()
fig.savefig("optimum_robustness.pdf", bbox_inches="tight")
plt.show()

# What this means:
# - If #1 is 10% better than #2, you have a clear winner
# - If #1 is 0.1% better than #2, everything clusters together,
#   and your design recommendation needs to include fabrication
#   feasibility as the tie-breaker (not just performance)

# %%
# ANALYSIS 5: DOMINANCE MAP
# Question: Which geometries are CONSISTENTLY good vs. briefly spectacular?
# Output: Nested bars: Top10 (widest) → Top5 → Top3 → Top1 (narrowest)
# =========================================================================

# Compute: for each geometry, what share of the time was it in Top1/Top3/Top5/Top10
dominance = aa.compute_dominance_scores(rank_df, thresholds=(1, 3, 5, 10))
print(dominance.head())

# Plot: nested horizontal bars, narrower bars = narrower time in tighter threshold
fig, ax = plt.subplots(figsize=(8, 6))
aa.plot_dominance_map(dominance, labels=labels, top_n=15, ax=ax)
fig.tight_layout()
fig.savefig("dominance_map.pdf", bbox_inches="tight")
plt.show()

# What this means:
# - A geometry with all four bars nearly equal length was consistently great
#   ("I'm always in the top 10, always top 5, almost always top 3, sometimes top 1")
# - A geometry with a wide Top10 bar but thin Top1 bar was "good sometimes but not often"
# - Different design philosophies: pick the consistent winner OR the occasional miracle

# %%
# ANALYSIS 6: PENETRATION DEPTH + GROWTH RATE
# Question: WHY do rankings change? Physical mechanism explanation.
# Output: Two plots per family showing diffusion front and adsorption rate
# =========================================================================

# Pick one family to examine (e.g., the strongest performer)
family_to_examine = "cone_recessed"
fam_series = FAMILIES[family_to_examine]["series"]

# Plot 1: Diffusion penetration depth overlay
# Shows: sqrt(D_eff*t) curve vs. horizontal lines at each H value
# Prediction: curves start to DIVERGE once sqrt(D_eff*t) crosses H
fig, ax = plt.subplots(figsize=(8, 5))
aa.plot_penetration_vs_depth(fam_series, family_name=family_to_examine, ax=ax)
fig.tight_layout()
fig.savefig(f"penetration_{family_to_examine}.pdf", bbox_inches="tight")
plt.show()

# Plot 2: dn/dt (adsorption RATE) vs. n(t) (cumulative uptake)
# Same family, log-x. Shows WHEN each geometry is absorbing fastest.
fig, ax = plt.subplots(figsize=(8, 5))
aa.plot_growth_rate_overlay(fam_series, family_name=family_to_examine, palette=R_COLORS, ax=ax)
fig.tight_layout()
fig.savefig(f"growth_rate_{family_to_examine}.pdf", bbox_inches="tight")
plt.show()

# What this means:
# - Penetration plot: "all geometries behave identically until sqrt(D_eff*t) ≈ H"
# - Growth-rate plot: "cones spike early and decay; pillars sustain longer"
#   → this is the mechanistic REASON for the rank crossover, not just the fact

# %%
# ANALYSIS 7: PERFORMANCE LANDSCAPE EVOLUTION
# Question: How does the OPTIMUM move across R-H space as time passes?
# Output: 2-3 side-by-side heatmaps showing the optimum's trajectory
# =========================================================================

# Side-by-side heatmaps at three representative times
# Uses a SHARED color scale across all panels so you can see the actual
# magnitude change (optimum doesn't just move; the entire landscape shifts)
fig, axes = aa.plot_landscape_evolution(
    master, family_to_examine,
    [("n_t2000", "Early: t=2,000s"),
     ("n_t15000", "Mid: t=15,000s"),
     ("n_near_eq", "Late: near-equilibrium")]
)
fig.savefig(f"landscape_{family_to_examine}.pdf", bbox_inches="tight")
plt.show()

# What this means:
# - Red marker location CHANGES between panels = optimum migrates
# - Panel colors GET BRIGHTER right-to-left = capacity increases
# - If optimum moves from top-left (small R/H) to bottom-right (large R/H),
#   that's evidence of the early-vs-late trade-off playing out spatially

# %%
# ANALYSIS 8: VOLUME-NORMALIZED COMPARISON
# Question: Do grooves only look strong because they remove less volume?
# Output: Scatter plot of n_eq (absolute) vs. n_eq_per_V (capacity density)
# =========================================================================

# This column was added to the master table in master_pipeline.py
# n_eq_per_V = equilibrium uptake / volume removed

# family_colors = build_family_colors(master["family"].unique())
fig, ax = plt.subplots(figsize=(7, 5))
aa.plot_volume_normalized_comparison(master, family_colors=family_colors, ax=ax)
fig.tight_layout()
fig.savefig("volume_normalized.pdf", bbox_inches="tight")
plt.show()

# What this means:
# - Groove with low n_eq but HIGH n_eq_per_V = "saturates small volume quickly"
#   (volume-depletion hypothesis confirmed)
# - Groove with low n_eq AND low n_eq_per_V = "genuinely bad, not just small"
#   (need a different explanation)
# - If ALL geometries cluster along a diagonal, n_eq_per_V is just a scaled
#   version of n_eq (capacity-driven, not volume-removal-driven)

# %%
# SUMMARY: Which analyses answer which questions?
# =========================================================================

# Q1: How do rankings evolve over time?
#   → Rank trajectory, rank volatility, time-to-X%
#
# Q2: Why do rankings change?
#   → Penetration depth, growth-rate, dominance map
#
# Q3: Which parameter matters most?
#   → Sensitivity map
#
# Q4: Is there a real optimum?
#   → Robustness of optimum, landscape evolution
#
# Supporting evidence:
#   → Volume-normalized comparison (tests groove hypothesis)

# %%
recess_families = ["cone_recessed", "cylinder_recessed", "square_recessed", "grooves"]

# ========== FIGURE 1: Which recess wins? ==========
tca.plot_family_best_comparison(master, families=recess_families)
plt.suptitle("Best recess, early vs. equilibrium")
plt.tight_layout()
plt.show()

# ========== FIGURE 2: Why? Feature isolation ==========
tca.plot_recess_feature_isolation(master, metric="n_near_eq")
plt.show()

# Also run at early time to see if the story changes
tca.plot_recess_feature_isolation(master, metric="n_t2000")
plt.show()

# ========== FIGURE 3: Best recess vs. best pillar ==========
# CHANGE: pass FAMILIES instead of combined_series
ax, (best_recess_fam, r_r, h_r, n_r, recess_key), (best_pillar_fam, r_p, h_p, n_p, pillar_key) = \
    tca.plot_recess_vs_additive_auto(master, FAMILIES, metric="n_near_eq")
plt.show()

# Print the summary numbers
print(f"\n=== RECESS vs PILLAR TRADEOFF ===")
print(f"Best recess:  {best_recess_fam:20s} R={r_r*1e6:5.0f}µm H={h_r*1e6:5.0f}µm  n={n_r:.2e} mol")
print(f"Best pillar:  {best_pillar_fam:20s} R={r_p*1e6:5.0f}µm H={h_p*1e6:5.0f}µm  n={n_p:.2e} mol")
print(f"Capacity gap: {(n_p - n_r) / n_r * 100:+.1f}%")

# %%
# %%
from pathlib import Path
import matplotlib.pyplot as plt
import texture_classification_analysis
importlib.reload(texture_classification_analysis)
import texture_classification_analysis as tca

# Create output folder
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

DPI = 300

# =============================================================================
# FIGURE SET 1
# =============================================================================

# Fig 1: Who wins? Early vs equilibrium
tca.plot_family_best_comparison(master)
plt.savefig(FIG_DIR / "Fig1_family_best_comparison.png",
            dpi=DPI, bbox_inches="tight")
plt.savefig(FIG_DIR / "Fig1_family_best_comparison.pdf",
            bbox_inches="tight")
plt.show()


# Fig 2a: Variable isolation (equilibrium)
tca.plot_variable_isolation_panel(master, metric="n_near_eq")
plt.savefig(FIG_DIR / "Fig2a_variable_isolation_near_eq.png",
            dpi=DPI, bbox_inches="tight")
plt.savefig(FIG_DIR / "Fig2a_variable_isolation_near_eq.pdf",
            bbox_inches="tight")
plt.show()


# Fig 2b: Variable isolation (early time)
tca.plot_variable_isolation_panel(master, metric="n_t15000")
plt.savefig(FIG_DIR / "Fig2b_variable_isolation_t15000.png",
            dpi=DPI, bbox_inches="tight")
plt.savefig(FIG_DIR / "Fig2b_variable_isolation_t15000.pdf",
            bbox_inches="tight")
plt.show()


# Fig 4: Sensitivity analysis
sens = tca.build_sensitivity_table_elasticity(
    master,
    metric="n_near_eq"
)

tca.plot_sensitivity_elasticity(sens)

plt.savefig(FIG_DIR / "Fig4_sensitivity_elasticity.png",
            dpi=DPI, bbox_inches="tight")
plt.savefig(FIG_DIR / "Fig4_sensitivity_elasticity.pdf",
            bbox_inches="tight")
plt.show()


# Fig 3: Controlled recess vs additive comparison
combined_series, labels = aa.build_combined_series(FAMILIES)

# tca.plot_recess_vs_additive(
#     combined_series,
#     recess_key=("cylinder_recessed", (8e-5, 3.5e-5)),
#     pillar_key=("square_pillar", (8e-5, 3.5e-5))
# )



plt.savefig(FIG_DIR / "Fig3_recess_vs_additive.png",
            dpi=DPI, bbox_inches="tight")
plt.savefig(FIG_DIR / "Fig3_recess_vs_additive.pdf",
            bbox_inches="tight")
plt.show()


# %%
# =============================================================================
# RECESS-ONLY ANALYSIS
# =============================================================================

recess_families = [
    "cone_recessed",
    "cylinder_recessed",
    "square_recessed",
    "grooves",
]


# Fig 1: Best recess family
tca.plot_family_best_comparison(master, families=recess_families)

plt.suptitle("Best recess, early vs. equilibrium")
plt.tight_layout()

plt.savefig(FIG_DIR / "Fig5_best_recess_family.png",
            dpi=DPI, bbox_inches="tight")
plt.savefig(FIG_DIR / "Fig5_best_recess_family.pdf",
            bbox_inches="tight")
plt.show()


# Fig 2a: Recess feature isolation (equilibrium)
tca.plot_recess_feature_isolation(
    master,
    metric="n_near_eq"
)

plt.savefig(FIG_DIR / "Fig6a_recess_feature_isolation_near_eq.png",
            dpi=DPI, bbox_inches="tight")
plt.savefig(FIG_DIR / "Fig6a_recess_feature_isolation_near_eq.pdf",
            bbox_inches="tight")
plt.show()


# Fig 2b: Recess feature isolation (early time)
tca.plot_recess_feature_isolation(
    master,
    metric="n_t2000"
)

plt.savefig(FIG_DIR / "Fig6b_recess_feature_isolation_t2000.png",
            dpi=DPI, bbox_inches="tight")
plt.savefig(FIG_DIR / "Fig6b_recess_feature_isolation_t2000.pdf",
            bbox_inches="tight")
plt.show()


# Fig 3: Automatically determine best recess vs best pillar
combined_series, labels = aa.build_combined_series(FAMILIES)

ax, (best_recess_fam, r_r, h_r, n_r, recess_key), (best_pillar_fam, r_p, h_p, n_p, pillar_key) = \
    tca.plot_recess_vs_additive_auto(master, FAMILIES, metric="n_t2000")
# plt.show()

plt.savefig(FIG_DIR / "Fig7_best_recess_vs_best_pillar.png",
            dpi=DPI, bbox_inches="tight")
plt.savefig(FIG_DIR / "Fig7_best_recess_vs_best_pillar.pdf",
            bbox_inches="tight")
plt.show()


# =============================================================================
# SUMMARY
# =============================================================================

print("\n=== RECESS vs PILLAR TRADEOFF ===")
print(
    f"Best recess:  {best_recess_fam:20s} "
    f"R={r_r*1e6:5.0f}µm "
    f"H={h_r*1e6:5.0f}µm "
    f"n={n_r:.2e} mol"
)
print(
    f"Best pillar:  {best_pillar_fam:20s} "
    f"R={r_p*1e6:5.0f}µm "
    f"H={h_p*1e6:5.0f}µm "
    f"n={n_p:.2e} mol"
)
print(f"Capacity gap: {(n_p - n_r) / n_r * 100:+.1f}%")

print(f"\nFigures saved to: {FIG_DIR.resolve()}")


# %%
# %load_ext autoreload
# %autoreload 2
import early_regime_fit
importlib.reload(early_regime_fit)
from early_regime_fit import compute_regime_summary, collapse_validity_report, plot_collapse_check, nested_dict_to_long_df
import pandas as pd
import preprocessing
from pathlib import Path

from early_regime_fit import nested_dict_to_long_df
# your existing pipeline already builds a long-format df with these columns
# (adjust to whatever your master_pipeline.py currently outputs)
# df = pd.read_csv("your_combined_sweep_data.csv")

# data_dir = Path("data")
# # Uniform structure: tuple of tuples
# nested_tuple = preprocessing.parse_comsol_txt((data_dir / "edges-uptake.txt").read_text())


# # Pass directly to the constructor
# df = pd.DataFrame(nested_tuple, columns=['R', 'H', 't', 'n'])
# print(df)

df = nested_dict_to_long_df(combined_series)
summary = compute_regime_summary(df, early_window_t=2000, tol=0.05)
summary, by_family = collapse_validity_report(summary, r2_threshold=0.98)
by_family
# # %%
summary = compute_regime_summary(
    df,
    early_window_t=2000,   # fit k using data up to your validated t=2000s point
    tol=0.05                # 5% deviation from sqrt(t) = crossover
)
summary.head()

summary, by_family = collapse_validity_report(summary, r2_threshold=0.98)
# by_family

plot_collapse_check(df, geometry="cone_recessed", R=8e-05, H=3.5e-05)


# %%
# %%
print(df["geometry"].unique())
print(sorted(df["R"].unique()))
print(sorted(df["H"].unique()))

# %%
master

# %%
import sec2_figure_data_export as sec2
sec2.export_all(master)

# %%

# %%
