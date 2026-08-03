"""
texture_classification_analysis.py

Replaces the "everything at once" plots (rank trajectory over 20+ series,
dominance map, robustness-of-top-5) with a small set of figures that each
answer exactly one question, organized around the actual design variables
(corner sharpness, dimensionality, additive vs subtractive) instead of
raw family names.

Drop these functions into advanced_analysis.py (they reuse `interp_at`,
`master`, and `FAMILIES` from master_pipeline.py / advanced_analysis.py --
no new data structures required).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Taxonomy -- the organizing structure for every figure below
# ---------------------------------------------------------------------------

FAMILY_TAXONOMY = {
    "square_pillar":     {"group": "additive", "dim": "2d", "corner": "sharp"},
    "cone_recessed":      {"group": "subtractive", "dim": "2d", "corner": "round"},
    "cylinder_recessed":  {"group": "subtractive", "dim": "2d", "corner": "round"},
    "square_recessed":    {"group": "subtractive", "dim": "2d", "corner": "sharp"},
    "grooves":            {"group": "subtractive", "dim": "1d", "corner": "sharp"},
    # add "edges" here when it's built: dim="1d", corner="sharp" or "round"
    # add "frustum_pillar" here when it's built: group="additive"
}

FAMILY_COLORS = {
    "square_pillar":     "#C0392B",  # additive -- red
    "cone_recessed":      "#5DADE2",  # 2D round, tapered -- light blue
    "cylinder_recessed":  "#1B4F72",  # 2D round, straight-wall -- dark blue
    "square_recessed":    "#8E44AD",  # 2D sharp -- purple
    "grooves":            "#27AE60",  # 1D -- green
}

FAMILY_LABELS = {
    "square_pillar": "Square pillar (additive)",
    "cone_recessed": "Cone recess (2D, round, tapered)",
    "cylinder_recessed": "Cylinder recess (2D, round, straight)",
    "square_recessed": "Square recess (2D, sharp)",
    "grooves": "Groove (1D, sharp)",
}


def _color(family):
    return FAMILY_COLORS.get(family, "#7F8C8D")


# ---------------------------------------------------------------------------
# Figure 1 -- Best-in-class per family, early vs. equilibrium
# ---------------------------------------------------------------------------

def plot_family_best_comparison(master, early_col="n_t2000", eq_col="n_near_eq", ax=None):
    """One bar per family, at that family's OWN best (R,H) -- not a fixed
    point. Two panels: early-time winner, equilibrium winner. Colored by
    taxonomy group so additive/round/sharp/1D read at a glance.

    Answers: "which single family wins, at each time horizon" -- nothing
    else. Does not try to also show consistency, robustness, or kinetics.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 3, figsize=(11, 4.5))

    for col, a, title in [(early_col, ax[0], "Early (t=2,000 s)"),
                           (eq_col, ax[1], "Near-equilibrium"),
                        ('n_t15000', ax[2], "mid early (t=15,000s)")]:
        best_rows = (
            master.sort_values(col, ascending=False)
            .groupby("family", as_index=False)
            .first()
        )
        best_rows = best_rows.sort_values(col, ascending=False)

        colors = [_color(f) for f in best_rows["family"]]
        bars = a.bar(best_rows["family"], best_rows[col], color=colors)
        for bar, (_, row) in zip(bars, best_rows.iterrows()):
            # R/H are stored in meters (e.g. 0.00008) -- convert to um for
            # the label, or "R=0 H=0" is all you get from :.0f on a value
            # that's already < 1.
            r_um = row["R"] * 1e6
            h_um = row["H"] * 1e6
            a.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                   f"R={r_um:.0f}\u00b5m H={h_um:.0f}\u00b5m",
                   ha="center", va="bottom", fontsize=8, rotation=0)

        a.set_title(title)
        a.set_ylabel("n (mol)")
        a.tick_params(axis="x", rotation=20)
        a.grid(True, axis="y", linewidth=0.3, alpha=0.4)

        plt.tight_layout()

    return ax


# ---------------------------------------------------------------------------
# Figure 2 -- Controlled pairwise isolation of ONE design variable at a time
# ---------------------------------------------------------------------------

def plot_isolate_variable(master, family_a, family_b, metric="n_near_eq",
                           vary="R", fixed_at="median", ax=None):
    """Holds every design variable fixed except one, and compares two
    families that differ in exactly that one variable. E.g.:

        plot_isolate_variable(master, "cone_recessed", "cylinder_recessed",
                               vary="R")
        -> isolates TAPER (same corner roundness, same dimensionality)

        plot_isolate_variable(master, "cylinder_recessed", "square_recessed",
                               vary="R")
        -> isolates CORNER SHARPNESS (same dimensionality)

        plot_isolate_variable(master, "square_recessed", "grooves",
                               vary="R")
        -> isolates DIMENSIONALITY (both sharp-cornered)

    This is the actual "what feature matters" evidence -- one variable
    changes, everything else held constant, difference attributed to
    that one variable alone.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    other = "H" if vary == "R" else "R"

    for fam in (family_a, family_b):
        sub = master[master["family"] == fam].copy()
        fix_val = sub[other].median() if fixed_at == "median" else fixed_at
        sub = sub[np.isclose(sub[other], fix_val)]
        sub = sub.sort_values(vary)
        # R/H are stored in meters -- plot in um so the axis reads 5,10,20,50,80
        # instead of 5e-6,1e-5,...
        ax.plot(sub[vary] * 1e6, sub[metric], marker="o",
                 color=_color(fam), label=FAMILY_LABELS.get(fam, fam), linewidth=2)

    fix_val_um = fix_val * 1e6
    ax.set_xlabel(f"{vary} (\u00b5m)  [{other} held at {fix_val_um:.1f}\u00b5m]")
    ax.set_ylabel(metric)
    ax.set_title(f"Isolating {vary}: {FAMILY_LABELS.get(family_a, family_a)} vs "
                 f"{FAMILY_LABELS.get(family_b, family_b)}")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    return ax


def plot_isolate_by_descriptor(master, family_a, family_b, metric="n_near_eq",
                                descriptor="SA_V", ax=None):
    """Same idea as plot_isolate_variable, but the x-axis is a geometric
    descriptor (SA_V or L_c) instead of raw R. No "hold H fixed" filter --
    every (R,H) point for both families is plotted.

    Why this matters: if two shape families collapse onto the SAME curve
    here despite being on separate curves in plot_isolate_variable (which
    holds H fixed and varies raw R), that means the descriptor -- not the
    shape label -- is the thing actually governing performance, and
    "square_recessed beats cylinder_recessed" would really mean "whichever
    shape reaches a given SA/V beats the other," which is a stronger and
    more general claim than a per-family ranking.

    If they do NOT collapse -- i.e. two families with equal SA_V still
    give different metric values -- that's evidence shape has a real
    independent effect beyond what SA/V captures (e.g. the corner
    contribution flagged as unproven).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    for fam in (family_a, family_b):
        sub = master[master["family"] == fam].sort_values(descriptor)
        ax.plot(sub[descriptor], sub[metric], marker="o", linestyle="",
                 color=_color(fam), label=FAMILY_LABELS.get(fam, fam),
                 alpha=0.85, markersize=6)

    ax.set_xlabel(descriptor + (" (1/m)" if descriptor == "SA_V" else " (m)"))
    ax.set_ylabel(metric)
    ax.set_title(f"By {descriptor}: {FAMILY_LABELS.get(family_a, family_a)} vs "
                 f"{FAMILY_LABELS.get(family_b, family_b)}")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    return ax


def plot_descriptor_isolation_panel(master, metric="n_near_eq", descriptor="SA_V"):
    """Descriptor-axis version of plot_variable_isolation_panel -- same
    three pairwise comparisons, x-axis swapped from raw R to SA_V (or
    L_c). Run both versions side by side: raw-R tells you which shape to
    build, descriptor-axis tells you whether shape matters at all once
    you control for SA/V."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    plot_isolate_by_descriptor(master, "cone_recessed", "cylinder_recessed",
                                metric=metric, descriptor=descriptor, ax=axes[0])
    plot_isolate_by_descriptor(master, "cylinder_recessed", "square_recessed",
                                metric=metric, descriptor=descriptor, ax=axes[1])
    plot_isolate_by_descriptor(master, "square_recessed", "grooves",
                                metric=metric, descriptor=descriptor, ax=axes[2])
    fig.suptitle(f"Does shape matter beyond {descriptor}? ({metric})")
    plt.tight_layout()
    return fig, axes


def plot_variable_isolation_panel(master, metric="n_near_eq"):
    """The three controlled comparisons in one figure, side by side:
    taper, corner sharpness, dimensionality. Run this instead of the old
    all-25-geometries overlay -- three clean one-variable tests instead
    of one crowded plot."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    plot_isolate_variable(master, "cone_recessed", "cylinder_recessed",
                           metric=metric, vary="R", ax=axes[0])
    plot_isolate_variable(master, "cylinder_recessed", "square_recessed",
                           metric=metric, vary="R", ax=axes[1])
    plot_isolate_variable(master, "square_recessed", "grooves",
                           metric=metric, vary="R", ax=axes[2])
    fig.suptitle(f"Which feature actually matters? ({metric})")
    plt.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Figure 3 -- Best recess vs. best additive, head-to-head
# ---------------------------------------------------------------------------

def plot_recess_vs_additive(combined_series, recess_key, pillar_key, ax=None):
    """combined_series is keyed by (family, (R, H)) as built in
    build_master_table(). recess_key / pillar_key are those same tuples
    for the two winning geometries identified from Figures 1-2.

    Just two curves. Annotates the crossover time (if any) and the final
    % gap. This is the fabrication-constraint answer: what do you give
    up, concretely, by choosing the best buildable recess over the best
    (currently unbuildable) additive geometry.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5.5))

    def get_pts(key):
        g = combined_series[key]
        return sorted(g["points"], key=lambda p: p[0])

    pts_r = get_pts(recess_key)
    pts_p = get_pts(pillar_key)

    ax.plot(*zip(*pts_r), color=_color(recess_key[0]), linewidth=2,
             label=f"{FAMILY_LABELS.get(recess_key[0], recess_key[0])} "
                   f"R={recess_key[1][0]:.0f} H={recess_key[1][1]:.0f}")
    ax.plot(*zip(*pts_p), color=_color(pillar_key[0]), linewidth=2,
             label=f"{FAMILY_LABELS.get(pillar_key[0], pillar_key[0])} "
                   f"R={pillar_key[1][0]:.0f} H={pillar_key[1][1]:.0f}")

    # crossover: first t where the curves swap leader
    ts = sorted(set(t for t, _ in pts_r) | set(t for t, _ in pts_p))
    ts = [t for t in ts if t > 0]
    n_r = np.interp(ts, *zip(*pts_r))
    n_p = np.interp(ts, *zip(*pts_p))
    sign = np.sign(n_p - n_r)
    flips = np.where(np.diff(sign) != 0)[0]
    if len(flips):
        t_cross = ts[flips[0]]
        ax.axvline(t_cross, color="gray", linestyle="--", linewidth=1)
        ax.text(t_cross, ax.get_ylim()[1] * 0.95, f"  crossover ~{t_cross:.0f}s",
                fontsize=8, va="top")

    gap_pct = (n_p[-1] - n_r[-1]) / n_r[-1] * 100
    ax.text(0.02, 0.95, f"Pillar leads eq. capacity by {gap_pct:.1f}%",
            transform=ax.transAxes, fontsize=9, va="top")

    ax.set_xscale("log")
    ax.set_xlabel("t (s, log scale)")
    ax.set_ylabel("n (mol)")
    ax.set_title("Best buildable recess vs. best additive")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
    return ax


# ---------------------------------------------------------------------------
# Figure 4 -- Sensitivity, fixed: elasticity-normalized so R and H
# (16x vs 7x swept range) are actually comparable
# ---------------------------------------------------------------------------

def compute_sensitivity_elasticity(master, family, metric="n_near_eq"):
    """Same intent as the old build_sensitivity_table, but normalizes R
    and H to [0,1] over their OWN swept range before differentiating, so
    a parameter swept over a wider absolute range doesn't automatically
    look more "sensitive" just because of how it was sampled. This is
    what should be compared against the t~L^2 / depth-dominant claim --
    the old metric (mean|slope| x range) mechanically favored whichever
    parameter had the wider sweep, independent of the underlying physics.
    """
    sub = master[master["family"] == family]
    pivot = sub.pivot_table(index="H", columns="R", values=metric)
    pivot = pivot.sort_index().sort_index(axis=1)

    R = pivot.columns.values.astype(float)
    H = pivot.index.values.astype(float)
    R_norm = (R - R.min()) / (R.max() - R.min())
    H_norm = (H - H.min()) / (H.max() - H.min())

    # d(metric)/dR_norm, averaged over H; d(metric)/dH_norm, averaged over R
    dM_dR = np.gradient(pivot.values, R_norm, axis=1)
    dM_dH = np.gradient(pivot.values, H_norm, axis=0)

    return {
        "family": family,
        "sensitivity_R": np.nanmean(np.abs(dM_dR)),
        "sensitivity_H": np.nanmean(np.abs(dM_dH)),
    }


def build_sensitivity_table_elasticity(master, metric="n_near_eq"):
    rows = []
    for family in sorted(master["family"].unique()):
        try:
            rows.append(compute_sensitivity_elasticity(master, family, metric=metric))
        except Exception as e:
            print(f"[warning] skipping {family}: {e}")
    return pd.DataFrame(rows)


def plot_sensitivity_elasticity(sens_df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(sens_df))
    width = 0.35
    colors = [_color(f) for f in sens_df["family"]]
    ax.bar(x - width / 2, sens_df["sensitivity_R"], width, color=colors, alpha=0.55, label="R")
    ax.bar(x + width / 2, sens_df["sensitivity_H"], width, color=colors, alpha=1.0, label="H")
    ax.set_xticks(x)
    ax.set_xticklabels(sens_df["family"], rotation=20, ha="right")
    ax.set_ylabel("Elasticity-normalized sensitivity\n(d(metric)/d(param, normalized to [0,1]))")
    ax.set_title("Parameter sensitivity -- R and H on equal footing")
    ax.legend()
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.4)
    return ax
