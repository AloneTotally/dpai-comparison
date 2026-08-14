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

def plot_family_best_comparison(master, families=None, early_col="n_t2000", eq_col="n_near_eq", ax=None):
    """One bar per family, at that family's OWN best (R,H) -- not a fixed
    point. Two panels: early-time winner, equilibrium winner. Colored by
    taxonomy group so additive/round/sharp/1D read at a glance.

    Answers: "which single family wins, at each time horizon" -- nothing
    else. Does not try to also show consistency, robustness, or kinetics.
    
    If families=None, uses all families. Pass families=["cone_recessed", ...]
    to filter (e.g. recess-only comparison).
    """
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

    if families is not None:
        master = master[master["family"].isin(families)]

    for col, a, title in [(early_col, ax[0], "Early (t=2,000 s)"),
                           (eq_col, ax[1], "Near-equilibrium")]:
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


def plot_isolate_variable_merged(master, families=None, metric="n_near_eq",
                                  vary="R", fixed_at="median", ax=None):
    """All families on ONE axis instead of 3 separate pairwise panels --
    there are only 5 families, no reason to split them up. Use this
    instead of plot_variable_isolation_panel."""
    if families is None:
        families = list(FAMILY_TAXONOMY.keys())
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5.5))

    other = "H" if vary == "R" else "R"
    for fam in families:
        sub = master[master["family"] == fam].copy()
        if sub.empty:
            continue
        fix_val = sub[other].median() if fixed_at == "median" else fixed_at
        sub = sub[np.isclose(sub[other], fix_val)].sort_values(vary)
        ax.plot(sub[vary] * 1e6, sub[metric], marker="o",
                 color=_color(fam), label=FAMILY_LABELS.get(fam, fam), linewidth=2)

    ax.set_xlabel(f"{vary} (\u00b5m)  [{other} held at its median]")
    ax.set_ylabel(metric)
    ax.set_title(f"All families -- {metric}")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    return ax


def compute_per_family_optimal_trajectory(combined_series, t_start=1, t_end=200_000, n_steps=300):
    """For each family independently, compute the argmax geometry (best
    uptake at that instant) at every time step. Returns a dict keyed by
    family, containing a DataFrame with t, R*, H*.
    
    This is different from the global star trajectory -- it shows HOW EACH
    FAMILY's optimum drifts over time, rather than WHICH FAMILY leads at
    each instant. Much more informative about per-family behavior."""
    time_grid = np.logspace(np.log10(t_start), np.log10(t_end), n_steps)
    families = {}
    seen = set()
    for key in combined_series.keys():
        family_name = key[0]
        if family_name in seen:
            continue
        seen.add(family_name)
        fam_geoms = {k: combined_series[k] for k in combined_series.keys() if k[0] == family_name}
        if not fam_geoms:
            continue
        
        rows = []
        for t in time_grid:
            best_key, best_val = None, -np.inf
            for key, g in fam_geoms.items():
                pts = sorted(g["points"], key=lambda p: p[0])
                if not pts or t < pts[0][0]:
                    continue
                val = np.interp(t, [p[0] for p in pts], [p[1] for p in pts])
                if val > best_val:
                    best_val, best_key = val, key
            if best_key:
                _, (R, H) = best_key
                rows.append({"t": t, "R": R * 1e6, "H": H * 1e6, "n": best_val})
        families[family_name] = pd.DataFrame(rows)
    
    return families


def plot_per_family_optimal_trajectory(family_trajs, ax=None):
    """For each family, plot R*(t) and H*(t) on separate panels, all 5
    families on the same axes (side-by-side within each panel, colored
    by family). Shows whether each family's optimum is stable (horizontal)
    or drifting (sloped), and whether R and H drift together or independently.
    
    E.g. if cone_recessed's R*(t) slopes downward while H*(t) slopes 
    upward, you can see the early-vs-late tradeoff directly -- you don't
    need snapshots, it's a continuous trace."""
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    
    for row_idx, (param, label) in enumerate([("R", "Optimal R (\u00b5m)"),
                                                ("H", "Optimal H (\u00b5m)")]):
        a = ax[row_idx]
        for fam, traj_df in family_trajs.items():
            if traj_df.empty:
                continue
            a.plot(traj_df["t"], traj_df[param], 
                   color=_color(fam), linewidth=1.8, 
                   label=FAMILY_LABELS.get(fam, fam), marker="", alpha=0.85)
        
        a.set_xlabel("t (s, log scale)")
        a.set_ylabel(label)
        a.set_xscale("log")
        a.grid(True, linewidth=0.3, alpha=0.4)
    
    ax[0].legend(fontsize=8, frameon=True, loc="best")
    fig.suptitle("Per-family optimal parameter trajectories -- how R* and H* drift over time")
    return ax


def compute_optimal_trajectory(rank_df):
    """GLOBAL star trajectory: which family holds rank==1 at each instant.
    Use this for "who's winning right now" -- the per-family version above
    is better for "how does each family's strategy shift over time."
    """
    rows = []
    leader_idx = rank_df.values.argmin(axis=1)  # rank 1 = min rank value
    cols = rank_df.columns
    for t, idx in zip(rank_df.index, leader_idx):
        family, (R, H) = cols[idx]
        rows.append({"t": t, "family": family, "R": R * 1e6, "H": H * 1e6})
    return pd.DataFrame(rows)


def plot_optimal_trajectory(traj_df, ax=None):
    """Global star trajectory: which family is optimum at each time.
    Step-like by construction since grid is discrete."""
    if ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)

    for row_idx, (param, label) in enumerate([("R", "Optimal R (\u00b5m)"),
                                                ("H", "Optimal H (\u00b5m)")]):
        a = ax[row_idx]
        # color each segment by the family holding the optimum there
        for fam in traj_df["family"].unique():
            mask = traj_df["family"] == fam
            a.scatter(traj_df.loc[mask, "t"], traj_df.loc[mask, param],
                      color=_color(fam), s=10, label=FAMILY_LABELS.get(fam, fam))
        a.plot(traj_df["t"], traj_df[param], color="lightgray", linewidth=0.8, zorder=0)
        a.set_ylabel(label)
        a.set_xscale("log")
        a.grid(True, linewidth=0.3, alpha=0.4)

    ax[0].legend(fontsize=7, ncol=2, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax[1].set_xlabel("t (s, log scale)")
    ax[0].set_title("Global optimum trajectory -- which family leads at each time")
    return ax


def compute_elasticity_loglog(master, family, metric="n_near_eq"):
    """True elasticity: d(ln metric)/d(ln param), fit by regression across
    ALL 25 points at once (ln metric = a + bR*ln R + bH*ln H). Regression
    handles the uneven (5,10,20,50,80) grid correctly -- unlike averaging
    finite differences, it doesn't care that the gaps between points are
    unequal. bR and bH ARE the elasticities: bR=0.3 means a 1% increase in
    R gives ~0.3% increase in the metric. r2 tells you how much this
    single-number summary is actually trustworthy -- low r2 means the
    real relationship isn't well described by one slope and you should
    look at the local distribution (below) instead.
    """
    sub = master[master["family"] == family]
    lnR, lnH, lnM = np.log(sub["R"]), np.log(sub["H"]), np.log(sub[metric])
    X = np.column_stack([np.ones(len(sub)), lnR, lnH])
    coef, *_ = np.linalg.lstsq(X, lnM, rcond=None)
    a, bR, bH = coef
    pred = X @ coef
    ss_res = np.sum((lnM - pred) ** 2)
    ss_tot = np.sum((lnM - lnM.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"family": family, "elasticity_R": bR, "elasticity_H": bH, "r2": r2}


def build_elasticity_table(master, metric="n_near_eq"):
    return pd.DataFrame([
        compute_elasticity_loglog(master, fam, metric=metric)
        for fam in sorted(master["family"].unique())
    ])


def compute_local_elasticity_distribution(master, family, param="R", metric="n_near_eq"):
    """Instead of ONE elasticity per family, compute one per slice (e.g.
    one R-elasticity for each fixed H value), by log-log regression on
    just that slice. Returns the individual slope estimates so you can
    see the spread -- if they're all similar, the single regression
    number above is a fair summary; if they vary a lot, sensitivity
    genuinely depends on WHERE in the grid you are, and that's worth
    knowing rather than averaging away.
    """
    other = "H" if param == "R" else "R"
    sub = master[master["family"] == family]
    slopes = []
    for fix_val, slice_df in sub.groupby(other):
        if len(slice_df) < 2:
            continue
        x = np.log(slice_df[param].values)
        y = np.log(slice_df[metric].values)
        b, a = np.polyfit(x, y, 1)
        slopes.append({"family": family, "fixed_" + other: fix_val, "elasticity": b})
    return pd.DataFrame(slopes)


def plot_elasticity_distribution(master, metric="n_near_eq", ax=None):
    """Boxplot of local (per-slice) elasticities, one box per family per
    parameter -- shows the spread the single-number bar chart hides."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5.5))

    families = sorted(master["family"].unique())
    data, positions, colors, xticklabels = [], [], [], []
    pos = 0
    for fam in families:
        for param in ("R", "H"):
            dist = compute_local_elasticity_distribution(master, fam, param=param, metric=metric)
            data.append(dist["elasticity"].values)
            positions.append(pos)
            colors.append(_color(fam))
            xticklabels.append(f"{fam}\n{param}")
            pos += 1
        pos += 0.5  # gap between families

    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(xticklabels, fontsize=7, rotation=0)
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.set_ylabel(f"Local elasticity (d ln {metric} / d ln param)")
    ax.set_title("Elasticity spread across the grid -- not just the average")
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.4)
    return ax


def plot_recess_feature_isolation(master, metric="n_near_eq", ax=None):
    """Three controlled recess-only comparisons, showing which FEATURE
    actually drives performance differences among recesses.
    
    Isolates ONE variable at a time:
    - Panel 1: Taper (cone vs cylinder, same roundness+dimensionality)
    - Panel 2: Corner sharpness (cylinder vs square_recessed, same dimensionality)
    - Panel 3: Dimensionality (square_recessed vs grooves, both sharp)
    
    This is the cleanest evidence for "what feature matters" -- not all
    5 families at once, just the 4 recesses in three focused pairs."""
    if ax is None:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    comparisons = [
        (ax[0], "cone_recessed", "cylinder_recessed", "R",
         "Taper: cone (tapered) vs cylinder (straight)"),
        (ax[1], "cylinder_recessed", "square_recessed", "R",
         "Corner sharpness: cylinder (round) vs square (sharp)"),
        (ax[2], "square_recessed", "grooves", "R",
         "Dimensionality: square (2D bounded) vs groove (1D channel)"),
    ]

    for a, fam_a, fam_b, vary, title in comparisons:
        plot_isolate_variable(master, fam_a, fam_b, metric=metric, vary=vary, ax=a)
        a.set_title(title)
        # Remove the "held at median" text, tighten x-label
        a.set_xlabel(f"{vary} (\u00b5m)")

    fig.suptitle(f"What feature drives recess performance? ({metric})", fontsize=11)
    plt.tight_layout()
    return fig, ax


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

def get_best_of_group(master, families, metric="n_near_eq"):
    """Extract the single best (R,H) point from a group of families,
    measured by the given metric. Returns (family_name, R, H, metric_value)."""
    sub = master[master["family"].isin(families)]
    best_row = sub.loc[sub[metric].idxmax()]
    return (best_row["family"], best_row["R"], best_row["H"], best_row[metric])


def plot_recess_vs_additive_auto(master, families_dict, metric="n_near_eq", ax=None):
    """Automatically find the best recess and best pillar, then plot them
    head-to-head. Simpler than having to look up the keys manually.
    
    This IS the "what do you give up by choosing recess" answer.
    
    families_dict is the FAMILIES dict from master_pipeline.py, not
    combined_series (avoids float-key mismatch issues)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5.5))

    # Find best recess and best pillar
    recess_families = ["cone_recessed", "cylinder_recessed", "square_recessed", "grooves"]
    pillar_families = ["square_pillar"]
    
    best_recess_fam, r_r, h_r, n_r = get_best_of_group(master, recess_families, metric=metric)
    best_pillar_fam, r_p, h_p, n_p = get_best_of_group(master, pillar_families, metric=metric)
    
    # Find the matching (R, H) key in families_dict using np.isclose
    def find_matching_key(family_name, target_r, target_h, families_dict, tol=1e-8):
        """Find the (R,H) key in families_dict[family]["series"] that matches
        (target_r, target_h) within floating-point tolerance."""
        series = families_dict[family_name]["series"]
        for key in series.keys():
            if isinstance(key, tuple):
                r, h = key
            else:
                r, h = key, None
            if h is None:
                if np.isclose(r, target_r, rtol=1e-9, atol=tol):
                    return key
            else:
                if np.isclose(r, target_r, rtol=1e-9, atol=tol) and \
                   np.isclose(h, target_h, rtol=1e-9, atol=tol):
                    return key
        # Fallback: if exact match not found, raise with helpful message
        raise KeyError(f"No key in {family_name} matches R={target_r}, H={target_h} "
                       f"(available: {list(series.keys())})")

    recess_key = find_matching_key(best_recess_fam, r_r, h_r, families_dict)
    pillar_key = find_matching_key(best_pillar_fam, r_p, h_p, families_dict)
    
    def get_pts(family_name, key):
        g = families_dict[family_name]["series"][key]
        return sorted(g["points"], key=lambda p: p[0])

    pts_r = get_pts(best_recess_fam, recess_key)
    pts_p = get_pts(best_pillar_fam, pillar_key)

    r_um = r_r * 1e6
    h_um = h_r * 1e6
    r_p_um = r_p * 1e6
    h_p_um = h_p * 1e6
    
    ax.plot(*zip(*pts_r), color=_color(best_recess_fam), linewidth=2.2,
             label=f"Best recess: {FAMILY_LABELS.get(best_recess_fam, best_recess_fam)}\n"
                   f"R={r_um:.0f}\u00b5m, H={h_um:.0f}\u00b5m")
    ax.plot(*zip(*pts_p), color=_color(best_pillar_fam), linewidth=2.2,
             label=f"Best pillar: {FAMILY_LABELS.get(best_pillar_fam, best_pillar_fam)}\n"
                   f"R={r_p_um:.0f}\u00b5m, H={h_p_um:.0f}\u00b5m")

    # Annotate crossover
    ts = sorted(set(t for t, _ in pts_r) | set(t for t, _ in pts_p))
    ts = [t for t in ts if t > 0]
    n_r_interp = np.interp(ts, *zip(*pts_r))
    n_p_interp = np.interp(ts, *zip(*pts_p))
    sign = np.sign(n_p_interp - n_r_interp)
    flips = np.where(np.diff(sign) != 0)[0]
    if len(flips):
        t_cross = ts[flips[0]]
        ax.axvline(t_cross, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.text(t_cross, ax.get_ylim()[1] * 0.90, f"crossover\n~{t_cross:.0f}s",
                fontsize=8, va="top", ha="center", bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="white", alpha=0.7))
    else:
        ax.text(0.98, 0.05, "No crossover\n(pillar leads throughout)",
                transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    gap_pct = (n_p_interp[-1] - n_r_interp[-1]) / n_r_interp[-1] * 100
    gap_mol = n_p_interp[-1] - n_r_interp[-1]
    ax.text(0.02, 0.95, 
            f"Capacity loss if constrained to recesses:\n"
            f"{gap_pct:+.1f}% ({gap_mol:+.2e} mol)",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    ax.set_xscale("log")
    ax.set_xlabel("t (s, log scale)")
    ax.set_ylabel("n (mol)")
    ax.set_title("Fabrication tradeoff: best recess vs. best pillar")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
    return ax, (best_recess_fam, r_r, h_r, n_r, recess_key), (best_pillar_fam, r_p, h_p, n_p, pillar_key)


def plot_recess_vs_additive(combined_series, recess_key, pillar_key, ax=None):
    """Manual version: you pass in the keys directly. Use this if you want
    to compare a specific pair instead of auto-selecting the best."""
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