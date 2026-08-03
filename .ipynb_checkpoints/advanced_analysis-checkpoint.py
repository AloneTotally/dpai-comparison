"""
advanced_analysis.py

New analyses on top of preprocessing.py / master_pipeline.py, implementing the
Step 6 roadmap from the pipeline-review conversation. Every function here is
tied to one of the four core scientific questions the thesis is actually
trying to answer -- noted in each section header -- rather than being an
extra plot for its own sake. Two proposed analyses from that discussion
(relative improvement vs. flat film; dissolved/adsorbed split) are NOT
implemented here because they require new COMSOL exports (an untextured
baseline run, and a separately-exported C_CO2s series) that don't exist in
the current data -- see the conversation notes for why these are blocked on
data, not on code.

USAGE: import this after FAMILIES is populated and build_master_table() has
been run, e.g.:

    from master_pipeline import FAMILIES, build_master_table
    master = build_master_table()
    import advanced_analysis as aa

    combined_series, labels = aa.build_combined_series(FAMILIES)
    rank_df = aa.compute_rank_trajectory(combined_series)
    aa.plot_rank_trajectory(rank_df, top_n=10, labels=labels, palette=R_COLORS)

Every plotting function returns (fig, ax) or ax so you can save/tweak
afterwards, consistent with the existing plots.py conventions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from viz_utils import normalize_label, get_color
from preprocessing import window_average, NEAR_EQ_WINDOW

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "savefig.dpi": 300,
    "figure.dpi": 150,
})

# ---------------------------------------------------------------------------
# Fallback helpers, in case this module is imported without plots.py already
# in the path. plots.py should already define these (used throughout your
# existing Cells 2-9) -- duplicated here defensively so advanced_analysis.py
# works standalone. If plots.py IS importable, its versions are used instead,
# so there's still only one real definition in normal use.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# build_family_colors / METRIC_COLS -- defined locally here rather than
# imported from plots.py. plots.py (as currently structured) is a notebook
# with `# %%` cell markers and real execution code mixed into module scope
# (e.g. `master["family"].unique()` in Cell 2b) -- importing it runs that
# code immediately and fails, since `master` only exists in the notebook's
# own namespace at that point, not inside plots.py as a standalone module.
# If plots.py is later refactored into a clean library (defs only, no
# top-level execution), swap this back to `from plots import ...`.
# ---------------------------------------------------------------------------
METRIC_COLS = {
    "family", "SA", "V", "n_t2000", "n_t15000", "n_near_eq", "n_eq",
    "n_eq_per_V", "pct_eq_t15000", "early_leadership_share", "SA_V", "L_c",
    "volume_type",  # categorical metadata (removed/added) -- not a swept parameter
    "n_eq_t2f", "t10", "t25", "t50", "t75", "t90",  # from merge_time_to_fraction_into_master
}

def build_family_colors(family_names):
    cmap = plt.get_cmap("tab10")
    return {name: cmap(i % 10) for i, name in enumerate(sorted(family_names))}


# ---------------------------------------------------------------------------
# Shared setup: one combined series across every family
# ---------------------------------------------------------------------------
# Several analyses below (rank trajectory, volatility, dominance map) need to
# compare EVERY geometry against every other geometry, regardless of family --
# the same cross-family principle behind the early_leadership_share fix in
# master_pipeline.py. Building this once here means all of these analyses are
# scored on an identical basis, and can't silently drift out of sync with
# each other or with early_leadership_share.

def build_combined_series(families):
    """
    Merges every family's series dict into one, keyed by (family_name, raw_key).
    Also returns a matching {key: display_label} dict for plot legends, so
    labels stay tied to the same keys used for computation.
    """
    combined = {}
    labels = {}
    for family_name, fam in families.items():
        for raw_key, g in fam["series"].items():
            combined_key = (family_name, raw_key)
            combined[combined_key] = g
            labels[combined_key] = f"{family_name}: {normalize_label(g['label'])}"
    return combined, labels


# ===========================================================================
# SECTION 1 -- Rank trajectory + rank volatility
# Answers Q1 (how rankings evolve) far more completely than 2-3 fixed
# snapshots: shows WHEN crossovers happen, whether gradually or suddenly,
# and WHEN (if ever) the ranking stabilizes.
# ===========================================================================

def compute_rank_trajectory(combined_series, t_start=1, t_end=200_000, n_steps=300):
    """
    Ranks every geometry at each of n_steps log-spaced time points.

    Returns a DataFrame: index = time (s), columns = geometry keys,
    values = rank (1 = highest n(t) at that instant).

    This is the single computation behind BOTH the rank-trajectory plot and
    the rank-volatility plot below -- built once and reused, so the two
    figures can't end up telling slightly different stories due to a
    computation mismatch.

    Geometries that haven't started absorbing yet at a given t (i.e. t is
    before their first simulated timestep) are pushed to last place at that
    instant via -inf, rather than being silently dropped from the ranking --
    dropping them would shrink the total rank count over time in a way that's
    easy to misread as "everyone moved up".
    """
    time_grid = np.logspace(np.log10(t_start), np.log10(t_end), n_steps)
    keys = list(combined_series.keys())
    n_keys = len(keys)

    # Pre-sort and pre-extract each geometry's (t, n) arrays once, rather than
    # re-sorting inside the n_steps loop -- this loop runs n_steps x n_keys
    # times, so avoiding repeated sorts matters for anything beyond a toy
    # dataset (your real sweep is ~150 geometries x 300+ time steps).
    sorted_points = {}
    for key in keys:
        pts = sorted(combined_series[key]["points"], key=lambda p: p[0])
        sorted_points[key] = (
            np.array([p[0] for p in pts]),
            np.array([p[1] for p in pts]),
        )

    rank_matrix = np.full((n_steps, n_keys), np.nan)

    for i, t in enumerate(time_grid):
        values = np.empty(n_keys)
        for j, key in enumerate(keys):
            ts, ns = sorted_points[key]
            if len(ts) == 0 or t < ts[0]:
                values[j] = -np.inf
            else:
                values[j] = np.interp(t, ts, ns)
        # argsort descending: highest value gets rank 1
        order = np.argsort(-values, kind="stable")
        ranks = np.empty(n_keys, dtype=int)
        ranks[order] = np.arange(1, n_keys + 1)
        rank_matrix[i, :] = ranks

    return pd.DataFrame(rank_matrix, index=time_grid, columns=pd.Index(keys, tupleize_cols=False))


def plot_rank_trajectory(rank_df, top_n=10, labels=None, palette=None, ax=None):
    """
    Continuous rank-vs-time plot for every geometry that was EVER ranked
    <= top_n at ANY point in the time range -- not just top_n at one instant.
    A single-timestamp selection would miss exactly the geometries this plot
    exists to reveal (e.g. #2 early, #20 late, or vice versa).
    """
    ever_top = rank_df.columns[(rank_df <= top_n).any(axis=0)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    for key in ever_top:
        label = labels[key] if labels else str(key)
        color = get_color(label, palette) if palette is not None else None
        ax.plot(rank_df.index, rank_df[key], label=label, color=color,
                 linewidth=1.3, alpha=0.85)

    ax.set_xscale("log")
    ax.invert_yaxis()  # rank 1 at the top of the plot
    ax.set_xlabel("t (s, log scale)")
    ax.set_ylabel("Rank (1 = highest uptake)")
    ax.set_title(f"Rank trajectory -- geometries ever in top {top_n}")
    ax.legend(fontsize=7, ncol=2, frameon=False, loc="upper left",
               bbox_to_anchor=(1.01, 1))
    ax.grid(True, linewidth=0.3, alpha=0.4)
    return ax


def compute_rank_volatility(rank_df):
    """
    At each timestep (after the first), counts how many geometries' rank
    changed relative to the previous sampled timestep. A direct, quantitative
    answer to "when do rankings stop changing" -- rather than eyeballing when
    lines in the rank-trajectory plot stop crossing.
    """
    diffs = rank_df.diff().iloc[1:]
    volatility = (diffs != 0).sum(axis=1)
    volatility.name = "n_rank_changes"
    return volatility


def find_stabilization_time(volatility, zero_run_length=5):
    """
    Earliest time at which volatility stays at zero for `zero_run_length`
    consecutive samples in a row. Requiring a RUN of zeros (not just one
    zero reading) avoids misreading a brief lull between two separate
    crossover events as "rankings have stabilized".

    Returns None if volatility never reaches a zero run within the
    simulated/sampled range -- itself a meaningful result (rankings never
    fully settle), worth reporting rather than raising.
    """
    is_zero = (volatility.values == 0)
    times = volatility.index.values
    for i in range(len(is_zero) - zero_run_length + 1):
        if is_zero[i:i + zero_run_length].all():
            return times[i]
    return None


def plot_rank_volatility(volatility, stabilization_time=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(volatility.index, volatility.values, color="black", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("t (s, log scale)")
    ax.set_ylabel("# geometries that changed\nrank since previous sample")
    ax.set_title("Ranking volatility over time")
    if stabilization_time is not None:
        ax.axvline(stabilization_time, color="crimson", linestyle="--", linewidth=1,
                    label=f"stabilizes at t \u2248 {stabilization_time:,.0f} s")
        ax.legend(fontsize=8, frameon=False)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    return ax


# ===========================================================================
# SECTION 2 -- Time-to-X% uptake
# Answers "who reaches a USEFUL operating level first", not just "who's
# ahead at one arbitrary instant". Also gives per-geometry t1/2 directly,
# rather than relying on a single geometry's t1/2 as a global estimate.
# ===========================================================================

def compute_time_to_fraction(points, fractions=(0.10, 0.25, 0.50, 0.75, 0.90), n_eq=None):
    """
    For one geometry's (t, n) points, finds the time at which n(t) FIRST
    reaches each target fraction of n_eq, via linear interpolation between
    the bracketing sampled points (smoother than snapping to the nearest
    sampled timestep).

    n_eq must be passed explicitly -- this should be the windowed
    near-equilibrium average from preprocessing.window_average(), consistent
    with the rest of the pipeline's equilibrium definition, NOT the raw final
    data point (which we already know can sit slightly off the true
    plateau due to solver noise / incomplete convergence -- see the pillar
    vs. recess tail-behavior comparison from earlier).
    """
    if not points or n_eq is None or n_eq <= 0:
        return {f: None for f in fractions}

    pts = sorted(points, key=lambda p: p[0])
    ts = np.array([p[0] for p in pts])
    ns = np.array([p[1] for p in pts])

    result = {}
    for frac in fractions:
        target = frac * n_eq
        idx = np.searchsorted(ns, target)
        if idx == 0:
            result[frac] = float(ts[0])
        elif idx >= len(ts):
            result[frac] = None  # never reached within the simulated range
        else:
            t0, t1 = ts[idx - 1], ts[idx]
            n0, n1 = ns[idx - 1], ns[idx]
            result[frac] = float(t0) if n1 == n0 else float(t0 + (target - n0) * (t1 - t0) / (n1 - n0))
    return result


def build_time_to_fraction_table(families, fractions=(0.10, 0.25, 0.50, 0.75, 0.90)):
    """
    Runs compute_time_to_fraction() for every geometry in every family, using
    each geometry's OWN windowed near-eq average as n_eq (not a single shared
    value) -- equilibrium capacity genuinely differs by geometry, so a shared
    n_eq would silently mix up "reached its own 50%" with "reached 50% of
    some other geometry's capacity".

    Emits the family's own parameter columns (e.g. R, H -- or width, depth
    for grooves) via each family's param_names, rather than an opaque
    (raw_key) tuple column, specifically so this table can be merged
    directly into the master table via merge_time_to_fraction_into_master()
    below -- that merge is what lets you run sensitivity/robustness analysis
    against a genuinely KINETIC metric like t50, instead of only against
    capacity metrics like n_near_eq.
    """
    rows = []
    for family_name, fam in families.items():
        param_names = fam["param_names"]
        for raw_key, g in fam["series"].items():
            pts = sorted(g["points"], key=lambda p: p[0])
            n_eq = window_average(pts, *NEAR_EQ_WINDOW)
            times = compute_time_to_fraction(pts, fractions, n_eq=n_eq)
            row = {"family": family_name, "label": normalize_label(g["label"])}
            if isinstance(raw_key, tuple):
                for name, val in zip(param_names, raw_key):
                    row[name] = val
            else:
                row[param_names[0]] = raw_key
            row["n_eq_t2f"] = n_eq  # suffixed to avoid colliding with master's own n_eq column on merge
            for frac, t in times.items():
                row[f"t{int(frac * 100)}"] = t
            rows.append(row)
    return pd.DataFrame(rows)


def merge_time_to_fraction_into_master(master, t2f_df, param_cols=("R", "H")):
    """
    Left-merges time-to-fraction milestones (t10...t90) into the master
    table on (family, *param_cols), so sensitivity/robustness/etc can be run
    against a genuinely kinetic metric instead of only capacity metrics.

    Example:
        t2f = build_time_to_fraction_table(FAMILIES)
        master_k = merge_time_to_fraction_into_master(master, t2f)
        kinetic_sens = build_sensitivity_table(master_k, metric="t50")

    Families whose parameters aren't named in param_cols (e.g. grooves,
    keyed by width/depth rather than R/H) simply won't find a match and get
    NaN in the merged t-columns for those rows -- pass the right param_cols
    per call, or merge family-by-family, if you need this for a
    non-(R,H)-keyed family.
    """
    merge_cols = ["family"] + [c for c in param_cols if c in master.columns and c in t2f_df.columns]
    if len(merge_cols) < 2:
        raise ValueError(...)
    t_cols = [c for c in t2f_df.columns if c.startswith("t") and c[1:].isdigit()]

    master_r = master.copy()
    t2f_r = t2f_df[merge_cols + t_cols].copy()
    for c in merge_cols:
        if c != "family":
            master_r[c] = master_r[c].round(ndigits)
            t2f_r[c] = t2f_r[c].round(ndigits)

    return master_r.merge(t2f_r, on=merge_cols, how="left")


def plot_time_to_fraction_race(t2f_df, fractions=(0.10, 0.25, 0.50, 0.75, 0.90),
                                 top_n_per_fraction=5, family_colors=None, ax=None):
    """
    'Race' plot: x-axis = milestone (% of own equilibrium), y-axis = time
    (log) taken to reach it. At each milestone, only the top_n_per_fraction
    FASTEST geometries are plotted -- with ~150 geometries total, plotting
    everyone at every milestone is unreadable, and the interesting result is
    specifically WHO is fastest and whether that changes as the milestone
    gets harder (e.g. cones fastest to low milestones, pillars fastest to
    high ones) -- exactly what this plot is designed to surface.
    """
    if family_colors is None:
        family_colors = build_family_colors(t2f_df["family"].unique())
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    frac_cols = [f"t{int(f * 100)}" for f in fractions]
    x_positions = list(range(len(fractions)))
    plotted_families = set()

    for i, col in enumerate(frac_cols):
        sub = t2f_df.dropna(subset=[col]).nsmallest(top_n_per_fraction, col)
        for _, row in sub.iterrows():
            fam = row["family"]
            legend_label = fam if fam not in plotted_families else None
            plotted_families.add(fam)
            ax.scatter(x_positions[i], row[col], color=family_colors[fam],
                        s=40, edgecolor="white", linewidth=0.5,
                        label=legend_label, zorder=3)

    ax.set_yscale("log")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{int(f * 100)}%" for f in fractions])
    ax.set_xlabel("Fraction of that geometry's own equilibrium")
    ax.set_ylabel("Time (s, log scale)")
    ax.set_title(f"Fastest {top_n_per_fraction} geometries to reach each uptake milestone")
    ax.legend(fontsize=8, frameon=False, title="Family")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    return ax


# ===========================================================================
# SECTION 3 -- Sensitivity map
# Answers Q3: does depth actually matter more than radius/pitch, or is that
# an assumption from t~L^2 intuition that's never been checked against the
# actual sweep grid? Directly tests the "depth is the dominant kinetic
# lever" claim currently stated in the project notes as a design principle.
# ===========================================================================

def compute_sensitivity(master, family, metric="n_near_eq"):
    """
    Estimates how sensitive `metric` is to each of a family's two swept
    parameters, using centered finite differences (np.gradient) across the
    existing 5x5 grid.

    Reports both:
      - mean_abs_slope: mean |d(metric)/d(param)| across the grid
      - range_normalized: mean_abs_slope x (max(param) - min(param))

    The range-normalized version is the one to actually compare between the
    two parameters of a family (or between families) -- a parameter that's
    swept over a narrow range can look "insensitive" on raw slope alone even
    if it matters a lot, and vice versa. Both are returned so you can see
    whether a parameter matters because it's a steep slope, a wide sweep
    range, or both.
    """
    sub = master[master["family"] == family]
    param_cols = [c for c in sub.columns if c not in METRIC_COLS]
    active_cols = [c for c in param_cols if sub[c].notna().any()]
    if len(active_cols) != 2:
        raise ValueError(
            f"Expected exactly 2 active parameter columns for family "
            f"'{family}', found {active_cols}. Sensitivity analysis needs a "
            f"full 2D grid -- check param_names for this family."
        )
    p1, p2 = active_cols

    pivot = sub.pivot(index=p2, columns=p1, values=metric)
    if pivot.isna().any().any():
        print(f"[warning] {family}: sensitivity grid for '{metric}' has "
              f"{pivot.isna().sum().sum()} missing cell(s) -- gradient will "
              f"propagate NaN through adjacent cells.")

    p1_vals = pivot.columns.values.astype(float)
    p2_vals = pivot.index.values.astype(float)

    # d(metric)/d(p1): gradient along columns, computed per row (fixed p2)
    d_dp1 = np.vstack([np.gradient(pivot.values[row_i, :], p1_vals)
                        for row_i in range(pivot.shape[0])])
    # d(metric)/d(p2): gradient along rows, computed per column (fixed p1)
    d_dp2 = np.column_stack([np.gradient(pivot.values[:, col_i], p2_vals)
                              for col_i in range(pivot.shape[1])])

    mean_abs_slope_p1 = float(np.nanmean(np.abs(d_dp1)))
    mean_abs_slope_p2 = float(np.nanmean(np.abs(d_dp2)))
    range_p1 = float(p1_vals.max() - p1_vals.min())
    range_p2 = float(p2_vals.max() - p2_vals.min())
    range_norm_p1 = mean_abs_slope_p1 * range_p1
    range_norm_p2 = mean_abs_slope_p2 * range_p2

    return {
        "family": family,
        "param1": p1, "mean_abs_slope_param1": mean_abs_slope_p1,
        "range_normalized_param1": range_norm_p1,
        "param2": p2, "mean_abs_slope_param2": mean_abs_slope_p2,
        "range_normalized_param2": range_norm_p2,
        "dominant_param": p1 if range_norm_p1 > range_norm_p2 else p2,
    }


def build_sensitivity_table(master, metric="n_near_eq"):
    """Runs compute_sensitivity() across every family; skips (with a warning,
    not a hard failure) any family that doesn't have a clean 2D grid."""
    rows = []
    for family in sorted(master["family"].unique()):
        try:
            rows.append(compute_sensitivity(master, family, metric=metric))
        except ValueError as e:
            print(f"[warning] skipping sensitivity for {family}: {e}")
    return pd.DataFrame(rows)


def plot_sensitivity_bars(sens_df, ax=None):
    """
    Grouped bar chart: for each family, range-normalized sensitivity to its
    two swept parameters, side by side. Bars are annotated with the actual
    parameter name (R/H for most families, width/depth for grooves) since
    families don't necessarily share parameter identities.

    If H's (or the equivalent depth-like parameter's) bar is consistently
    taller than the other across families, that's the "depth is the
    dominant lever" claim now backed by a computed slope rather than
    asserted from t~L^2 intuition alone.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(sens_df))
    width = 0.35
    ax.bar(x - width / 2, sens_df["range_normalized_param1"], width, color="#4C72B0")
    ax.bar(x + width / 2, sens_df["range_normalized_param2"], width, color="#DD8452")

    for i, row in sens_df.reset_index(drop=True).iterrows():
        ax.text(i - width / 2, row["range_normalized_param1"], row["param1"],
                 ha="center", va="bottom", fontsize=7, rotation=90)
        ax.text(i + width / 2, row["range_normalized_param2"], row["param2"],
                 ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(sens_df["family"], rotation=20, ha="right")
    ax.set_ylabel("Range-normalized sensitivity\n(mean |slope| \u00d7 swept range)")
    ax.set_title("Parameter sensitivity by family")
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.4)
    return ax


# ===========================================================================
# SECTION 4 -- Robustness of the optimum
# Answers Q4: is the top-ranked geometry meaningfully better, or is it
# sitting in a plateau of near-identical designs? Currently completely
# absent -- every existing plot treats rank #1 as categorically best,
# whatever the actual margin to #2/#3 happens to be.
# ===========================================================================

def compute_optimum_robustness(master, metric="n_near_eq", top_n=5):
    """
    Takes the top_n geometries by `metric` ACROSS THE WHOLE MASTER TABLE
    (all families together), and reports each one's gap to the best,
    in both absolute and percentage terms.
    """
    ranked = (master.dropna(subset=[metric])
              .sort_values(metric, ascending=False)
              .head(top_n).copy().reset_index(drop=True))
    if ranked.empty:
        raise ValueError(f"No rows with non-null '{metric}' to rank.")
    best_val = ranked.loc[0, metric]
    ranked["gap_to_best_abs"] = best_val - ranked[metric]
    ranked["gap_to_best_pct"] = 100 * ranked["gap_to_best_abs"] / best_val
    return ranked


def plot_optimum_robustness(robustness_df, metric="n_near_eq", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    labels = [f"{row['family']}\n(#{i + 1})" for i, (_, row) in enumerate(robustness_df.iterrows())]
    ax.bar(labels, robustness_df[metric], color="#4C72B0")
    for i, (_, row) in enumerate(robustness_df.iterrows()):
        text = "best" if i == 0 else f"-{row['gap_to_best_pct']:.2f}%"
        ax.text(i, row[metric], text, ha="center", va="bottom", fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title("Robustness of the top performer(s)")
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.4)
    return ax


# ===========================================================================
# SECTION 5 -- Dominance map
# Distinguishes "consistently good" from "briefly spectacular" -- a
# different question from early_leadership_share, which collapses this
# distinction into one number.
# ===========================================================================

def compute_dominance_scores(rank_df, thresholds=(1, 3, 5, 10)):
    """
    For every geometry, the share of sampled (log-spaced) time points where
    its rank was <= each threshold. Reuses the rank_df already computed for
    the rank-trajectory plot -- same underlying data, different summary.
    """
    n_samples = len(rank_df)
    scores = {}
    for key in rank_df.columns:
        ranks = rank_df[key].values
        scores[key] = {f"top{t}_share": float((ranks <= t).sum()) / n_samples for t in thresholds}
    return pd.DataFrame.from_dict(scores, orient="index")


def plot_dominance_map(dominance_df, labels=None, thresholds=(1, 3, 5, 10), top_n=15, ax=None):
    """
    Nested horizontal bars: for the top_n geometries (ranked by their widest
    threshold's share, e.g. Top10), draws progressively narrower/darker bars
    for each tighter threshold layered on top. A geometry whose bars are all
    nearly the same length was dominant every time it appeared in the top 10;
    one with a wide Top10 bar but a thin Top1 sliver was occasionally great
    but not consistent -- a distinction early_leadership_share alone can't show.
    """
    sort_col = f"top{max(thresholds)}_share"
    ranked = dominance_df.sort_values(sort_col, ascending=False).head(top_n)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(ranked))))

    y_pos = np.arange(len(ranked))
    cmap = plt.get_cmap("Blues")
    sorted_thresholds = sorted(thresholds, reverse=True)  # widest first (drawn behind)
    for i, t in enumerate(sorted_thresholds):
        shade = 0.3 + 0.6 * (i / max(1, len(sorted_thresholds) - 1))
        ax.barh(y_pos, ranked[f"top{t}_share"], color=cmap(shade), label=f"Top {t}", height=0.7)

    label_list = [labels[k] if labels else str(k) for k in ranked.index]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(label_list, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Share of simulated time spent at this rank threshold")
    ax.set_title("Dominance map -- consistency vs. brief leadership")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(True, axis="x", linewidth=0.3, alpha=0.4)
    return ax


# ===========================================================================
# SECTION 6 -- Diffusion penetration overlay + growth-rate (dn/dt)
# Mechanistic (Q2): WHY rankings change, not just that they do.
# ===========================================================================

# From the Langmuir retardation-factor estimate (R ~ 1,106-2,132 applied to
# D_CO2p = 6.68e-11 m^2/s). Override per-call if a more precise, geometry- or
# coverage-specific D_eff is derived later -- this is a single representative
# value, not re-derived per geometry.
D_EFF_DEFAULT = 3e-14  # m^2/s


def penetration_depth(t, D_eff=D_EFF_DEFAULT):
    """sqrt(D_eff * t): characteristic diffusion penetration depth at time t."""
    return np.sqrt(D_eff * t)


def plot_penetration_vs_depth(series, family_name=None, H_values=None, D_eff=D_EFF_DEFAULT, ax=None):
    """
    Plots the penetration-depth curve sqrt(D_eff*t) against a log-time axis,
    with horizontal reference lines at each distinct H present in the
    family, and a marker at the PREDICTED crossing time (t = H^2 / D_eff) for
    each. This turns "cone curves overlap early because depth doesn't matter
    yet" from a qualitative observation into a specific, checkable
    prediction per depth -- you can directly compare the marked crossing
    time against where curves actually start to diverge in the n(t) overlay
    plot for the same family.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    t_range = np.logspace(-1, 5.3, 300)  # ~0.1 s to ~200,000 s
    depth_curve = penetration_depth(t_range, D_eff)
    ax.plot(t_range, depth_curve * 1e6, color="black", linewidth=1.8,
             label=r"$\sqrt{D_{eff}\,t}$")

    if H_values is None:
        H_values = sorted({g["H"] for g in series.values()})

    cmap = plt.get_cmap("plasma")
    for i, H in enumerate(H_values):
        color = cmap(i / max(1, len(H_values) - 1))
        ax.axhline(H * 1e6, color=color, linestyle="--", linewidth=1, alpha=0.8,
                    label=f"H = {H * 1e6:.1f} \u00b5m")
        t_cross = (H ** 2) / D_eff  # solve sqrt(D_eff*t) = H for t
        if t_range[0] <= t_cross <= t_range[-1]:
            ax.plot(t_cross, H * 1e6, marker="o", color=color, markersize=5, zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("t (s, log scale)")
    ax.set_ylabel("Depth (\u00b5m)")
    title = "Diffusion penetration vs. feature depth"
    if family_name:
        title += f" -- {family_name}"
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    return ax


def compute_growth_rate(points):
    """
    dn/dt via np.gradient on the actual (nonuniformly spaced) sampled time
    points -- your COMSOL exports are densely sampled early (0.1 s steps)
    and sparsely sampled late (10,000 s steps); np.gradient handles this
    correctly as long as the real t array is passed in, not assumed uniform.
    Drops exact-duplicate t values first (defensive; parse_comsol_txt already
    deduplicates on read, but this function may be fed data from elsewhere).
    """
    pts = sorted(points, key=lambda p: p[0])
    ts = np.array([p[0] for p in pts])
    ns = np.array([p[1] for p in pts])
    _, unique_idx = np.unique(ts, return_index=True)
    ts, ns = ts[unique_idx], ns[unique_idx]
    if len(ts) < 2:
        return ts, np.array([])
    return ts, np.gradient(ns, ts)


def plot_growth_rate_overlay(series, family_name=None, palette=None, ax=None,
                              t_min=5.0, log_y=True):
    """
    dn/dt vs t for every geometry in a family, log-x. Answers a different
    question than the cumulative n(t) overlay: not "who has captured the
    most so far" but "who is absorbing fastest RIGHT NOW".

    t_min: excludes points before this time (default 5.0 s). The COMSOL
    exports show a sharp startup transient around t~1s (a near-instant jump
    from n=0 to a small nonzero value, consistent across every geometry --
    almost certainly a boundary-condition "turn on" artifact, not a real
    kinetic feature). Left in, this transient dominates the y-axis by 1-2
    orders of magnitude and makes every geometry's actual rate curve --
    which is what this plot exists to compare -- look like a flat line at
    zero for the rest of the simulated range. t_min=5.0 clips past that
    transient while still starting well before t1/2 (~10,000s).

    log_y: plots dn/dt on a log y-axis (default True). Adsorption rate
    typically spans several orders of magnitude between early and late
    time even after excluding the t~1s transient, so a linear y-axis will
    still flatten the late-time detail unless you're specifically comparing
    early-time rates only. Set False if you want a linear view of a
    narrower time slice.
    """
    if palette is None:
        raise ValueError("Pass palette=R_COLORS (or your color list) explicitly.")
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    for key, g in series.items():
        ts, rates = compute_growth_rate(g["points"])
        if len(ts) == 0:
            continue
        label = normalize_label(g["label"])
        mask = ts >= t_min
        # log_y requires strictly positive values -- dn/dt should be
        # monotonically non-negative for a pure uptake curve, but clip any
        # near-zero/negative noise (e.g. from the solver-noise dip seen near
        # the plateau) rather than letting it silently break the log axis
        plot_rates = rates[mask]
        plot_ts = ts[mask]
        if log_y:
            valid = plot_rates > 0
            plot_ts, plot_rates = plot_ts[valid], plot_rates[valid]
        ax.plot(plot_ts, plot_rates, label=label, color=get_color(label, palette), linewidth=1.3)

    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("t (s, log scale)")
    ax.set_ylabel("dn/dt (mol/s)" + (", log scale" if log_y else ""))
    title = "Adsorption rate over time"
    if family_name:
        title += f" -- {family_name}"
    if t_min:
        title += f" (t \u2265 {t_min:g}s)"
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    return ax


# ===========================================================================
# SECTION 7 -- Performance landscape evolution + volume-normalized comparison
# Q2/Q3: watching the optimum MOVE across (R,H) space over time, and testing
# the volume-depletion hypothesis numerically rather than only visually.
# ===========================================================================

def plot_landscape_evolution(master, family, time_cols_labels):
    """
    Side-by-side R x H heatmap panels at several different times/metrics, so
    the optimum's movement across design space is visible directly, rather
    than only comparable by memory across separate figures. Scoped down from
    an animated version to 2-3 static panels, appropriate for a printed
    thesis figure.

    time_cols_labels: ordered dict/list of (column_name, panel_title) pairs,
    using columns that already exist in the master table, e.g.
        [("n_t2000", "t=2,000 s"), ("n_t15000", "t=15,000 s"),
         ("n_near_eq", "near-equilibrium")]

    Uses a SHARED color scale across all panels (not each panel's own) --
    otherwise every panel's shape looks superficially similar even if the
    underlying magnitudes differ hugely between early and late time, which
    would visually understate how much the landscape actually changed.
    """
    sub = master[master["family"] == family]
    param_cols = [c for c in sub.columns if c not in METRIC_COLS]
    active_cols = [c for c in param_cols if sub[c].notna().any()]
    if len(active_cols) != 2:
        raise ValueError(f"Expected 2 active parameter columns for {family}, got {active_cols}")
    p1, p2 = active_cols

    n_panels = len(time_cols_labels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    cols = [c for c, _ in time_cols_labels]
    all_vals = np.concatenate([sub[c].dropna().values for c in cols])
    vmin, vmax = float(all_vals.min()), float(all_vals.max())

    im = None
    for ax, (col, title) in zip(axes, time_cols_labels):
        pivot = sub.pivot(index=p2, columns=p1, values=col)
        im = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap="viridis",
                         vmin=vmin, vmax=vmax,
                         extent=[pivot.columns.min(), pivot.columns.max(),
                                  pivot.index.min(), pivot.index.max()])
        if not np.all(np.isnan(pivot.values)):
            max_idx = np.unravel_index(np.nanargmax(pivot.values), pivot.values.shape)
            ax.plot(pivot.columns[max_idx[1]], pivot.index[max_idx[0]],
                     marker="*", color="red", markersize=15,
                     markeredgecolor="white", markeredgewidth=0.8, zorder=5)
        ax.set_xlabel(p1)
        ax.set_ylabel(p2)
        ax.set_title(title)

    fig.colorbar(im, ax=axes, label="n (mol)", shrink=0.8)
    fig.suptitle(f"{family}: performance landscape evolution (red marker = optimum)")
    return fig, axes


def plot_volume_normalized_comparison(master, family_colors=None, ax=None):
    """
    n_eq_per_V vs. n_eq, across all families -- split by volume_type.

    IMPORTANT: n_eq_per_V means OPPOSITE things depending on volume_type:
      - "removed" (recesses, grooves): V_unit is polymer volume taken away.
        LOW n_eq + HIGH n_eq_per_V is consistent with the volume-depletion
        story (saturates a small remaining volume quickly).
      - "added" (pillars): V_unit is extra polymer volume added. HIGH
        n_eq_per_V here means the added material itself is being used
        efficiently -- a genuinely different question, not a depletion
        artifact, since these families have MORE material, not less.

    Plotting both types with the same marker/interpretation (as the
    earlier version of this function did) makes them look directly
    comparable when they aren't -- this version uses a different marker
    shape per volume_type (circle = removed, triangle = added) and
    prints a warning if volume_type is missing (i.e. master was built
    before this field existed, defaulting everything to "removed", which
    would silently mislabel any additive/pillar family).
    """
    if "volume_type" not in master.columns:
        raise ValueError(
            "master table has no 'volume_type' column -- rebuild it with "
            "the updated master_pipeline.py (build_master_table now carries "
            "volume_type through from each family's build_family_entry() call)."
        )

    if family_colors is None:
        family_colors = build_family_colors(master["family"].unique())
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    marker_by_type = {"removed": "o", "added": "^"}
    seen_types = set(master["volume_type"].unique())
    unknown_types = seen_types - set(marker_by_type)
    if unknown_types:
        print(f"[warning] unrecognized volume_type value(s) {unknown_types} -- "
              f"expected 'removed' or 'added'. Plotting with a default marker; "
              f"check build_family_entry() calls for these families.")

    for (family, vtype), sub in master.groupby(["family", "volume_type"]):
        marker = marker_by_type.get(vtype, "s")
        ax.scatter(sub["n_eq"], sub["n_eq_per_V"],
                    label=f"{family} ({vtype})",
                    color=family_colors[family], marker=marker,
                    s=45, edgecolor="white", linewidth=0.5, alpha=0.9)

    ax.set_xlabel("Near-equilibrium uptake, n_eq (mol)")
    ax.set_ylabel("Capacity density, n_eq / V_unit (mol/m\u00b3)")
    ax.set_title("Volume-normalized capacity vs. absolute capacity\n"
                  "(\u25cf = subtractive/removed volume, \u25b2 = additive/added volume)")
    ax.legend(fontsize=7, ncol=2, frameon=False, title="Family (volume type)")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    return ax
