"""
plots.py

Visualizations for the cross-family master table (from master_pipeline.py).
Written with "# %%" cell markers so it can be pasted into a Jupyter
notebook (or opened directly as a notebook in VS Code / JupyterLab, which
both recognize "# %%" as a cell boundary) and run cell by cell.

Assumes:
  - `master` = the dataframe from build_master_table() (master_pipeline.py)
  - `families` = the same FAMILIES dict passed into build_master_table(),
    needed for the family-level timeline since that one needs full curves,
    not just the reduced per-geometry summary rows.
"""

# %% [Cell 1] Imports and shared setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from master_pipeline import leadership_segments  # reused for family-level timeline too

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


# %% [Cell 2] Plot 1 -- early-leadership vs. universal descriptor, all families
def plot_leadership_vs_descriptor(master, descriptor="SA_V", family_colors=None, ax=None):
    """
    The core cross-family test: does a geometry-agnostic descriptor
    (surface-area-to-volume ratio, or L_c) predict early-time performance
    regardless of which family produced the shape? A clean trend here is
    the evidence for a general "maximize X" recommendation; a split by
    family is evidence that shape matters beyond the simple descriptor.
    """
    if family_colors is None:
        family_colors = build_family_colors(master["family"].unique())
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    for family, sub in master.groupby("family"):
        ax.scatter(
            sub[descriptor], sub["early_leadership_share"],
            label=family, color=family_colors[family],
            s=45, edgecolor="white", linewidth=0.5, alpha=0.9,
        )

    ax.set_xlabel(descriptor)
    ax.set_ylabel("Early-time leadership share (0-1)")
    ax.set_title(f"Early leadership vs. {descriptor}, all families")
    ax.legend(fontsize=8, frameon=False, title="Family", title_fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    return ax


# %% [Cell 2b] Run it -- try both SA_V and L_c side by side, since they're
# mathematically related (L_c = 1/SA_V) but may read differently on a plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
family_colors = build_family_colors(master["family"].unique())
plot_leadership_vs_descriptor(master, descriptor="SA_V", family_colors=family_colors, ax=axes[0])
plot_leadership_vs_descriptor(master, descriptor="L_c", family_colors=family_colors, ax=axes[1])
axes[1].legend().remove()  # avoid duplicate legend, keep only the left one
fig.tight_layout()
fig.savefig("early_leadership_vs_descriptors.pdf", bbox_inches="tight")
plt.show()


# %% [Cell 3] Plot 2 -- ranked bar chart of early-leadership share, all geometries
def plot_leadership_ranked_bar(master, top_n=20, family_colors=None):
    """
    Replaces the old "ranked at t=15,000s" bar chart with the robust
    leadership-duration score. Capped at top_n bars since a full 6-family
    sweep can easily produce 50+ rows -- not readable as one bar chart.
    """
    if family_colors is None:
        family_colors = build_family_colors(master["family"].unique())

    param_cols = [c for c in master.columns if c not in METRIC_COLS]
    ranked = master.sort_values("early_leadership_share", ascending=False).head(top_n)
    labels = [geometry_label(row, param_cols) for _, row in ranked.iterrows()]
    colors = [family_colors[f] for f in ranked["family"]]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(ranked))))
    ax.barh(labels[::-1], ranked["early_leadership_share"][::-1], color=colors[::-1])
    ax.set_xlabel("Early-time leadership share (0-1)")
    ax.set_title(f"Top {top_n} geometries by early-time leadership")
    ax.grid(True, axis="x", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    return fig, ax


# %% [Cell 3b] Run it
fig, ax = plot_leadership_ranked_bar(master, top_n=20, family_colors=family_colors)
fig.savefig("early_leadership_ranked_bar.pdf", bbox_inches="tight")
plt.show()


# %% [Cell 4] Plot 3 -- early-time vs. equilibrium trade-off
def plot_early_vs_equilibrium_tradeoff(master, family_colors=None):
    """
    Makes the "deep wins early, shallow wins at equilibrium" trade-off
    from the last report quantitative and testable across every family,
    not just cones. Bottom-right of this plot = fast early, poor
    equilibrium performance; top-left = the reverse.
    """
    if family_colors is None:
        family_colors = build_family_colors(master["family"].unique())

    fig, ax = plt.subplots(figsize=(7, 5))
    for family, sub in master.groupby("family"):
        ax.scatter(
            sub["early_leadership_share"], sub["pct_eq_t15000"],
            label=family, color=family_colors[family],
            s=45, edgecolor="white", linewidth=0.5, alpha=0.9,
        )
    ax.set_xlabel("Early-time leadership share (0-1)")
    ax.set_ylabel("% of equilibrium reached by t=15,000s")
    ax.set_title("Early-time performance vs. equilibrium progress")
    ax.legend(fontsize=8, frameon=False, title="Family", title_fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    return fig, ax


# %% [Cell 4b] Run it
fig, ax = plot_early_vs_equilibrium_tradeoff(master, family_colors=family_colors)
fig.savefig("early_vs_equilibrium_tradeoff.pdf", bbox_inches="tight")
plt.show()


# %% [Cell 5] Plot 4 -- family-level leadership timeline
def compute_family_envelope(series, t_start=1, t_end=200_000, n_steps=300):
    """
    'Best-in-family' curve: at each time point, the max uptake across
    every geometry in this family. This is what lets the family-level
    timeline answer "which family wins," using each family's strongest
    member rather than averaging away its best performers.
    """
    time_grid = np.logspace(np.log10(t_start), np.log10(t_end), n_steps)
    envelope = []
    for t in time_grid:
        vals = []
        for g in series.values():
            pts = sorted(g["points"], key=lambda p: p[0])
            if pts and t >= pts[0][0]:
                vals.append(np.interp(t, [p[0] for p in pts], [p[1] for p in pts]))
        if vals:
            envelope.append((t, max(vals)))
    return envelope


def plot_family_leadership_timeline(families, t_start=1, t_end=200_000, family_colors=None):
    """
    Family-level version of leadership_timeline_report: instead of tracking
    individual (R,H) combos, tracks which *family's* best member is
    winning at each moment. Reuses leadership_segments() from
    master_pipeline.py by feeding it a pseudo-series of family envelopes.
    """
    if family_colors is None:
        family_colors = build_family_colors(families.keys())

    # build one envelope curve per family, then treat those envelopes as
    # if they were individual "geometries" for the existing leadership
    # segment-finder to compare against each other
    pseudo_series = {
        name: {"points": compute_family_envelope(fam["series"], t_start, t_end)}
        for name, fam in families.items()
    }
    segments = leadership_segments(pseudo_series, t_start=t_start, t_end=t_end)

    fig, ax = plt.subplots(figsize=(10, 2.5))
    for start, end, family in segments:
        ax.barh(0, end - start, left=start, height=0.6,
                color=family_colors[family], edgecolor="white", linewidth=0.5)
        mid = np.sqrt(start * end) if start > 0 else end / 2
        ax.text(mid, 0.42, family, ha="center", va="bottom", fontsize=9)

    ax.set_xscale("log")
    ax.set_xlim(t_start, t_end)
    ax.set_ylim(-0.5, 0.85)
    ax.set_yticks([])
    ax.set_xlabel("t (s, log scale)")
    ax.set_title("Leading microtexture family over time")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig, ax


# %% [Cell 6] Plot 6 -- R x H (or family-native param) heatmap, one per family
def plot_family_heatmap(master, family, metric="early_leadership_share", ax=None):
    """
    Heatmap of `metric` across a family's own 2D parameter grid (e.g. R x H
    for cones, width x depth for grooves -- auto-detected per family, since
    different families can have different active parameter columns).

    Useful for the open question of whether SA_V/L_c actually explains
    performance everywhere in parameter space, or only on average --  a
    heatmap shows *where* a descriptor-based prediction breaks down,
    which a scatter plot averages away.
    """
    sub = master[master["family"] == family]
    param_cols = [c for c in master.columns if c not in METRIC_COLS]
    active_cols = [c for c in param_cols if sub[c].notna().any()]

    if len(active_cols) != 2:
        raise ValueError(
            f"Expected exactly 2 active parameter columns for family "
            f"'{family}', found {active_cols}. Heatmap needs a 2D grid --"
            f" check param_names for this family in FAMILIES."
        )
    p1, p2 = active_cols  # p1 -> x-axis (columns), p2 -> y-axis (rows)

    pivot = sub.pivot(index=p2, columns=p1, values=metric)
    if pivot.isna().any().any():
        print(f"[warning] {family}: heatmap has {pivot.isna().sum().sum()} "
              f"missing cell(s) for metric='{metric}' -- likely a sparse "
              f"or incomplete sweep grid, not a plotting bug.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(
        pivot.values, origin="lower", aspect="auto", cmap="viridis",
        extent=[pivot.columns.min(), pivot.columns.max(),
                pivot.index.min(), pivot.index.max()],
    )
    plt.colorbar(im, ax=ax, label=metric)
    ax.set_xlabel(p1)
    ax.set_ylabel(p2)
    ax.set_title(f"{family}: {metric}")
    return ax


def plot_all_family_heatmaps(master, metric="early_leadership_share"):
    """Grid of one heatmap per family, side by side for easy comparison."""
    families_list = sorted(master["family"].unique())
    fig, axes = plt.subplots(1, len(families_list),
                              figsize=(5 * len(families_list), 4.5))
    if len(families_list) == 1:
        axes = [axes]
    for ax, family in zip(axes, families_list):
        plot_family_heatmap(master, family, metric=metric, ax=ax)
    fig.tight_layout()
    return fig, axes


# %% [Cell 6b] Run it
fig, axes = plot_all_family_heatmaps(master, metric="early_leadership_share")
fig.savefig("family_heatmaps_leadership_share.pdf", bbox_inches="tight")
plt.show()


# %% [Cell 7] Plot 7 -- n(t) log-x overlay, all geometries in one family
def plot_uptake_overlay(series, family_name=None, palette=None, ax=None):
    """
    Classic "every geometry's uptake curve on one log-time axis" plot.
    This is the figure that most directly shows the H-driven early/
    equilibrium ranking inversion -- curves crossing each other is the
    visual signature of that effect.

    palette: required, passed through to viz_utils.get_color so colors
             stay consistent with single_family_timeline.py figures for
             the same family.
    """
    from viz_utils import get_color, normalize_label

    if palette is None:
        raise ValueError("Pass palette=R_COLORS (or your color list) explicitly.")
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    for key, g in series.items():
        pts = sorted(g["points"], key=lambda p: p[0])
        ts = [p[0] for p in pts]
        ns = [p[1] for p in pts]
        label = normalize_label(g["label"])
        ax.plot(ts, ns, label=label, color=get_color(label, palette), linewidth=1.5)

    ax.set_xscale("log")
    ax.set_xlabel("t (s, log scale)")
    ax.set_ylabel("n(t) (mol)")
    title = "Uptake curves over time" if family_name is None else f"{family_name}: uptake curves, all geometries"
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    return ax


# %% [Cell 7b] Run it -- needs a specific family's `series`, e.g. FAMILIES["cone"]["series"]
fig, ax = plt.subplots(figsize=(8, 5))
plot_uptake_overlay(FAMILIES["cone"]["series"], family_name="cone", palette=R_COLORS, ax=ax)
fig.tight_layout()
fig.savefig("cone_uptake_overlay.pdf", bbox_inches="tight")
plt.show()


# %% [Cell 8] Plot 8 -- snapshot bar chart at a fixed reference time
def plot_bar_at_tprocess(master, t_col="n_t15000", top_n=None, family_colors=None):
    """
    Simple ranked bar at ONE fixed real time -- distinct from the
    leadership-share ranked bar (Cell 3), which scores duration-of-
    leadership across the whole early window. This one answers "who's
    ahead right at t=15,000s (or 2,000s / 134,900s)", the plain snapshot
    version, useful as a sanity-check companion to the more complex
    leadership-share metric.
    """
    if family_colors is None:
        family_colors = build_family_colors(master["family"].unique())

    param_cols = [c for c in master.columns if c not in METRIC_COLS]
    ranked = master.dropna(subset=[t_col]).sort_values(t_col, ascending=False)
    if top_n:
        ranked = ranked.head(top_n)
    labels = [geometry_label(row, param_cols) for _, row in ranked.iterrows()]
    colors = [family_colors[f] for f in ranked["family"]]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(ranked))))
    ax.barh(labels[::-1], ranked[t_col][::-1], color=colors[::-1])
    ax.set_xlabel(f"n at {t_col.replace('n_', '')} (mol)")
    ax.set_title(f"Ranking at fixed time: {t_col}")
    ax.grid(True, axis="x", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    return fig, ax


# %% [Cell 8b] Run it for each of your three reference times
for t_col in ["n_t2000", "n_t15000", "n_t134900"]:
    fig, ax = plot_bar_at_tprocess(master, t_col=t_col, top_n=20, family_colors=family_colors)
    fig.savefig(f"ranked_bar_{t_col}.pdf", bbox_inches="tight")
    plt.show()


# %% [Cell 9] Static results table export -- for direct drop-in to the report
def format_results_table(master, columns=None, round_digits=4):
    """
    Make a report-ready copy of ``master`` without changing the analysis
    dataframe. R/H are shown in micrometres, while small physical quantities
    retain scientific notation instead of being rounded to zero.
    """
    out = master.copy()
    if columns is not None:
        out = out.loc[:, columns].copy()

    rename = {}
    for col in ("R", "H"):
        if col in out:
            out[col] = out[col].map(
                lambda value: f"{value * 1e6:.{round_digits}f}" if pd.notna(value) else ""
            )
            rename[col] = f"{col} (um)"

    scientific_cols = {
        "SA", "V", "SA_V", "L_c", "n_t2000", "n_t15000", "n_t134900", "n_eq"
    }
    for col in scientific_cols & set(out.columns):
        out[col] = out[col].map(
            lambda value: f"{value:.{round_digits}e}" if pd.notna(value) else ""
        )

    fixed_cols = {"pct_eq_t15000", "early_leadership_share"}
    for col in fixed_cols & set(out.columns):
        out[col] = out[col].map(
            lambda value: f"{value:.{round_digits}f}" if pd.notna(value) else ""
        )

    return out.rename(columns=rename)


def export_results_table(master, path="results_table.csv", columns=None, round_digits=4):
    """Write a report-ready CSV without altering ``master``."""
    out = format_results_table(master, columns=columns, round_digits=round_digits)
    out.to_csv(path, index=False)
    print(f"Saved {len(out)} rows to {path}")
    return out


def results_table_markdown(master, columns=None, round_digits=4):
    """
    Same idea, but returns a Markdown table string -- paste straight into
    a Markdown-based report/README. Requires the `tabulate` package
    (pip install tabulate) for df.to_markdown(); falls back to a plain
    manual join if tabulate isn't installed.
    """
    out = format_results_table(master, columns=columns, round_digits=round_digits)
    try:
        return out.to_markdown(index=False)
    except ImportError:
        header = " | ".join(out.columns)
        sep = " | ".join(["---"] * len(out.columns))
        rows = " |\n| ".join(" | ".join(str(v) for v in row) for row in out.values)
        return f"| {header} |\n| {sep} |\n| {rows} |"


# %% [Cell 9b] Run it
master_rounded = export_results_table(master, path="master_table_all_families.csv")
print(results_table_markdown(
    master,
    columns=["family", "R", "H", "n_t15000", "pct_eq_t15000", "early_leadership_share"],
))
