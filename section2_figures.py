"""
section2_figures.py

Generates Figures 4-7 for the Recess/Pillar Family section, directly
from `master` (the dataframe your notebook already builds via
build_master_table()). Requires n_t500 to exist -- see
master_pipeline_patch.diff for the one-line change to REFERENCE_TIMES.

USAGE:

    from section2_figures import generate_all
    generate_all(master, out_dir="section2_figures")

Family name strings match your actual FAMILIES dict keys:
    cone_recessed, cylinder_recessed, square_recessed, grooves, edges,
    square_pillar, cylindrical_pillars

Families are organized into three geometric categories:
    Subtractive point recesses    -> cone_recessed, cylinder_recessed, square_recessed
    Subtractive extruded recesses -> grooves, edges
    Additive pillars               -> square_pillar, cylindrical_pillars

NOTE ON PILLARS: square_pillar and cylindrical_pillars are included in the
classification and color/label maps below so the script is ready for
them, but if your `master` dataframe does not yet contain rows for
these two families, every figure function will simply print a
"no rows for family ..., skipping" message and continue -- nothing
will break. Add pillar rows to `master` (with the same R/H/SA/V/n_*
columns as the recess families) and they'll appear automatically.

UNITS NOTE: all uptake quantities (n_t500, n_t2000, n_t15000, n_near_eq,
and everything derived from them) are converted from moles of CO2 to
kilograms of CO2 for every figure, using the CO2 molar mass
(44.01 g/mol). This conversion happens at plot time only -- it does not
modify `master` itself.

GROOVE NOTE: groove surface area is set by H only (SA ~= 2*H*L_unit, fixed
sidewall length), so R changes V without changing SA at fixed H -- unlike
cone/cylinder/square, where R moves both together. This is a real
geometric property of the family, not a data issue, so grooves is
included in every figure below. sa_normalized_table() and
fig5_sa_overlay() flag this explicitly for grooves (via the 'note' column
and legend label), since a flat line there means something different for
grooves (flat by construction) than for the other families (flat =
evidence that early uptake is SA-governed).

FIG 5 NOTE: the original per-family 2x2 panel figures (5 families x 4
panels = 20 panels) were cut after an audit found only one panel per
family (n(500s)/SA vs R) was ever cited in the report text, and only as a
single collapsed percentage; the other three panels were either unused
or fully superseded by the pooled n_eq/V collapse in fig6. Replaced with
sa_normalized_table() (the one real number per family, as a CSV/table)
and fig5_sa_overlay() (one consolidated plot, all families on a shared
axis). See those two functions' docstrings for details.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Unit conversion -- moles of CO2 -> kilograms of CO2
# ---------------------------------------------------------------------------

CO2_MOLAR_MASS_G_PER_MOL = 44.01
CO2_MOLAR_MASS_KG_PER_MOL = CO2_MOLAR_MASS_G_PER_MOL / 1000.0  # 0.04401 kg/mol


def _kg_co2(series_mol):
    """moles of CO2 -> kilograms of CO2, for plotting."""
    return series_mol * CO2_MOLAR_MASS_KG_PER_MOL


# ---------------------------------------------------------------------------
# Thesis style -- serif fonts, tight layout, consistent per-family colors
# ---------------------------------------------------------------------------

FAMILIES_SEC2 = [
    "cone_recessed", "cylinder_recessed", "square_recessed",  # subtractive point recesses
    "grooves", "edges",                                        # subtractive extruded recesses
    "square_pillar", "cylindrical_pillars",                        # additive pillars
]

# FAMILIES_SEC2 = all 7 families. It's the right default for the
# cross-family SYNTHESIS figures (fig6_neq_per_v, fig7_sav_vs_early,
# fig4b_cross_family), whose whole point is testing whether a relationship
# holds across subtractive AND additive geometries -- that's not a scoping
# bug, it's the analysis.
#
# But Section 2's own figures/tables (sa_normalized_table, fig5_sa_overlay)
# are scoped to subtractive geometry only -- pillars are analyzed
# separately in Section 3, same split already applied to spread_table.csv
# there. Those two use FAMILIES_SUBTRACTIVE below instead, not
# FAMILIES_SEC2, to keep that split correct by default rather than by
# remembering to pass the right argument every time.

FAMILIES_SUBTRACTIVE = [
    "cone_recessed", "cylinder_recessed", "square_recessed", "grooves", "edges",
]

FAMILIES_PILLARS = ["square_pillar", "cylindrical_pillars"]

FAMILY_LABELS = {
    "cone_recessed": "Conical Recess",
    "cylinder_recessed": "Cylindrical Recess",
    "square_recessed": "Square Recess",
    "grooves": "Rectangular Groove",
    "edges": "Triangular Edge Recess",
    "square_pillar": "Square Pillar",
    "cylindrical_pillars": "Cylindrical Pillar",
}

# Geometric classification used to group families in section headers /
# report structure. Order matches the requested taxonomy.
FAMILY_CATEGORY = {
    "cone_recessed": "Subtractive point recesses",
    "cylinder_recessed": "Subtractive point recesses",
    "square_recessed": "Subtractive point recesses",
    "grooves": "Subtractive extruded recesses",
    "edges": "Subtractive extruded recesses",
    "square_pillar": "Additive pillars",
    "cylindrical_pillars": "Additive pillars",
}

CATEGORY_ORDER = [
    "Subtractive point recesses",
    "Subtractive extruded recesses",
    "Additive pillars",
]

FAMILY_COLORS = {
    "cone_recessed": "#2E86AB",
    "cylinder_recessed": "#A23B72",
    "square_recessed": "#F18F01",
    "grooves": "#3B7A57",
    "edges": "#C1443C",
    "square_pillar": "#6A4C93",
    "cylindrical_pillars": "#1B998B",
}

# Times used for the cross-family comparison figure (fig4b). Not every
# REFERENCE_TIME needs a cross-family row -- t500/t15000 are more useful
# for the within-family Fig 4 (seeing one family evolve) than for
# across-family comparison, so only the two moments that matter most for
# "who's ahead" are given a dedicated cross-family panel.
CROSS_FAMILY_TIMES = [("n_t2000", "t = 2,000 s"), ("n_near_eq", "Equilibrium")]

GROOVE_SA_SET_BY_H_ONLY = True  # confirmed geometric fact, not a data bug:
# groove SA ~= 2*H*L_unit (fixed-length sidewalls), so R changes V without
# changing SA at fixed H. Cone/cylinder/square don't have this decoupling
# -- R changes both SA and V together for them. Kept as a flag so fig5's
# bottom-left panel (n(500s)/SA vs R) can be labeled correctly: for
# grooves that panel is expected to be flat by construction, not evidence
# of "SA governs early time" the way it is for the other three families.


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def _check_required_columns(master, cols):
    missing = [c for c in cols if c not in master.columns]
    if missing:
        raise ValueError(
            f"master is missing columns {missing}. If 'n_t500' is one of "
            f"them, apply master_pipeline_patch.diff (add 't500': 500 to "
            f"REFERENCE_TIMES) and re-run build_master_table()."
        )


def _um(series_m):
    """meters -> micrometers, for axis labels."""
    return series_m * 1e6


# ---------------------------------------------------------------------------
# Fig 4 -- R x H heatmap, 4 panels per family (t500 / t2000 / t15000 / eq)
# ---------------------------------------------------------------------------

FIG4_METRICS = [
    ("n_t500", "t = 500 s"),
    ("n_t2000", "t = 2,000 s"),
    ("n_t15000", "t = 15,000 s"),
    ("n_near_eq", "Equilibrium"),
]


def fig4_heatmap(master, family, out_dir):
    _check_required_columns(master, [m for m, _ in FIG4_METRICS] + ["R", "H"])
    sub = master[master["family"] == family].copy()
    if sub.empty:
        print(f"[fig4] no rows for family '{family}', skipping")
        return None

    sub["R_um"] = _um(sub["R"])
    sub["H_um"] = _um(sub["H"])
    # convert uptake columns to kg CO2 before pivoting
    for metric, _ in FIG4_METRICS:
        sub[metric] = _kg_co2(sub[metric])
    R_vals = sorted(sub["R_um"].unique())
    H_vals = sorted(sub["H_um"].unique())

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, (metric, title) in zip(axes, FIG4_METRICS):
        grid = sub.pivot(index="H_um", columns="R_um", values=metric).reindex(
            index=H_vals, columns=R_vals
        )
        im = ax.imshow(grid.values, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(R_vals)))
        ax.set_xticklabels([f"{r:.0f}" for r in R_vals])
        ax.set_yticks(range(len(H_vals)))
        ax.set_yticklabels([f"{h:.0f}" for h in H_vals])
        ax.set_xlabel("R (\u00b5m)")
        if ax is axes[0]:
            ax.set_ylabel("H (\u00b5m)")
        ax.set_title(title)

        # annotate each cell with its value (kg CO2)
        vmax_idx = np.unravel_index(np.nanargmax(grid.values), grid.values.shape)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                val = grid.values[i, j]
                if np.isnan(val):
                    continue
                is_opt = (i, j) == vmax_idx
                ax.text(j, i, f"{val:.2e}", ha="center", va="center",
                         fontsize=6, color="white" if not is_opt else "red",
                         fontweight="bold" if is_opt else "normal")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label("CO$_2$ uptake (kg)", fontsize=7)

    category = FAMILY_CATEGORY.get(family, "")
    fig.suptitle(f"{FAMILY_LABELS[family]} ({category}) -- CO$_2$ uptake vs. R, H across four process times", y=1.05)
    fig.tight_layout()
    path = os.path.join(out_dir, f"fig4_heatmap_{family}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[fig4] saved -> {path}")
    return path


# ---------------------------------------------------------------------------
# Fig 4b -- cross-family comparison: one row of panels per family, at a
# fixed time, shared color scale within the row so families are actually
# visually comparable (unlike Fig 4, where each family's 4 panels use
# independent scales and are meant for within-family time evolution, not
# cross-family comparison).
#
# Defaults to FAMILIES_SUBTRACTIVE, matching sa_normalized_table,
# fig5_sa_overlay, fig6_neq_per_v, and fig7_sav_vs_early -- Section 2's
# draft prose only discusses the five subtractive families in its
# Cross-Family Synthesis subsection; pillars are deferred to Section
# 1.1.2. Pass families=FAMILIES_SEC2 explicitly if/when an all-7-family
# version is needed for Section 4.
# ---------------------------------------------------------------------------

def fig4b_cross_family(master, out_dir, metric, title, families=FAMILIES_SUBTRACTIVE):
    _check_required_columns(master, [metric, "R", "H"])
    n = len(families)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6))
    if n == 1:
        axes = [axes]

    # shared color scale across the row (kg CO2)
    vmin, vmax = np.inf, -np.inf
    grids = {}
    for family in families:
        sub = master[master["family"] == family].copy()
        if sub.empty:
            continue
        sub["R_um"] = _um(sub["R"])
        sub["H_um"] = _um(sub["H"])
        sub[metric] = _kg_co2(sub[metric])
        R_vals = sorted(sub["R_um"].unique())
        H_vals = sorted(sub["H_um"].unique())
        grid = sub.pivot(index="H_um", columns="R_um", values=metric).reindex(index=H_vals, columns=R_vals)
        grids[family] = (grid, R_vals, H_vals)
        vmin = min(vmin, np.nanmin(grid.values))
        vmax = max(vmax, np.nanmax(grid.values))

    im = None
    for ax, family in zip(axes, families):
        if family not in grids:
            ax.axis("off")
            continue
        grid, R_vals, H_vals = grids[family]
        im = ax.imshow(grid.values, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(R_vals)))
        ax.set_xticklabels([f"{r:.0f}" for r in R_vals], fontsize=7)
        ax.set_yticks(range(len(H_vals)))
        ax.set_yticklabels([f"{h:.0f}" for h in H_vals], fontsize=7)
        ax.set_xlabel("R (\u00b5m)", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("H (\u00b5m)", fontsize=8)
        label = FAMILY_LABELS[family]
        if GROOVE_SA_SET_BY_H_ONLY and family == "grooves":
            label += "*"
        ax.set_title(label, fontsize=9)

    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="CO$_2$ uptake (kg)")

    fig.suptitle(f"{title} -- cross-family comparison, shared color scale", y=1.03)
    footnote = "* groove SA is set by H only, independent of R at this fixed H (see Fig 5)" \
        if GROOVE_SA_SET_BY_H_ONLY else ""
    if footnote:
        fig.text(0.01, -0.02, footnote, fontsize=7, color="#666666")

    safe_name = metric.replace("n_", "")
    path = os.path.join(out_dir, f"fig4b_cross_family_{safe_name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig4b] saved -> {path}")
    return path


# ---------------------------------------------------------------------------
# Fig 5 REPLACEMENT -- sa_normalized_table() + one consolidated overlay figure
#
# The original fig5_radius_comparison() produced a 2x2 panel figure per
# family (5 families x 4 panels = 20 panels total). Audit of the report
# text found only one panel per family (bottom-left, n(500s)/SA vs R) was
# ever actually cited, and only as a single collapsed percentage (0.5%,
# 2.3%, 2.7%, flat-by-construction, +1.6%). The other three panels per
# family were either never quoted (raw n(500s), raw n_eq) or fully
# superseded by the pooled n_eq/V collapse already shown in fig6
# (bottom-right panel restated "n_eq/V is flat" once per family with no
# new information -- fig6 already makes this claim once, better, with a
# quantified coefficient of variation across all families).
#
# Replaced with:
#   1. sa_normalized_table() -- the one real number per family, computed
#      from master (not read off a plot), matching the spread_table()
#      pattern already used elsewhere in this section.
#   2. fig5_sa_overlay() -- ONE consolidated figure, all families on a
#      shared Delta-percent-from-R=5 axis. This is the only panel type
#      that does something a table can't (letting the reader see cone's
#      flatness and edges' sign-flip on the same plot, same scale).
#
# The old fig5_radius_comparison() function is intentionally not kept in
# this file -- if you need the original 20-panel version for reference,
# it's in version control history prior to this change.
# ---------------------------------------------------------------------------

def sa_normalized_table(master, families=FAMILIES_SUBTRACTIVE, out_dir="section2_figures",
                         H_fixed_um=35, filename_suffix=""):
    """
    The one real number per family from the old Fig 5 bottom-left panel:
    percent change in n(t=500s)/SA between the smallest and largest R
    sampled, at fixed H. Computed directly from master, not read off a
    plot -- same discipline as spread_table().

    For grooves, this is flat by construction (SA is set by H only, see
    module docstring), not because grooves obeys the same near-flat-slab
    early-time physics as the other families -- flagged in the 'note'
    column rather than left to look like a coincidence.
    """
    _check_required_columns(master, ["n_t500", "SA", "R", "H"])
    rows = []
    for family in families:
        sub = master[master["family"] == family].copy()
        if sub.empty:
            continue
        sub["H_um"] = _um(sub["H"])
        sub = sub[np.isclose(sub["H_um"], H_fixed_um)].sort_values("R")
        if sub.empty:
            continue
        sub["n_t500_kg"] = _kg_co2(sub["n_t500"])
        sub["n_t500_per_SA"] = sub["n_t500_kg"] / sub["SA"]

        y0 = sub["n_t500_per_SA"].iloc[0]
        y1 = sub["n_t500_per_SA"].iloc[-1]
        pct = (y1 - y0) / y0 * 100 if y0 != 0 else float("nan")

        note = ""
        if GROOVE_SA_SET_BY_H_ONLY and family == "grooves":
            note = "flat by construction (SA set by H only, independent of R)"

        rows.append({
            "family": FAMILY_LABELS[family],
            "category": FAMILY_CATEGORY.get(family, ""),
            "H_fixed_um": H_fixed_um,
            "R_min_um": round(_um(sub["R"].iloc[0]), 1),
            "R_max_um": round(_um(sub["R"].iloc[-1]), 1),
            "delta_n_t500_per_SA_pct": round(pct, 1),
            "note": note,
        })

    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"sa_normalized_table{filename_suffix}.csv")
    df.to_csv(path, index=False)
    print(f"[sa_normalized_table] Delta n(500s)/SA, R_min->R_max at H={H_fixed_um}um -> {path}")
    print(df.to_string(index=False))
    return df


def fig5_sa_overlay(master, out_dir, families=FAMILIES_SUBTRACTIVE, H_fixed_um=35, filename_suffix=""):
    """
    The one figure worth keeping from the old Fig 5 set: n(500s)/SA vs R
    for every family, overlaid on a single shared axis, normalized to
    percent change from each family's own R=R_min value. Replaces the
    bottom-left panel of all five (or seven) old per-family figures.
    """
    _check_required_columns(master, ["n_t500", "SA", "R", "H"])
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for family in families:
        sub = master[master["family"] == family].copy()
        if sub.empty:
            continue
        sub["H_um"] = _um(sub["H"])
        sub = sub[np.isclose(sub["H_um"], H_fixed_um)].sort_values("R")
        if sub.empty:
            continue
        sub["R_um"] = _um(sub["R"])
        sub["n_t500_kg"] = _kg_co2(sub["n_t500"])
        sub["n_t500_per_SA"] = sub["n_t500_kg"] / sub["SA"]

        y0 = sub["n_t500_per_SA"].iloc[0]
        sub["pct_from_Rmin"] = (sub["n_t500_per_SA"] - y0) / y0 * 100

        label = FAMILY_LABELS[family]
        if GROOVE_SA_SET_BY_H_ONLY and family == "grooves":
            label += " (SA set by H only)"
        ax.plot(sub["R_um"], sub["pct_from_Rmin"], "o-", color=FAMILY_COLORS[family],
                label=label, linewidth=1.5, markersize=5)

    ax.axhline(0, color="#999999", linewidth=0.8, linestyle=":")
    ax.set_xlabel("R (\u00b5m)")
    ax.set_ylabel("$\\Delta\\,[n(500\\,\\mathrm{s})/SA]$ from $R_{min}$ (%)")
    ax.set_title(f"Surface-normalized early uptake vs. R (H = {H_fixed_um} \u00b5m)")
    ax.legend(loc="best", framealpha=0.9, fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, f"fig5_sa_overlay{filename_suffix}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[fig5_sa_overlay] saved -> {path}")
    return path


# ---------------------------------------------------------------------------
# Fig 6 -- n_eq / V vs R, all families overlaid, with pooled constant line
#
# Section 2's draft prose cites this figure as covering the five
# SUBTRACTIVE families only (pooled CoV = 0.0001) -- the all-7-family
# version (pillars included, CoV = 0.0002 per the Section 4/5 handover
# notes) is a distinct, later result and belongs in Section 4, not here.
# Defaults to FAMILIES_SUBTRACTIVE for that reason, matching
# sa_normalized_table/fig5_sa_overlay. Pass families=FAMILIES_SEC2
# explicitly if/when the Section 4 all-7 version is actually being built.
#
# Plot is in % deviation from the pooled mean, not raw kg CO2/m^3 --
# the raw axis (~1.5570e7 to 1.5578e7) forced matplotlib to compress a
# genuinely tiny 0.08% spread into a range that reads as illegible noise
# on the original scale. Percent-deviation makes the "collapse" claim
# actually visible instead of only provable via the CoV in the caption.
# Small per-family horizontal jitter at each R value is added so five
# families' markers don't stack exactly on top of each other and hide
# all but the last-drawn color.
# ---------------------------------------------------------------------------

def fig6_neq_per_v(master, out_dir, families=FAMILIES_SUBTRACTIVE, flat_ref=None):
    """
    flat_ref: optional (n_eq_flat_mol, V_flat_m3) tuple for the untextured
    baseline -- e.g. (baseline["n_near_eq"], V_flat) where baseline comes
    from preprocessing.build_untextured_baseline(). When given, adds a
    second reference line for rho_flat alongside the pooled-family rho,
    so the plot answers both "do families collapse onto one constant"
    AND "how does that constant compare to no texture at all". V_flat
    must be supplied separately (this module doesn't parse volume
    exports) since the untextured run currently has no accompanying
    volume .csv -- hand-enter it from the COMSOL model if you have it,
    or back it out from rho and n_eq_flat as a sanity check (see report
    note: predicted V_flat ~4.999e-13 m^3 already lines up with the
    known V_unit range for the shallowest recess geometries).
    """
    _check_required_columns(master, ["n_near_eq", "V", "R"])
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    all_ratios = []
    per_family = []
    for family in families:
        sub = master[master["family"] == family].copy()
        if sub.empty:
            continue
        sub["R_um"] = _um(sub["R"])
        sub["n_eq_per_V"] = _kg_co2(sub["n_near_eq"]) / sub["V"]
        all_ratios.extend(sub["n_eq_per_V"].tolist())
        per_family.append((family, sub))

    rho = np.mean(all_ratios)
    cv = np.std(all_ratios) / rho

    # small deterministic horizontal jitter per family so markers at the
    # same R value fan out instead of stacking on top of each other
    n_fam = len(per_family)
    jitter_span_um = 3.0  # total spread across all families, in um
    for i, (family, sub) in enumerate(per_family):
        offset = (i - (n_fam - 1) / 2) * (jitter_span_um / max(n_fam - 1, 1))
        pct_dev = (sub["n_eq_per_V"] - rho) / rho * 100

        label = FAMILY_LABELS[family]
        if GROOVE_SA_SET_BY_H_ONLY and family == "grooves":
            label += " (SA set by H only)"
        ax.scatter(sub["R_um"] + offset, pct_dev, color=FAMILY_COLORS[family],
                   label=label, s=40, alpha=0.85)

    ax.axhline(0, color="black", linestyle="--", linewidth=1,
               label=f"pooled mean $\\rho$ = {rho:.3e} kg CO$_2$/m$^3$ (CV={cv:.4f})")

    if flat_ref is not None:
        n_eq_flat_mol, V_flat = flat_ref
        if n_eq_flat_mol is not None and V_flat:
            rho_flat = _kg_co2(n_eq_flat_mol) / V_flat
            pct_flat = (rho_flat - rho) / rho * 100
            ax.axhline(pct_flat, color="#888888", linestyle=":", linewidth=1.2,
                       label=f"untextured $\\rho_{{flat}}$ = {rho_flat:.3e} kg/m$^3$ "
                             f"({pct_flat:+.2f}% vs. pooled)")

    ax.set_xlabel("R (\u00b5m)")
    ax.set_ylabel("$n_{eq} / V$ deviation from pooled mean (%)")
    ax.set_title("Equilibrium capacity density is independent of geometry\n"
                 "(all points collapse onto one constant, regardless of shape or R)")
    ax.legend(loc="best", framealpha=0.9, fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig6_neq_per_v_collapse.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[fig6] saved -> {path}  (rho={rho:.4e}, CV={cv:.5f})")
    return path, rho


# ---------------------------------------------------------------------------
# Fig 7 -- SA/V vs early uptake (n_t500), all families overlaid, no L_c
# ---------------------------------------------------------------------------

def fig7_sav_vs_early(master, out_dir, families=FAMILIES_SUBTRACTIVE, metric="n_t500"):
    _check_required_columns(master, ["SA_V", metric])
    fig, ax = plt.subplots(figsize=(7, 5))

    xs, ys = [], []
    for family in families:
        sub = master[master["family"] == family].copy()
        if sub.empty:
            continue
        sub[metric] = _kg_co2(sub[metric])
        ax.scatter(sub["SA_V"], sub[metric], color=FAMILY_COLORS[family],
                   label=FAMILY_LABELS[family], s=35, alpha=0.85)
        xs.extend(sub["SA_V"].tolist())
        ys.extend(sub[metric].tolist())

    xs, ys = np.array(xs), np.array(ys)
    b, a = np.polyfit(xs, ys, 1)
    r = np.corrcoef(xs, ys)[0, 1]
    x_line = np.linspace(xs.min(), xs.max(), 50)
    ax.plot(x_line, a + b * x_line, color="black", linestyle="--", linewidth=1,
           label=f"pooled fit, r = {r:.2f}")

    ax.set_xlabel("SA/V (1/m)")
    metric_label = {"n_t500": "$t=500$ s", "n_t2000": "$t=2{,}000$ s"}.get(metric, metric)
    ax.set_ylabel(f"CO$_2$ uptake, n({metric_label}) (kg)")
    ax.set_title("Early uptake collapses onto a single SA/V trend across shapes\n"
                 "(shape has no independent effect beyond SA/V at early time)")
    ax.legend(loc="best", framealpha=0.9)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    path = os.path.join(out_dir, f"fig7_sav_vs_{metric}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[fig7] saved -> {path}  (pooled r={r:.3f})")
    return path, (a, b, r)


# ---------------------------------------------------------------------------
# Fig 8 -- enhancement/reduction over the untextured (flat-film) baseline
#
# This is the "how much better than no texture" figure -- previously
# missing entirely, since every comparison up to this point has been
# texture-vs-texture, not texture-vs-flat. Belongs in the Equilibrium
# Trade-off / Sign-Invariance section: subtractive families should sit
# BELOW 1.0 at equilibrium (less remaining polymer volume than the flat
# film) and additive pillars should sit ABOVE 1.0 (more), which is the
# direct quantitative complement to the sign-invariance finding there.
# ---------------------------------------------------------------------------

UNTEXTURED_METRICS = [
    ("n_t500", "t = 500 s"),
    ("n_t2000", "t = 2,000 s"),
    ("n_t15000", "t = 15,000 s"),
    ("n_near_eq", "Equilibrium"),
]


def fig8_vs_untextured(master, out_dir, baseline, families=FAMILIES_SEC2):
    """
    baseline: dict from preprocessing.build_untextured_baseline(), i.e.
    {"n_t500": ..., "n_t2000": ..., "n_t15000": ..., "n_near_eq": ...},
    values in mol CO2 (same raw units as master's n_* columns -- kg
    conversion happens here internally, same convention as every other
    figure in this module).

    For each family, takes its best-performing configuration at each
    checkpoint (max over that family's R x H grid) and expresses it as a
    ratio against the untextured baseline at the same checkpoint. A bar
    at 1.0 means "no better than an untextured film"; subtractive
    families are expected to fall below 1.0 at equilibrium (less
    remaining polymer than the flat film) and pillars above it (more).

    Note: n_t15000 in baseline is linearly interpolated (via
    preprocessing.interp_at), matching how every textured family's own
    n_t15000 is computed -- not a bespoke interpolation scheme for the
    baseline alone. See build_untextured_baseline()'s docstring.
    """
    setup_style()
    _check_required_columns(master, [m for m, _ in UNTEXTURED_METRICS])
    fig, ax = plt.subplots(figsize=(10, 5.5))

    n_metrics = len(UNTEXTURED_METRICS)
    n_fam = len(families)
    bar_w = 0.8 / n_metrics
    x = np.arange(n_fam)

    for i, (metric, label) in enumerate(UNTEXTURED_METRICS):
        n_flat_mol = baseline.get(metric)
        if n_flat_mol is None:
            print(f"[fig8] baseline missing '{metric}', skipping this checkpoint")
            continue
        n_flat_kg = _kg_co2(n_flat_mol)
        ratios = []
        for family in families:
            sub = master[master["family"] == family]
            vals = sub[metric].dropna() if metric in sub.columns else pd.Series(dtype=float)
            if vals.empty:
                ratios.append(np.nan)
                continue
            ratios.append(_kg_co2(vals.max()) / n_flat_kg)
        offset = (i - (n_metrics - 1) / 2) * bar_w
        ax.bar(x + offset, ratios, width=bar_w, label=label)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1,
               label="untextured baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([FAMILY_LABELS[f] for f in families], rotation=30, ha="right")
    ax.set_ylabel("$n(t) / n_{flat}(t)$ -- best configuration per family")
    ax.set_title("Enhancement (or reduction) over the untextured film,\n"
                 "best-performing configuration per family, at each checkpoint")
    ax.legend(loc="best", framealpha=0.9, fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig8_vs_untextured.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[fig8] saved -> {path}")
    return path


# ---------------------------------------------------------------------------
# Spread table -- (max-min)/min, ONE consistent formula, across all metrics
# and families. Generated from code, not read off heatmap pixels -- avoids
# the mean-vs-min inconsistency found when cross-checking hand-typed prose
# numbers against the actual figures.
#
# NOTE: spreads here are computed on the raw mol values (units cancel out
# of a ratio like (max-min)/min), so this table is correct whether or not
# the figures elsewhere in this module are displaying kg CO2 or mol --
# the % spread numbers are identical either way.
# ---------------------------------------------------------------------------

def spread_table(master, families=FAMILIES_SEC2,
                  metrics=("n_t500", "n_t2000", "n_t15000", "n_near_eq"),
                  out_dir="section2_figures"):
    rows = []
    for family in families:
        sub = master[master["family"] == family]
        if sub.empty:
            continue
        row = {"family": FAMILY_LABELS[family], "category": FAMILY_CATEGORY.get(family, "")}
        for metric in metrics:
            if metric not in sub.columns:
                continue
            vals = sub[metric].dropna()
            if vals.empty:
                continue
            vmax, vmin = vals.max(), vals.min()
            spread_pct = (vmax - vmin) / vmin * 100 if vmin != 0 else float("nan")
            row[f"{metric}_spread_pct"] = round(spread_pct, 1)
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "spread_table.csv")
    df.to_csv(path, index=False)
    print(f"[spread_table] (max-min)/min, one formula throughout -> {path}")
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------

def generate_all(master, out_dir="section2_figures", untextured_baseline=None, V_flat=None):
    """
    untextured_baseline: optional dict from
    preprocessing.build_untextured_baseline() -- if given, also generates
    fig8_vs_untextured(). V_flat: optional untextured unit-cell volume
    (m^3) -- if given ALONGSIDE untextured_baseline, also adds the
    rho_flat reference line to fig6. Both default to None so this
    function still runs unchanged for anyone not yet using the baseline.
    """
    setup_style()
    os.makedirs(out_dir, exist_ok=True)

    for family in FAMILIES_SEC2:
        fig4_heatmap(master, family, out_dir)

    for metric, title in CROSS_FAMILY_TIMES:
        fig4b_cross_family(master, out_dir, metric, title)

    # Section 2 (subtractive only)
    sa_normalized_table(master, families=FAMILIES_SUBTRACTIVE, out_dir=out_dir)
    fig5_sa_overlay(master, families=FAMILIES_SUBTRACTIVE, out_dir=out_dir)

    # Section 3 (pillars only) -- same two outputs, filtered to the 2
    # additive families, written under a distinct filename so they don't
    # overwrite Section 2's subtractive-only versions above.
    sa_normalized_table(master, families=FAMILIES_PILLARS, out_dir=out_dir, filename_suffix="_pillars")
    fig5_sa_overlay(master, families=FAMILIES_PILLARS, out_dir=out_dir, filename_suffix="_pillars")

    flat_ref = None
    if untextured_baseline is not None and V_flat is not None:
        flat_ref = (untextured_baseline.get("n_near_eq"), V_flat)
    fig6_neq_per_v(master, out_dir, flat_ref=flat_ref)
    fig7_sav_vs_early(master, out_dir, metric="n_t500")
    spread_table(master, out_dir=out_dir)

    if untextured_baseline is not None:
        fig8_vs_untextured(master, out_dir, untextured_baseline)

    print(f"\nDone. Figures written to ./{out_dir}/")


if __name__ == "__main__":
    print(__doc__)