"""Section 4 cross-family figures, including the actual untextured film.

The seven texture families remain the swept dataset.  The flat-film run is a
single reference simulation: it is never assigned R/H or included in
family-level correlations/regressions.

Run this file directly to rebuild the current master table from ``data/`` and
write all four figures to ``section4_figures/``.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from preprocessing import (AREA_FLAT_M2, V_FLAT_M3, build_family_entry,
                           build_untextured_baseline, interp_at)
from master_pipeline import build_master_table
from section2_figures import FAMILY_LABELS, setup_style


ALL_FAMILIES = [
    "cone_recessed", "cylinder_recessed", "square_recessed",
    "grooves", "edges", "square_pillar", "cylindrical_pillars",
]

FAMILY_COLORS = {
    "cone_recessed": "#2E86AB", "cylinder_recessed": "#A23B72",
    "square_recessed": "#F18F01", "grooves": "#3B7A57",
    "edges": "#C1443C", "square_pillar": "#6A4C93",
    "cylindrical_pillars": "#1B998B",
}
EARLY_COL = "n_t500_kg"
EQ_COL = "n_near_eq_kg"
UNTEXTURED_LABEL = "Untextured"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _check_required_columns(master, cols):
    missing = [c for c in cols if c not in master.columns]
    if missing:
        raise ValueError(f"master is missing columns {missing}; rebuild it with build_master_table().")


def _default_baseline():
    with open(os.path.join(DATA_DIR, "untextured-uptake.txt"), encoding="utf-8") as handle:
        return build_untextured_baseline(handle.read())


def _flat_values_kg(baseline):
    """Comparable checkpoint/window values; baseline itself stays in mol."""
    return (baseline["n_t500"] * 0.04401, baseline["n_near_eq"] * 0.04401)


def _plot_family_points(ax, sub, x_col, y_col):
    for family in ALL_FAMILIES:
        group = sub[sub["family"] == family]
        if group.empty:
            continue
        additive = (group["volume_type"] == "added").all()
        ax.scatter(group[x_col], group[y_col], color=FAMILY_COLORS[family],
                   label=FAMILY_LABELS.get(family, family), marker="^" if additive else "o",
                   s=45 if additive else 35, alpha=0.85,
                   edgecolor="black" if additive else "none", linewidth=0.6)


def check_tradeoff(master, families=ALL_FAMILIES):
    """Return per-textured-family rank correlations; excludes flat film."""
    _check_required_columns(master, [EARLY_COL, EQ_COL, "family", "volume_type"])
    sub = master[master["family"].isin(families)].dropna(subset=[EARLY_COL, EQ_COL])
    rows = []
    print("Per-family Spearman rank correlation (textured families only):")
    for family, group in sub.groupby("family"):
        if len(group) < 3:
            continue
        rho, p = spearmanr(group[EARLY_COL], group[EQ_COL])
        rows.append({"family": family, "rho": rho, "p": p, "n": len(group)})
        print(f"  {family:20s} rho={rho:+.3f}  p={p:.3g}  n={len(group)}")
    return pd.DataFrame(rows).sort_values("rho")


def fig4_1_tradeoff(master, out_dir, baseline=None, families=ALL_FAMILIES):
    """Early uptake vs equilibrium capacity, with the flat-film reference."""
    setup_style(); os.makedirs(out_dir, exist_ok=True)
    _check_required_columns(master, [EARLY_COL, EQ_COL, "family", "volume_type"])
    baseline = _default_baseline() if baseline is None else baseline
    sub = master[master["family"].isin(families)].dropna(subset=[EARLY_COL, EQ_COL])

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    _plot_family_points(ax, sub, EARLY_COL, EQ_COL)
    flat_early, flat_eq = _flat_values_kg(baseline)
    ax.scatter(flat_early, flat_eq, marker="s", s=60, color="white", edgecolor="black",
               linewidth=1.1, label=UNTEXTURED_LABEL, zorder=5)

    rhos = []
    for family in families:
        group = sub[sub["family"] == family]
        if len(group) >= 3 and (group["volume_type"] == "removed").all():
            rho, _ = spearmanr(group[EARLY_COL], group[EQ_COL])
            rhos.append(f"{family}: $\\rho$={rho:+.2f}")
    if rhos:
        ax.annotate("Subtractive rank correlation\n(textured only):\n" + "\n".join(rhos),
                    xy=(0.02, 0.98), xycoords="axes fraction", ha="left", va="top", fontsize=7.5,
                    bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.9))
    ax.set_xlabel("n(t = 500 s)  [kg CO$_2$]")
    ax.set_ylabel("n$_{eq}$ (windowed avg, 160k–200k s)  [kg CO$_2$]")
    ax.set_title("Early-time uptake vs. equilibrium capacity\n(triangles = additive; square = untextured)")
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig4_1_tradeoff.png")
    fig.savefig(path, dpi=200); plt.close(fig)
    print(f"[fig4.1] saved -> {path}")
    return path


def fig4_2_neq_per_v_all(master, out_dir, baseline=None, families=ALL_FAMILIES):
    """Categorical n_eq/V comparison; flat film has no artificial R coordinate."""
    setup_style(); os.makedirs(out_dir, exist_ok=True)
    _check_required_columns(master, [EQ_COL, "V", "family"])
    baseline = _default_baseline() if baseline is None else baseline
    sub = master[master["family"].isin(families)].dropna(subset=[EQ_COL, "V"]).copy()
    sub["density"] = sub[EQ_COL] / sub["V"]
    flat_density = baseline["n_near_eq"] * 0.04401 / V_FLAT_M3
    categories = [UNTEXTURED_LABEL] + families
    positions = {name: idx for idx, name in enumerate(categories)}
    rng = np.random.default_rng(4)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(positions[UNTEXTURED_LABEL], flat_density, marker="s", s=60, color="white",
               edgecolor="black", linewidth=1.1, zorder=5, label=UNTEXTURED_LABEL)
    for family in families:
        group = sub[sub["family"] == family]
        if group.empty:
            continue
        jitter = rng.uniform(-0.16, 0.16, len(group))
        additive = (group["volume_type"] == "added").all()
        ax.scatter(positions[family] + jitter, group["density"], color=FAMILY_COLORS[family],
                   marker="^" if additive else "o", s=32, alpha=0.8,
                   edgecolor="black" if additive else "none", linewidth=0.5,
                   label=FAMILY_LABELS.get(family, family))
        ax.plot(positions[family], group["density"].mean(), marker="_", markersize=20,
                markeredgewidth=2, color="black", linestyle="None", zorder=6)
    pooled = np.r_[sub["density"].to_numpy(), flat_density]
    pooled_mean = pooled.mean(); cv = pooled.std(ddof=1) / pooled_mean
    ax.axhline(pooled_mean, color="black", linestyle="--", linewidth=1,
               label=f"pooled mean = {pooled_mean:.3e} kg/m$^3$ (CV={cv:.4f})")
    texture_mean = sub["density"].mean()
    delta_flat = (flat_density - texture_mean) / texture_mean * 100
    ax.annotate(f"flat vs. textured mean: {delta_flat:+.3f}%", xy=(0.98, 0.04),
                xycoords="axes fraction", ha="right", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.9))
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([UNTEXTURED_LABEL] + [FAMILY_LABELS.get(f, f) for f in families], rotation=30, ha="right")
    ax.set_ylabel("$n_{eq} / V$  (kg CO$_2$/m$^3$)")
    ax.set_title("Equilibrium capacity density across texture, geometry, and sign\n(marker bar = family mean)")
    ax.legend(fontsize=7, loc="best", ncol=2)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig4_2_neq_per_v_all.png")
    fig.savefig(path, dpi=200); plt.close(fig)
    print(f"[fig4.2] saved -> {path}  (pooled density={pooled_mean:.4e} kg/m^3, CV={cv:.5f})")
    return path


def fig4_3_sav_early_all(master, out_dir, baseline=None, families=ALL_FAMILIES):
    """SA/V vs early uptake; the untextured square is not fit to the texture trend."""
    setup_style(); os.makedirs(out_dir, exist_ok=True)
    _check_required_columns(master, ["SA_V", EARLY_COL, "volume_type", "family"])
    baseline = _default_baseline() if baseline is None else baseline
    sub = master[master["family"].isin(families)].dropna(subset=["SA_V", EARLY_COL]).copy()
    subtractive = sub[sub["volume_type"] == "removed"]
    x_sub, y_sub = subtractive["SA_V"].to_numpy(), subtractive[EARLY_COL].to_numpy()
    slope, intercept = np.polyfit(x_sub, y_sub, 1)
    r_sub = np.corrcoef(x_sub, y_sub)[0, 1]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    _plot_family_points(ax, sub, "SA_V", EARLY_COL)
    flat_early, _ = _flat_values_kg(baseline)
    flat_sav = AREA_FLAT_M2 / V_FLAT_M3
    ax.scatter(flat_sav, flat_early, marker="s", s=60, color="white", edgecolor="black",
               linewidth=1.1, label=UNTEXTURED_LABEL, zorder=5)
    x_line = np.linspace(x_sub.min(), x_sub.max(), 50)
    ax.plot(x_line, intercept + slope * x_line, color="black", linestyle="--", linewidth=1,
            label=f"subtractive fit (textured only), r={r_sub:.2f}")
    ax.set_xlabel("SA/V (1/m)")
    ax.set_ylabel("n(t = 500 s)  [kg CO$_2$]")
    ax.set_title("Early-time uptake vs. SA/V, all textures plus untextured reference\n(triangles = additive; square = untextured)")
    ax.legend(fontsize=7.2, loc="lower right")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig4_3_sav_early_all.png")
    fig.savefig(path, dpi=200); plt.close(fig)
    print(f"[fig4.3] saved -> {path}  (subtractive textured-only r={r_sub:.3f})")
    return path


def representative_configurations(master, families=ALL_FAMILIES):
    """One fixed, reproducible configuration per family: its t=500-s maximum.

    Section 2's current best-configuration comparison is checkpoint-specific,
    so it does not supply a fixed trajectory. Selecting on the common early
    checkpoint preserves a single configuration for each full Fig. 4.4 curve.
    """
    _check_required_columns(master, ["family", "R", "H", EARLY_COL])
    reps = {}
    for family in families:
        group = master[master["family"] == family].dropna(subset=[EARLY_COL])
        if not group.empty:
            row = group.loc[group[EARLY_COL].idxmax()]
            reps[family] = (row["R"], row["H"])
    return reps


# ---------------------------------------------------------------------------
# Archived Fig. 4.4: continuous trajectories. Kept for reference only;
# generate_all() intentionally does not call this function.
# ---------------------------------------------------------------------------
def fig4_4_relative_vs_untextured_archived(master, family_entries, out_dir, baseline=None,
                                            representatives=None, families=ALL_FAMILIES):
    """ARCHIVED: fixed-family trajectories relative to the flat-film series."""
    setup_style(); os.makedirs(out_dir, exist_ok=True)
    baseline = _default_baseline() if baseline is None else baseline
    representatives = representative_configurations(master, families) if representatives is None else representatives
    flat_points = baseline["points"]
    fig, ax = plt.subplots(figsize=(8, 5.8))
    used = {}
    for family in families:
        if family not in representatives or family not in family_entries:
            continue
        R, H = representatives[family]
        series = family_entries[family]["series"]
        key = min(series, key=lambda candidate: abs(candidate[0] - R) + abs(candidate[1] - H))
        points = np.asarray(series[key]["points"], dtype=float)
        times, uptake = points[:, 0], points[:, 1]
        flat = np.asarray([interp_at(flat_points, time) for time in times])
        valid = (times > 0) & (flat > 0)
        relative = 100 * (uptake[valid] - flat[valid]) / flat[valid]
        ax.plot(times[valid], relative, color=FAMILY_COLORS[family], linewidth=1.6,
                label=FAMILY_LABELS.get(family, family))
        used[family] = key
    ax.axhline(0, color="black", linestyle="--", linewidth=1, label="untextured reference")
    ax.set_xscale("log")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Relative uptake vs. untextured (%)")
    ax.set_title("Fixed textured configurations relative to the untextured film")
    ax.legend(fontsize=7.5, loc="best", ncol=2)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig4_4_relative_vs_untextured.png")
    fig.savefig(path, dpi=200); plt.close(fig)
    choices = ", ".join(f"{family}=(R={R * 1e6:g}, H={H * 1e6:g}) µm" for family, (R, H) in used.items())
    print(f"[fig4.4] saved -> {path}  (fixed representatives: {choices})")
    return path, used


FIG4_4_TIMES = (500, 2_000, 15_000, 100_000)


def fig4_4_fixed_time_relative_vs_untextured(master, family_entries, out_dir,
                                               baseline=None, representatives=None,
                                               families=ALL_FAMILIES,
                                               times=FIG4_4_TIMES):
    """Grouped fixed-time texturing benefit for one fixed geometry per family.

    For each bar, improvement is ``100 * (n_textured - n_untextured) /
    n_untextured`` at the same simulation time.  The representative geometry
    is chosen once at t=500 s and is never switched at later checkpoints.
    """
    setup_style(); os.makedirs(out_dir, exist_ok=True)
    baseline = _default_baseline() if baseline is None else baseline
    representatives = representative_configurations(master, families) if representatives is None else representatives
    flat_points = baseline["points"]
    x = np.arange(len(times))
    bar_width = 0.11
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    used = {}

    for index, family in enumerate(families):
        if family not in representatives or family not in family_entries:
            continue
        R, H = representatives[family]
        series = family_entries[family]["series"]
        key = min(series, key=lambda candidate: abs(candidate[0] - R) + abs(candidate[1] - H))
        textured_points = series[key]["points"]
        flat = np.asarray([interp_at(flat_points, time) for time in times])
        textured = np.asarray([interp_at(textured_points, time) for time in times])
        if np.any(flat <= 0):
            raise ValueError("untextured baseline has a non-positive value at a Fig. 4.4 checkpoint.")
        improvement = 100 * (textured - flat) / flat
        offset = (index - (len(families) - 1) / 2) * bar_width
        ax.bar(x + offset, improvement, width=bar_width, color=FAMILY_COLORS[family],
               label=FAMILY_LABELS.get(family, family), alpha=0.9)
        used[family] = key

    ax.axhline(0, color="black", linestyle="--", linewidth=1, label="untextured reference")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{time:,} s" for time in times])
    ax.set_xlabel("Time")
    ax.set_ylabel("Relative uptake vs. untextured (%)")
    ax.set_title("Texturing benefit relative to the untextured film\n(one fixed representative geometry per family)")
    ax.legend(fontsize=7.5, loc="best", ncol=2)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig4_4_relative_vs_untextured.png")
    fig.savefig(path, dpi=200); plt.close(fig)
    choices = ", ".join(f"{family}=(R={R * 1e6:g}, H={H * 1e6:g}) µm" for family, (R, H) in used.items())
    print(f"[fig4.4] saved -> {path}  (fixed representatives: {choices})")
    return path, used


def load_current_data(data_dir=DATA_DIR):
    """Build the seven-family current master table directly from raw exports."""
    files = {
        "cone_recessed": ("cone-recessed", "removed"), "cylinder_recessed": ("cylinder-recessed", "removed"),
        "square_recessed": ("square-recessed", "removed"), "grooves": ("grooves", "removed"),
        "edges": ("edges", "removed"), "square_pillar": ("square-pillars", "added"),
        "cylindrical_pillars": ("cylindrical-pillars", "added"),
    }
    entries = {}
    for family, (stem, volume_type) in files.items():
        with open(os.path.join(data_dir, f"{stem}-uptake.txt"), encoding="utf-8") as uptake, \
             open(os.path.join(data_dir, f"{stem}-volume.csv"), encoding="utf-8") as volume, \
             open(os.path.join(data_dir, f"{stem}-area.csv"), encoding="utf-8") as area:
            entries[family] = build_family_entry(uptake.read(), volume.read(), area.read(), volume_type=volume_type)
    return build_master_table(entries), entries, _default_baseline()


def generate_all(master, out_dir="section4_figures", baseline=None, family_entries=None):
    """Generate Figures 4.1–4.4 from a current master and flat-film baseline."""
    baseline = _default_baseline() if baseline is None else baseline
    paths = [fig4_1_tradeoff(master, out_dir, baseline), fig4_2_neq_per_v_all(master, out_dir, baseline),
             fig4_3_sav_early_all(master, out_dir, baseline)]
    if family_entries is None:
        # Preserve the convenient former generate_all(master) call while
        # obtaining the raw trajectories needed by the fixed-time bars.
        _, family_entries, _ = load_current_data()
    path, _ = fig4_4_fixed_time_relative_vs_untextured(master, family_entries, out_dir, baseline)
    return paths + [path]


if __name__ == "__main__":
    master, family_entries, baseline = load_current_data()
    check_tradeoff(master)
    generate_all(master, baseline=baseline, family_entries=family_entries)
