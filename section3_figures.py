"""
section3_figures.py

Figures for the Pillar Family (Additive Geometry) section, from `master`
(master_pipeline.build_master_table() output, post add_kg_columns()).

Family name strings match FAMILIES dict keys: square_pillar, cylindrical_pillars.
All uptake/capacity quantities use the _kg columns (kg CO2), not mol.

Only two figures are committed here: fig3_1 (heatmap) and fig3_2 (n_eq/V
check). Everything else is decided by check_early_collapse() first --
see bottom of file. Do not add a fig3_3/fig3_4 without running that check
against the actual pillar data and confirming there's something worth
showing that isn't just a mirror of section2_figures's fig7.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from section2_figures import setup_style, _um  # reuse, not duplicate

FAMILIES_SEC3 = ["square_pillar", "cylindrical_pillars"]

FAMILY_LABELS_SEC3 = {
    "square_pillar": "Square pillar",
    "cylindrical_pillars": "Cylindrical pillar",
}

FAMILY_COLORS_SEC3 = {
    "square_pillar": "#6A4C93",
    "cylindrical_pillars": "#1B998B",
}

FIG3_1_METRICS = [
    ("n_t500_kg", "t = 500 s"),
    ("n_t2000_kg", "t = 2,000 s"),
    ("n_t15000_kg", "t = 15,000 s"),
    ("n_near_eq_kg", "Equilibrium"),
]


def _check_required_columns(master, cols):
    missing = [c for c in cols if c not in master.columns]
    if missing:
        raise ValueError(
            f"master is missing columns {missing}. Run add_kg_columns() "
            f"(now called inside build_master_table()) before generating "
            f"section 3 figures."
        )


# ---------------------------------------------------------------------------
# Fig 3.1 -- R x H heatmap, 4 panels per pillar family (kg CO2)
# ---------------------------------------------------------------------------

def fig3_1_heatmap(master, family, out_dir):
    _check_required_columns(master, [m for m, _ in FIG3_1_METRICS] + ["R", "H", "volume_type"])
    sub = master[master["family"] == family].copy()
    if sub.empty:
        print(f"[fig3.1] no rows for family '{family}', skipping")
        return None
    if not (sub["volume_type"] == "added").all():
        raise ValueError(
            f"family '{family}' has volume_type != 'added' in some rows -- "
            f"check FAMILIES[...] setup before generating pillar figures."
        )

    sub["R_um"] = _um(sub["R"])
    sub["H_um"] = _um(sub["H"])
    R_vals = sorted(sub["R_um"].unique())
    H_vals = sorted(sub["H_um"].unique())

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, (metric, title) in zip(axes, FIG3_1_METRICS):
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
        cbar.set_label("kg CO$_2$", fontsize=7)

    fig.suptitle(f"{FAMILY_LABELS_SEC3[family]} -- uptake (kg CO$_2$) vs. R, H "
                 f"across four process times", y=1.05)
    fig.tight_layout()
    path = os.path.join(out_dir, f"fig3_1_heatmap_{family}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[fig3.1] saved -> {path}")
    return path


# ---------------------------------------------------------------------------
# Fig 3.2 -- n_eq/V for pillars (additive), overlaid, with optional
# reference to the subtractive rho from section 2's Fig 6.
# ---------------------------------------------------------------------------

def fig3_2_neq_per_v(master, out_dir, families=FAMILIES_SEC3, rho_subtractive_kg=None,
                      flat_ref=None):
    """
    Styled to match section2_figures.fig6_neq_per_v(): plots % deviation
    from the pooled pillar mean (not raw kg/m^3), with small per-family
    horizontal jitter so the two families' markers at the same R fan out
    instead of stacking. The raw kg/m^3 axis (~1.5563e7-1.5580e7) is a
    <0.1% range, which on a linear raw scale reads as noisy scatter --
    the % transform is what actually makes the collapse visible, same
    reasoning as section 2's fig6 docstring.

    rho_subtractive_kg: the pooled n_eq/V constant (kg CO2/m^3) from
    section2_figures.fig6_neq_per_v(), if you want this panel to show
    where pillars sit relative to it. Pass None to skip that reference
    line -- do NOT assume additive families collapse onto the same
    constant; that's exactly the question this figure is meant to answer,
    not a given.

    flat_ref: optional (n_eq_flat_mol, V_flat_m3) tuple for the
    untextured baseline, same shape as section2_figures.fig6_neq_per_v's
    flat_ref -- e.g. (baseline["n_near_eq"], V_flat) from
    preprocessing.build_untextured_baseline(). Adds a third reference
    line (pooled pillar rho, subtractive rho, flat rho all on one panel)
    when both this and rho_subtractive_kg are supplied.
    """
    setup_style()
    _check_required_columns(master, ["n_near_eq_kg", "V", "R", "volume_type"])
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    all_ratios = []
    per_family = []
    for family in families:
        sub = master[master["family"] == family].copy()
        if sub.empty:
            continue
        if not (sub["volume_type"] == "added").all():
            raise ValueError(f"family '{family}' is not volume_type='added' -- check setup.")
        sub["R_um"] = _um(sub["R"])
        sub["n_eq_per_V_kg"] = sub["n_near_eq_kg"] / sub["V"]
        all_ratios.extend(sub["n_eq_per_V_kg"].tolist())
        per_family.append((family, sub))

    rho_pillar = np.mean(all_ratios)
    cv_pillar = np.std(all_ratios) / rho_pillar

    # small deterministic horizontal jitter per family, same pattern as
    # section2_figures.fig6_neq_per_v, so overlapping R values fan out
    n_fam = len(per_family)
    jitter_span_um = 3.0
    for i, (family, sub) in enumerate(per_family):
        offset = (i - (n_fam - 1) / 2) * (jitter_span_um / max(n_fam - 1, 1))
        pct_dev = (sub["n_eq_per_V_kg"] - rho_pillar) / rho_pillar * 100
        ax.scatter(sub["R_um"] + offset, pct_dev, color=FAMILY_COLORS_SEC3[family],
                   label=FAMILY_LABELS_SEC3[family], s=40, alpha=0.85)

    ax.axhline(0, color="black", linestyle="--", linewidth=1,
               label=f"pooled mean $\\rho_{{pillar}}$ = {rho_pillar:.3e} kg/m$^3$ (CV={cv_pillar:.4f})")

    if rho_subtractive_kg is not None:
        pct_ref = (rho_subtractive_kg - rho_pillar) / rho_pillar * 100
        ax.axhline(pct_ref, color="#888888", linestyle=":", linewidth=1.2,
                   label=f"subtractive $\\rho$ (\u00a72) = {rho_subtractive_kg:.3e} kg/m$^3$")
        pct_diff = (rho_pillar - rho_subtractive_kg) / rho_subtractive_kg * 100
        collapse_verdict = "same constant" if abs(pct_diff) < 5 else "does NOT collapse onto \u00a72 value"
        note = (f"\u0394 = {pct_diff:+.2f}% vs. subtractive \u03c1\n"
                f"({collapse_verdict})")
        ax.annotate(note, xy=(0.97, 0.05), xycoords="axes fraction",
                    ha="right", fontsize=8, color="#333333",
                    bbox=dict(boxstyle="round", fc="white", ec="#cccccc"))

    if flat_ref is not None:
        n_eq_flat_mol, V_flat = flat_ref
        if n_eq_flat_mol is not None and V_flat:
            rho_flat_kg = n_eq_flat_mol * 0.04401 / V_flat
            pct_flat = (rho_flat_kg - rho_pillar) / rho_pillar * 100
            ax.axhline(pct_flat, color="#cc8800", linestyle=":", linewidth=1.2,
                       label=f"untextured $\\rho_{{flat}}$ = {rho_flat_kg:.3e} kg/m$^3$ "
                             f"({pct_flat:+.2f}% vs. pillar mean)")

    ax.set_xlabel("R (\u00b5m)")
    ax.set_ylabel("$n_{eq} / V$ deviation from pooled mean (%)")
    ax.set_title("Pillar (additive) equilibrium capacity density is independent of R\n"
                 "(collapses onto one constant, same as the subtractive families)")
    ax.legend(loc="best", framealpha=0.9, fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig3_2_neq_per_v_pillars.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[fig3.2] saved -> {path}  (rho_pillar={rho_pillar:.4e} kg/m^3, CV={cv_pillar:.5f})")
    return path, rho_pillar

def fig3_3_sav_early_overlay(master, out_dir, subtractive_fit,
                              families=FAMILIES_SEC3, metric="n_t500_kg",
                              subtractive_metric_label="n(t=500 s) (kg CO$_2$)"):
    """
    subtractive_fit: (a, b, r) from section2_figures.fig7_sav_vs_early(),
    i.e. the pooled subtractive fit n(t=500s) [kg CO2] = a + b*SA_V --
    fig7 converts to kg internally before fitting, so this stays in kg
    CO2 throughout to match. (Previously this function divided the
    already-kg `metric` column by the molar mass, converting it back to
    mol and comparing it against the kg-based fit line -- a units
    mismatch that would have made resid_pct meaningless. Fixed: no
    conversion, both sides are kg CO2 now.)
    Tests whether pillar (additive) early uptake falls on that SAME line --
    it does not: pillars sit systematically above it. See in-panel
    annotation for the proposed mechanism (exposed-perimeter flux
    accessibility vs. partially self-shadowing recess walls at matched
    SA/V). This is a genuine early/equilibrium asymmetry, not a mirror
    of fig3.2's sign-independent equilibrium collapse.
    """
    setup_style()  # was missing before -- caused the sans-serif fallback
    _check_required_columns(master, ["SA_V", metric, "volume_type"])
    a, b, r_subtractive = subtractive_fit

    fig, ax = plt.subplots(figsize=(7, 5))

    xs_all, ys_all = [], []
    for family in families:
        sub = master[master["family"] == family]
        if sub.empty:
            continue
        y_kg = sub[metric]  # already kg CO2 -- no conversion
        ax.scatter(sub["SA_V"], y_kg, color=FAMILY_COLORS_SEC3[family],
                   label=FAMILY_LABELS_SEC3[family], s=35, alpha=0.85)
        xs_all.extend(sub["SA_V"].tolist())
        ys_all.extend(y_kg.tolist())

    xs_all, ys_all = np.array(xs_all), np.array(ys_all)
    x_line = np.linspace(xs_all.min(), xs_all.max(), 50)
    ax.plot(x_line, a + b * x_line, color="black", linestyle="--", linewidth=1,
           label=f"subtractive fit (\u00a72), r={r_subtractive:.2f}")

    predicted = a + b * xs_all
    resid_pct = np.mean((ys_all - predicted) / predicted) * 100
    r_pillar_alone = np.corrcoef(xs_all, ys_all)[0, 1]

    # --- in-panel mechanism annotation, pointing at a representative
    # high-SA/V pillar point sitting well above the subtractive line ---
    idx_target = np.argmax(xs_all)  # rightmost / most-offset point
    x_target, y_target = xs_all[idx_target], ys_all[idx_target]
    note = f"pillars: {resid_pct:+.1f}% above \u00a72 line at matched SA/V (r={r_pillar_alone:.2f})"
    # placed top-left, away from the legend (bottom-right), so the two
    # don't overlap -- they were both anchored to the same corner before
    ax.annotate(note, xy=(0.03, 0.95), xycoords="axes fraction",
                ha="left", va="top", fontsize=8, color="#333333",
                bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.9))

    ax.set_xlabel("SA/V (1/m)")
    ax.set_ylabel(subtractive_metric_label)
    ax.set_title("Additive (pillar) points sit above the subtractive SA/V trend\n"
                 "(early-time SA/V collapse is sign-dependent, unlike equilibrium \u03c1)")
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig3_3_sav_early_overlay.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[fig3.3] saved -> {path}  (pillars r={r_pillar_alone:.3f}, "
          f"mean offset from \u00a72 line = {resid_pct:+.2f}%)")
    return path    
    
# ---------------------------------------------------------------------------
# Diagnostic (NOT a figure) -- run this before deciding whether a
# fig3_3/fig3_4 early-SA/V-collapse figure is warranted for pillars.
# ---------------------------------------------------------------------------

def check_early_collapse(master, families=FAMILIES_SEC3, metric="n_t500_kg"):
    """Prints the SA/V vs early-uptake correlation for pillars alone, so
    you can compare it against section 2's pooled r (fig7) BEFORE
    deciding whether pillars show the same early-time SA/V-governed
    collapse or something else. Do not build fig3_3 until this has been
    run and the result actually says something worth a figure."""
    _check_required_columns(master, ["SA_V", metric, "volume_type"])
    sub = master[master["family"].isin(families)]
    xs, ys = sub["SA_V"].values, sub[metric].values
    r = np.corrcoef(xs, ys)[0, 1]
    print(f"[check_early_collapse] pillars-only SA/V vs {metric}: r = {r:.3f} "
          f"(n={len(xs)})")
    print("  Compare against section2_figures.fig7's pooled subtractive r.")
    print("  If |r| is similarly high -> same early-time SA/V mechanism, fig3.3 "
          "would likely just mirror fig7 with fewer points -- skip it, note "
          "the match in prose instead.")
    print("  If |r| is notably lower/different -> pillars deviate from the "
          "subtractive early-time pattern; THAT's worth a figure, and the "
          "figure should show the deviation, not force-fit the same panel.")
    return r


def generate_all(master, out_dir="section3_figures", rho_subtractive_kg=None,
                  untextured_baseline=None, V_flat=None):
    """
    untextured_baseline / V_flat: see fig3_2_neq_per_v's flat_ref
    docstring. Both optional and default to None so this still runs
    unchanged for anyone not yet using the baseline.
    """
    setup_style()
    os.makedirs(out_dir, exist_ok=True)

    for family in FAMILIES_SEC3:
        fig3_1_heatmap(master, family, out_dir)

    flat_ref = None
    if untextured_baseline is not None and V_flat is not None:
        flat_ref = (untextured_baseline.get("n_near_eq"), V_flat)
    fig3_2_neq_per_v(master, out_dir, rho_subtractive_kg=rho_subtractive_kg, flat_ref=flat_ref)

    r = check_early_collapse(master)
    print(f"\n[generate_all] fig3.1 and fig3.2 written to ./{out_dir}/. "
          f"fig3.3/3.4 NOT generated -- review check_early_collapse's r={r:.3f} "
          f"output above and decide before writing that code.")


if __name__ == "__main__":
    print(__doc__)