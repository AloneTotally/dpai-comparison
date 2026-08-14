"""Report figure: how far beyond the 2,000 s kinetic-comparison anchor each
recess configuration remains in the fitted k*sqrt(t) regime.
"""

from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import early_regime_fit as erf
import preprocessing as pp


ANCHOR_S = 2_000
FAMILY_FILES = {
    "cylinder_recessed": "cylinder-recessed",
    "square_recessed": "square-recessed",
    "cone_recessed": "cone-recessed",
    "square_pillar": "square-pillars",
    "edges": "edges",
    "cylindrical_pillars": "cylindrical-pillars",
    "grooves": "grooves",
}


def load_recess_curves(data_dir=Path("data")):
    """Return the three recess families in early_regime_fit's long format."""
    frames = []
    for family, stem in FAMILY_FILES.items():
        entry = pp.build_family_entry(
            (data_dir / f"{stem}-uptake.txt").read_text()
        )
        named_series = {(family, key): value for key, value in entry["series"].items()}
        frames.append(erf.nested_dict_to_long_df(named_series, geometry_name=family))
    return pd.concat(frames, ignore_index=True)


def build_figure():
    curves = load_recess_curves()
    summary = erf.compute_regime_summary(curves, early_window_t=ANCHOR_S)
    summary["R_um"] = summary["R"] * 1e6
    summary["H_um"] = summary["H"] * 1e6
    summary["safety_factor"] = summary["t_cross"] / ANCHOR_S

    fig, axes = plt.subplots(2, 4, figsize=(13.5, 7.2), sharey=True,
                             layout="constrained")
    axes = axes.ravel()
    cmap = plt.get_cmap("YlGn").copy()
    norm = colors.LogNorm(vmin=.05, vmax=20)
    image = None

    for ax, family in zip(axes, FAMILY_FILES):
        subset = summary[summary["geometry"] == family]
        table = subset.pivot(index="H_um", columns="R_um", values="safety_factor")
        table = table.sort_index(ascending=False)
        image = ax.imshow(table.to_numpy(), cmap=cmap, norm=norm, aspect="auto")

        ax.set_title(family.replace("_", " ").title(), fontsize=10)
        ax.set_xticks(range(len(table.columns)), [f"{r:g}" for r in table.columns])
        ax.set_yticks(range(len(table.index)), [f"{h:g}" for h in table.index])
        ax.set_xlabel("R (μm)")

        for row, h in enumerate(table.index):
            for col, r in enumerate(table.columns):
                factor = table.loc[h, r]
                cross_ks = factor * ANCHOR_S / 1000
                text_color = "white" if factor >= 4 else "#17320f"
                ax.text(col, row, f"{cross_ks:.0f}k\n{factor:.1f}×",
                        ha="center", va="center", fontsize=7.5, color=text_color)

        # A green outline explicitly denotes a configuration safe at the anchor.
        for row in range(len(table.index)):
            for col in range(len(table.columns)):
                is_exception = factor < 1
                ax.add_patch(plt.Rectangle(
                    (col - .5, row - .5), 1, 1, fill=False,
                    edgecolor="#b2182b" if is_exception else "white",
                    linewidth=2.2 if is_exception else .8,
                ))

    for ax in axes[:4]:
        ax.set_xlabel("R (μm)")
    axes[0].set_ylabel("H (μm)")
    axes[4].set_ylabel("H (μm)")
    axes[-1].axis("off")
    cbar = fig.colorbar(image, ax=axes, shrink=.84, pad=.035,
                        ticks=[.05, .1, .25, .5, 1, 2, 5, 10, 20])
    cbar.ax.set_yticklabels(["0.05×", "0.1×", "0.25×", "0.5×", "1×", "2×", "5×", "10×", "20×"])
    cbar.set_label(r"Safety factor  $t_{cross} / 2{,}000\,s$")
    fig.suptitle(r"Crossover safety at the 2,000 s kinetics-comparison anchor", y=1.02)
    fig.text(.5, -.02,
             "Each cell: crossover time (top) and safety factor (bottom). "
             "Red borders mark configurations with t_cross < 2,000 s.",
             ha="center", fontsize=9)
    return fig


if __name__ == "__main__":
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)
    figure = build_figure()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"crossover_safety_all_families.{suffix}",
                       dpi=300, bbox_inches="tight")
