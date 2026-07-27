"""
single_family_timeline.py

The per-family "leadership timeline" figure (which geometry within one
family wins at each moment). This is the version used before
master_pipeline.py existed; updated here to import normalize_label/
get_color from viz_utils instead of keeping its own separate copy, so
colors stay consistent with anything else that imports viz_utils
(master_pipeline.py, plots.py), and there's exactly one place the
µ/μ fix lives.
"""

import numpy as np
import matplotlib.pyplot as plt

from viz_utils import normalize_label, get_color

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


def leadership_timeline_report(series, t_start=1, t_end=200000, n_steps=400,
                                family_name=None, fmt_time=None, palette=None):
    """
    palette: list of colors (e.g. R_COLORS). Required now that get_color
             lives in viz_utils and needs an explicit palette argument
             rather than assuming a global R_COLORS exists.
    """
    if palette is None:
        raise ValueError("Pass palette=R_COLORS (or your color list) explicitly.")

    if fmt_time is None:
        def fmt_time(t):
            if t >= 1000:
                return f"{t/1000:.1f}k s"
            return f"{t:.0f} s"

    t_end_padded = t_end * 1.25
    time_grid = np.logspace(np.log10(t_start), np.log10(t_end_padded), n_steps)

    leaders = []
    for t in time_grid:
        best_label, best_val = None, -np.inf
        for key, g in series.items():
            pts = sorted(g["points"], key=lambda p: p[0])
            ts = [p[0] for p in pts]
            ns = [p[1] for p in pts]
            if not ts or t < ts[0]:
                continue
            val = np.interp(t, ts, ns)
            if val > best_val:
                best_val, best_label = val, g["label"]
        leaders.append(normalize_label(best_label))

    segments = []
    seg_start = time_grid[0]
    current = leaders[0]
    for t, lab in zip(time_grid[1:], leaders[1:]):
        if lab != current:
            segments.append((seg_start, t, current))
            seg_start, current = t, lab
    segments.append((seg_start, time_grid[-1], current))

    # clip the last segment's reported end back to the real t_end --
    # time_grid[-1] is t_end_padded, which is not a real reporting value
    last_start, _, last_lab = segments[-1]
    segments[-1] = (last_start, t_end, last_lab)

    fig, (ax, ax_table) = plt.subplots(
        2, 1, figsize=(10, 3.8),
        gridspec_kw={"height_ratios": [2, 1.1], "hspace": 0.5}
    )

    for start, end, lab in segments:
        ax.barh(0, end - start, left=start, height=0.6,
                color=get_color(lab, palette), edgecolor="white", linewidth=0.5)

    for i, (start, end, lab) in enumerate(segments):
        mid = np.sqrt(start * end) if start > 0 else end / 2
        above = (i % 2 == 0)
        y_text = 0.5 if above else -0.5
        va = "bottom" if above else "top"
        ax.plot([mid, mid], [0.3 if above else -0.3, y_text * 0.8],
                color="gray", linewidth=0.6, zorder=1)
        lab_two_line = lab.replace(" H=", "\nH=") if " H=" in lab else lab
        ax.text(mid, y_text, lab_two_line, ha="center", va=va,
                 fontsize=9, linespacing=1.3)

    ax.set_xscale("log")
    ax.set_xlim(t_start, t_end)
    ax.set_ylim(-0.85, 0.85)
    ax.set_yticks([])
    ax.set_xlabel("t (s, log scale)")
    title = "Leading texture over time" if family_name is None else f"Leading {family_name} geometry by early-time uptake"
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax_table.set_xlim(0, 1)
    ax_table.set_ylim(0, 1)
    ax_table.axis("off")

    n = len(segments)
    col_w = 1.0 / n
    for i, (start, end, lab) in enumerate(segments):
        x0 = i * col_w
        ax_table.add_patch(plt.Rectangle((x0 + 0.02*col_w, 0.5), 0.2*col_w, 0.4,
                                          color=get_color(lab, palette), transform=ax_table.transAxes))
        ax_table.text(x0 + 0.3*col_w, 0.7, lab.replace(" H=", "\nH="),
                      fontsize=8, va="center", ha="left", linespacing=1.2,
                      transform=ax_table.transAxes)
        duration_str = f"{fmt_time(start)} \u2192 {fmt_time(end)}"
        ax_table.text(x0 + 0.02*col_w, 0.15, duration_str,
                      fontsize=8, va="center", ha="left",
                      transform=ax_table.transAxes)

    fig.tight_layout()
    return fig, (ax, ax_table), segments
