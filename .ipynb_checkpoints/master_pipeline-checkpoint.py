"""
master_pipeline.py

Generalizes the existing single-family (cone) analysis to compare across all
microtexture families (cones, grooves, edges, recessed edges, clustered
pillars, ...), using geometry-agnostic descriptors and a leadership-duration
score instead of a single fixed-time snapshot.

WHAT YOU NEED TO FILL IN (marked TODO below):
  - L_UNIT / H_F: the baseline unit-cell length / film thickness used to
    normalize R_texture, H_texture. If these differ by family, put them in
    FAMILY_UNIT_CONSTANTS instead of a single global.
  - FAMILIES dict: one entry per microtexture family, pointing at that
    family's already-parsed series / vol_map / area_map (whatever you
    currently have in the notebook), plus the names of its geometry
    parameters (R,H for cones; width,depth for grooves; etc).

Everything else (bug fixes, leadership duration, master table assembly)
should run as-is once FAMILIES is populated.
"""

import numpy as np
import pandas as pd

from viz_utils import normalize_label  # shared with plots.py and leadership_timeline_report

# ---------------------------------------------------------------------------
# Reference times — anchored to the untextured baseline's own kinetics,
# not arbitrary. Keep all three in the table; none of them is "the" metric.
# ---------------------------------------------------------------------------
REFERENCE_TIMES = {
    "t2000": 2000,        # ~20% baseline saturation (mesh independence anchor)
    "t15000": 15000,      # ~baseline half-saturation
    "t134900": 134900,    # ~baseline near-equilibrium
}

# Window over which "early-time leadership" is scored. Adjust if your
# flipping behavior extends beyond 50k s.
EARLY_WINDOW = (1, 50_000)

# TODO: fill in. If L_unit / H_f are the same across all families, one
# constant each is enough. If they differ by family, replace with a dict
# keyed by family name.
L_UNIT = None   # e.g. 100e-6
H_F = None      # e.g. 70e-6


# ---------------------------------------------------------------------------
# Bug fixes from the existing notebook code
# ---------------------------------------------------------------------------

# NOTE: normalize_label is now imported from viz_utils (NFKC-based, so it
# handles µ/μ and similar issues generally rather than by hand-typed swap).
#
# The old get_color() (color-by-R-rank, parsed via regex from an
# "R=... H=..." label) has been removed: it's not called anywhere in this
# file or plots.py, and it assumed absolute-µm-formatted labels, which
# conflicts with this session's switch to R_texture/H_texture as fractions
# of L_unit/H_f. For per-geometry coloring, use viz_utils.get_color()
# directly, keyed by (family_name, key) -- see plots.py.


# ---------------------------------------------------------------------------
# Leadership duration — replaces "who wins at exactly t=15000" with
# "who leads for how long, across the whole early window"
# ---------------------------------------------------------------------------

def interp_at(points, t_query):
    if not points:
        return None
    ts = [p[0] for p in points]
    ns = [p[1] for p in points]
    return float(np.interp(t_query, ts, ns))


def leadership_segments(series, t_start=EARLY_WINDOW[0], t_end=EARLY_WINDOW[1], n_steps=400):
    """Returns list of (start, end, key) segments — same logic as
    leadership_timeline_report, but returns raw segments (keys, not
    display labels) so they can be aggregated into scores."""
    time_grid = np.logspace(np.log10(t_start), np.log10(t_end), n_steps)
    leaders = []
    for t in time_grid:
        best_key, best_val = None, -np.inf
        for key, g in series.items():
            pts = sorted(g["points"], key=lambda p: p[0])
            if not pts or t < pts[0][0]:
                continue
            val = np.interp(t, [p[0] for p in pts], [p[1] for p in pts])
            if val > best_val:
                best_val, best_key = val, key
        leaders.append(best_key)

    segments = []
    seg_start = time_grid[0]
    current = leaders[0]
    for t, key in zip(time_grid[1:], leaders[1:]):
        if key != current:
            segments.append((seg_start, t, current))
            seg_start, current = t, key
    segments.append((seg_start, time_grid[-1], current))
    return segments


def leadership_duration_scores(series, t_start=EARLY_WINDOW[0], t_end=EARLY_WINDOW[1]):
    """Score per geometry key = share of log-time spent leading, in [0,1].
    Log-time weighting matches how you actually look at these curves
    (log-scaled x-axis), so a geometry that leads from 1s-10s counts the
    same as one leading from 1000s-10000s, rather than being swamped by
    the linear-time size of later segments."""
    segments = leadership_segments(series, t_start, t_end)
    log_total = np.log10(t_end) - np.log10(t_start)
    scores = {}
    for start, end, key in segments:
        if key is None:
            continue
        share = (np.log10(end) - np.log10(start)) / log_total
        scores[key] = scores.get(key, 0.0) + share
    return scores


# ---------------------------------------------------------------------------
# Universal geometric descriptors
# ---------------------------------------------------------------------------

def add_universal_descriptors(df):
    df["SA_V"] = df["SA"] / df["V"]
    df["L_c"] = df["V"] / df["SA"]  # characteristic diffusion length
    return df


# ---------------------------------------------------------------------------
# Master table assembly across families
# ---------------------------------------------------------------------------

# TODO: populate this with your actual per-family data structures.
# param_names should list the geometry parameters that key each family's
# series/vol_map/area_map dicts, in the same order as the dict keys, e.g.
# cones are keyed by (R, H) -> param_names=("R", "H").
FAMILIES = {
    # "cone": {
    #     "series": cone_series, "vol_map": cone_vol_map, "area_map": cone_area_map,
    #     "param_names": ("R", "H"),
    # },
    # "groove": {
    #     "series": groove_series, "vol_map": groove_vol_map, "area_map": groove_area_map,
    #     "param_names": ("width", "depth"),
    # },
    # ... add the remaining families the same way
}


def _round_key(key, ndigits=9):
    """Round float(s) in a dict key so lookups across series/vol_map/area_map
    don't silently miss due to float-precision differences between whatever
    code paths generated each dict (e.g. COMSOL export vs. a hand-built
    dict). ndigits=9 is generous -- tight enough to not merge genuinely
    different geometries, loose enough to absorb float noise."""
    if isinstance(key, tuple):
        return tuple(round(k, ndigits) if isinstance(k, float) else k for k in key)
    return round(key, ndigits) if isinstance(key, float) else key


def build_master_table(families=FAMILIES):
    rows = []
    for family_name, fam in families.items():
        series = fam["series"]
        vol_map = {_round_key(k): v for k, v in fam["vol_map"].items()}
        area_map = {_round_key(k): v for k, v in fam["area_map"].items()}
        param_names = fam["param_names"]
        duration_scores = leadership_duration_scores(series)

        for raw_key, g in series.items():
            key = _round_key(raw_key)
            row = {"family": family_name}
            # key is assumed to be a tuple matching param_names, e.g. (R, H)
            if isinstance(key, tuple):
                for name, val in zip(param_names, key):
                    row[name] = val
            else:
                row[param_names[0]] = key

            row["SA"] = area_map.get(key)
            row["V"] = vol_map.get(key)

            # flag (rather than silently pass through) a lookup miss, since
            # a missing SA/V quietly turns into NaN descriptors downstream
            # with no indication of *why* that geometry vanished from a plot
            if row["SA"] is None or row["V"] is None:
                print(f"[warning] no SA/V match for {family_name} key={key} "
                      f"(raw key={raw_key}) -- check area_map/vol_map keys "
                      f"and rounding")

            pts = sorted(g["points"], key=lambda p: p[0])
            for label, t in REFERENCE_TIMES.items():
                row[f"n_{label}"] = interp_at(pts, t)

            steady_state = pts[-1][1] if pts else None
            row["n_eq"] = steady_state
            # use `is not None` rather than bare truthiness, so a
            # legitimately-zero steady_state or n_t15000 isn't treated the
            # same as "missing data"
            if steady_state is not None and row["n_t15000"] is not None:
                row["pct_eq_t15000"] = (row["n_t15000"] / steady_state * 100) if steady_state != 0 else None
            else:
                row["pct_eq_t15000"] = None

            # duration_scores is keyed by the raw series key, not the
            # rounded one -- look up with raw_key to match
            row["early_leadership_share"] = duration_scores.get(raw_key, 0.0)

            rows.append(row)

    df = pd.DataFrame(rows)
    df = add_universal_descriptors(df)
    return df


if __name__ == "__main__":
    if not FAMILIES:
        print("FAMILIES is empty — populate it with your actual per-family "
              "series/vol_map/area_map before running.")
    else:
        master = build_master_table()
        master.to_csv("master_table_all_families.csv", index=False)
        print(master)
