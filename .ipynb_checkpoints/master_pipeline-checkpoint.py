"""
master_pipeline.py

Generalizes the existing single-family (cone) analysis to compare across all
microtexture families (cones, grooves, edges, recessed edges, clustered
pillars, ...), using geometry-agnostic descriptors and a leadership-duration
score instead of a single fixed-time snapshot.

CHANGES IN THIS VERSION (from the previous draft):
  1. Cross-family leadership scoring bug fix: early_leadership_share is now
     computed ONCE across a combined series spanning every family, instead
     of once per family. Previously each family's geometries only ever
     competed against their own siblings, so every family could produce
     its own "winner" with an inflated share -- the ranked bar chart and
     SA/V scatter were then comparing numbers that were never actually on
     the same scale.
  2. REFERENCE_TIMES no longer includes t134900 (undocumented, inherited
     value). Near-equilibrium is now n_near_eq, a windowed average over
     NEAR_EQ_WINDOW = (160_000, 200_000) s -- see preprocessing.py's
     window_average() docstring for the empirical justification (checked
     against both the smallest and largest geometries in the sweep).
  3. EARLY_WINDOW upper bound corrected from 50,000 s to 10,000 s, to
     match the empirical t1/2 ~= 10,000 s (from the R=5um,H=5um curve)
     rather than the earlier hand-calculated 15,000 s estimate. This
     keeps "early leadership" inside the diffusion-controlled regime and
     out of the region where near-equilibrium volume-depletion effects
     (e.g. deep grooves) start to distort the score.
  4. Missing SA/V matches now hard-fail after collecting a full list,
     instead of printing a warning and silently continuing with NaN.
  5. New column: n_eq_per_V (near-equilibrium capacity density), to give
     the groove volume-depletion hypothesis a direct numeric test in the
     table itself.
  6. pct_eq_t15000 retained only as a diagnostic column -- do not use it
     as the y-axis for cross-family trade-off plots; use n_t15000 or
     n_near_eq instead (see report methodology note on why % of
     equilibrium is misleading for cross-family comparison).

STILL TODO (unchanged from before):
  - L_UNIT / H_F: fill in if fractional R/H normalization is needed.
  - FAMILIES dict: populate per family as before.
"""

import numpy as np
import pandas as pd

from viz_utils import normalize_label  # shared with plots.py and leadership_timeline_report
from preprocessing import window_average, NEAR_EQ_WINDOW  # shared definition, single source of truth

# ---------------------------------------------------------------------------
# Reference times
# ---------------------------------------------------------------------------
# Fixed-timestamp snapshots. t134900 removed -- see module docstring.
REFERENCE_TIMES = {
    "t500": 500,       # clean diffusion-kinetics checkpoint -- all families
                       # still at alpha ~= 0.50 in this window per Fig 2's
                       # fit; use for cross-family early comparisons instead
                       # of t2000 where regime-purity matters

    "t2000": 2000,     # mesh-independence validation anchor
    "t15000": 15000,   # ~baseline half-saturation (see note in preprocessing.py
                        # re: empirical t1/2 ~= 10,000 s -- update if/when this
                        # is confirmed across more than one geometry)
}

# Near-equilibrium reference is now a windowed average, not a fixed
# timestamp -- imported from preprocessing.py so there is exactly one
# definition of the window shared across the whole pipeline.
# NEAR_EQ_WINDOW = (160_000, 200_000)  # <- lives in preprocessing.py

# Window over which "early-time leadership" is scored. Bound corrected to
# match the empirical t1/2 (~10,000 s) rather than the earlier 50,000 s,
# which bled into the region where volume-depletion effects on low-capacity
# geometries (e.g. deep grooves) start to distort the score.
EARLY_WINDOW = (1, 10_000)

# TODO: fill in. If L_unit / H_f are the same across all families, one
# constant each is enough. If they differ by family, replace with a dict
# keyed by family name.
L_UNIT = None   # e.g. 100e-6
H_F = None      # e.g. 70e-6


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
    display labels) so they can be aggregated into scores.

    `series` can be a single family's series dict, OR a combined dict
    spanning multiple families (e.g. keyed by (family_name, R, H)) -- the
    function itself is family-agnostic; it just compares whatever keys
    it's given against each other. See build_master_table() for how the
    combined dict is built for cross-family scoring."""
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
    the linear-time size of later segments.

    IMPORTANT: pass a series dict that already spans everything you want
    these scores to be comparable against. If called once per family, the
    resulting scores are only comparable WITHIN that family -- see
    build_master_table() for the cross-family combined-series usage."""
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

# ---------------------------------------------------------------------------
# Unit conversion: mol -> kg CO2, to match outside literature
# ---------------------------------------------------------------------------
CO2_MOLAR_MASS_KG_PER_MOL = 44.01e-3  # 44.01 g/mol

def add_kg_columns(df):
    """Adds a '<col>_kg' version of every absolute molar-quantity column,
    alongside (not replacing) the mol original. Deliberately excludes
    pct_eq_t15000 (unitless ratio -- mol cancels, "converting" it is
    meaningless) and early_leadership_share (not a molar quantity).
    n_eq_per_V (mol/m^3) IS converted, since it's still an absolute molar
    density, just per-volume rather than per-geometry."""
    mol_cols = [c for c in df.columns
                if (c.startswith("n_") or c == "n_eq_per_V")
                and not c.endswith("_kg")]
    for col in mol_cols:
        df[f"{col}_kg"] = df[col] * CO2_MOLAR_MASS_KG_PER_MOL
    return df




def build_master_table(families=FAMILIES):
    # -----------------------------------------------------------------
    # Cross-family leadership scoring (fix #1): build ONE combined series
    # spanning every family, keyed by (family_name, raw_key) so identical
    # (R, H) pairs from different families don't collide. Scored once,
    # here, before the per-row loop -- NOT per family -- so every
    # geometry is genuinely competing against every other geometry in the
    # dataset for its leadership share, not just its own siblings.
    # -----------------------------------------------------------------
    combined_series = {}
    for family_name, fam in families.items():
        for raw_key, g in fam["series"].items():
            combined_series[(family_name, raw_key)] = g
    duration_scores = leadership_duration_scores(
        combined_series, t_start=EARLY_WINDOW[0], t_end=EARLY_WINDOW[1]
    )

    rows = []
    missing_sa_v = []  # collected, then hard-fail after the full pass (fix #4)

    for family_name, fam in families.items():
        series = fam["series"]
        vol_map = {_round_key(k): v for k, v in fam["vol_map"].items()}
        area_map = {_round_key(k): v for k, v in fam["area_map"].items()}
        param_names = fam["param_names"]
        # "removed" (subtractive: recesses/grooves) or "added" (additive:
        # pillars) -- see build_family_entry()'s docstring in preprocessing.py.
        # Defaults to "removed" for any FAMILIES entry built before this
        # field existed, so older data doesn't break.
        volume_type = fam.get("volume_type", "removed")

        for raw_key, g in series.items():
            key = _round_key(raw_key)
            row = {"family": family_name, "volume_type": volume_type}
            # key is assumed to be a tuple matching param_names, e.g. (R, H)
            if isinstance(key, tuple):
                for name, val in zip(param_names, key):
                    row[name] = val
            else:
                row[param_names[0]] = key

            row["SA"] = area_map.get(key)
            row["V"] = vol_map.get(key)

            if row["SA"] is None or row["V"] is None:
                missing_sa_v.append((family_name, key, raw_key))

            pts = sorted(g["points"], key=lambda p: p[0])
            for label, t in REFERENCE_TIMES.items():
                row[f"n_{label}"] = interp_at(pts, t)

            # --- Near-equilibrium reference (fix #2): windowed average,
            # not a fixed timestamp. See preprocessing.window_average()
            # and NEAR_EQ_WINDOW for the justification. Returns None if a
            # curve doesn't actually reach the window (shouldn't happen
            # here since every family runs to 200,000 s, but surfaced
            # rather than silently producing a wrong number if a family's
            # data ever changes).
            n_near_eq = window_average(pts, *NEAR_EQ_WINDOW)
            row["n_near_eq"] = n_near_eq
            row["n_eq"] = n_near_eq  # kept for backward compatibility with
                                      # any existing plot code referencing n_eq

            # --- New: near-equilibrium capacity density (fix #5). Direct
            # numeric test of the volume-depletion hypothesis -- a deep
            # groove with low n_eq but high n_eq_per_V is consistent with
            # "saturates a small remaining volume quickly", vs. a
            # geometry with both low n_eq AND low n_eq_per_V, which would
            # need a different explanation.
            if n_near_eq is not None and row["V"]:
                row["n_eq_per_V"] = n_near_eq / row["V"]
            else:
                row["n_eq_per_V"] = None

            # pct_eq_t15000: retained as a DIAGNOSTIC column only (fix #6).
            # Do not use as the axis for cross-family trade-off plots --
            # it normalizes away exactly the capacity differences those
            # plots are meant to reveal. Use n_t15000 / n_near_eq instead.
            if n_near_eq is not None and row["n_t15000"] is not None:
                row["pct_eq_t15000"] = (row["n_t15000"] / n_near_eq * 100) if n_near_eq != 0 else None
            else:
                row["pct_eq_t15000"] = None

            # duration_scores is keyed by (family_name, raw_key) now, to
            # match the combined_series construction above.
            row["early_leadership_share"] = duration_scores.get((family_name, raw_key), 0.0)

            rows.append(row)

    if missing_sa_v:
        print(f"[ERROR] {len(missing_sa_v)} row(s) missing SA/V match:")
        for family_name, key, raw_key in missing_sa_v:
            print(f"  family={family_name} key={key} (raw key={raw_key})")
        raise ValueError(
            f"{len(missing_sa_v)} row(s) missing SA/V data -- check "
            "vol_map/area_map keys for an R/H swap or a rounding mismatch "
            "before building the table. (Previously this only printed a "
            "warning and continued with NaN, which silently produced holes "
            "in downstream heatmaps -- see square_recessed frozen-geometry "
            "bug for why this is now a hard stop.)"
        )

    df = pd.DataFrame(rows)
    df = add_universal_descriptors(df)
    df = add_kg_columns(df)
    return df


if __name__ == "__main__":
    if not FAMILIES:
        print("FAMILIES is empty — populate it with your actual per-family "
              "series/vol_map/area_map before running.")
    else:
        master = build_master_table()
        master.to_csv("master_table_all_families.csv", index=False)
        print(master)
