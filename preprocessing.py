"""
preprocessing.py

Your existing COMSOL parsing/formatting code, extended with:
  - parse_scalar_csv(): a generalization of parse_volume_csv() so the same
    logic can pull out ANY named scalar column (volume OR area), instead
    of only "V_unit". parse_volume_csv() is kept as a thin wrapper around
    it so anything already calling parse_volume_csv() keeps working
    unchanged.
  - build_family_entry(): assembles series / vol_map / area_map /
    param_names into exactly the shape master_pipeline.py's FAMILIES dict
    expects for one family, so you're not hand-building that dict by hand
    for each of the 6 families.

Everything above the "---- NEW ----" marker is your existing code,
unmodified except for one line in geo_label (routed through
viz_utils.normalize_label -- see note there).
"""

import re
import csv
import numpy as np

from viz_utils import normalize_label

PRESET_TIMES = (2000, 15000, 134900)  # assumed defined elsewhere in your
                                       # codebase already (matches
                                       # REFERENCE_TIMES) -- included here
                                       # as a default so this file runs
                                       # standalone; delete if you already
                                       # import it from master_pipeline.py

# ---------------------------------------------------------------------------
# Geometry-data cleaning configuration
# ---------------------------------------------------------------------------
# Edit these two lists when you change the COMSOL sweeps.  They define the
# physical meaning of the two geometry axes, independently of a particular
# export's header order (or a mistaken header/value association).
RADIUS_SWEEP_UM = (5, 10, 20, 50, 80)
HEIGHT_SWEEP_UM = (5, 12.5, 20, 27.5, 35)
GEOMETRY_MATCH_RTOL = 1e-9


# ---- Formatters (mirror fmtTime/fmtMol/fmtVol/fmtArea/fmtR/fmtH) ----
def fmt_time(t):
    return f"{t/1000:.0f}k s" if t >= 1000 and t % 1000 == 0 else (
        f"{t/1000:.1f}k s" if t >= 1000 else f"{t:g} s")
def fmt_mol(v):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2e} mol"
def fmt_vol(v):
    return "—" if v is None else f"{v*1e13:.2f} ×10⁻¹³ m³"
def fmt_area(v):
    return "—" if v is None else f"{v*1e8:.3f} ×10⁻⁸ m²"
def fmt_r(r): return f"{r*1e6:.1f} µm"
def fmt_h(h): return f"{h*1e6:.0f} µm"

# ---- Parsers (ports of parseNumericCell / findColumnIndex / parseCOMSOLTxt / parseVolumeCsv) ----
def _to_metres(values_um):
    return tuple(value * 1e-6 for value in values_um)


def _belongs_to_grid(value, grid):
    return any(np.isclose(value, expected, rtol=GEOMETRY_MATCH_RTOL, atol=0.0)
               for expected in grid)


def canonicalize_geometry_orientation(rows):
    """Return rows with physical ``(R, H)`` ordering inferred from sweep grids.

    ``rows`` must contain dictionaries with ``R`` and ``H`` keys.  Their names
    come from the export headers, but their values are validated against the
    editable physical radius/height sweeps above.  A file whose two parameter
    columns are reversed is corrected once here; ambiguous or invalid files
    fail loudly instead of silently producing mislabeled plots.
    """
    if not rows:
        return rows

    radius_grid = _to_metres(RADIUS_SWEEP_UM)
    height_grid = _to_metres(HEIGHT_SWEEP_UM)
    first_values = {row["R"] for row in rows}
    second_values = {row["H"] for row in rows}

    direct = (all(_belongs_to_grid(value, radius_grid) for value in first_values)
              and all(_belongs_to_grid(value, height_grid) for value in second_values))
    reversed_axes = (all(_belongs_to_grid(value, height_grid) for value in first_values)
                     and all(_belongs_to_grid(value, radius_grid) for value in second_values))

    if direct and not reversed_axes:
        return rows
    if reversed_axes and not direct:
        return [{**row, "R": row["H"], "H": row["R"]} for row in rows]

    raise ValueError(
        "Cannot determine the physical R/H orientation from the configured "
        "sweep grids. Update RADIUS_SWEEP_UM and HEIGHT_SWEEP_UM if the "
        f"sweep changed. Observed header-R values: {sorted(first_values)}; "
        f"header-H values: {sorted(second_values)}."
    )
def parse_numeric_cell(value):
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned or cleaned.lower() in ("-", "--", "null", "nan", "none"):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None
def find_column_index(headers, candidates):
    if not headers:
        return -1
    norm = lambda s: re.sub(r"[^a-z0-9]+", "_", str(s).lower())
    normalized_headers = [norm(h) for h in headers]
    # Pass 1: exact match only (handles short candidates like "t", "R", "H" safely)
    for candidate in candidates:
        nc = norm(candidate)
        for i, h in enumerate(normalized_headers):
            if h == nc:
                return i
    # Pass 2: substring match, but only for candidates long enough to be unambiguous
    for candidate in candidates:
        nc = norm(candidate)
        if len(nc) < 3:
            continue
        for i, h in enumerate(normalized_headers):
            if nc in h or h in nc:
                return i
    return -1
def parse_comsol_txt(text: str):
    """Whitespace-delimited COMSOL .txt export -> list of (R, H, t, value) tuples.

    swap_rh: set True if this export's R_texture/H_texture columns are
             confirmed to hold the opposite physical parameters from
             what they should (e.g. R sweeping a range that's actually
             H's range, and vice versa) -- a sweep-setup mismatch in the
             COMSOL model, corrected here explicitly after parsing using
             the header's own column labels, not a silent guess.
    """
    rows = []
    headers = None
    for raw_line in text.replace("\r", "").split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("%"):
            header_text = line[1:].strip()
            if re.search(r"(R_texture|H_texture|H_cone|Total moles|intopA|intopV|Volume|value)",
                         header_text, re.I):
                headers = [h.strip() for h in re.split(r"\s{2,}", header_text) if h.strip()]
            continue
        vals = [parse_numeric_cell(v) for v in line.split()]
        if len(vals) < 4:
            continue
        r_idx = find_column_index(headers, ["R_texture", "R"])
        h_idx = find_column_index(headers, ["H_texture", "H_cone", "H"])
        t_idx = find_column_index(headers, ["t", "time"])
        v_idx = find_column_index(headers, ["moles", "intopA", "intopV", "Volume", "value"])
        R = H = t = value = None
        if min(r_idx, h_idx, t_idx, v_idx) >= 0:
            R, H, t, value = vals[r_idx], vals[h_idx], vals[t_idx], vals[v_idx]
        else:
            # fallback: simple 4-column cone-style layout (H, R, t, value)
            if all(v is not None for v in vals[:4]):
                if vals[0] < 1e-4 and vals[1] < 1e-4 and vals[2] >= 0:
                    H, R, t, value = vals[0], vals[1], vals[2], vals[3]
        if None not in (R, H, t, value):
            rows.append((R, H, t, value))
    cleaned = canonicalize_geometry_orientation([
        {"R": R, "H": H, "t": t, "value": value}
        for R, H, t, value in rows
    ])
    return [(row["R"], row["H"], row["t"], row["value"]) for row in cleaned]


# ---- NEW: generalized scalar-CSV parser (volume, area, or any future scalar) ----
def parse_scalar_csv(text: str, value_candidates, out_key):
    """
    Generalization of parse_volume_csv(): pulls out ANY named scalar
    column (e.g. "Volume"/"intopV" for volume, "intopA"/"Area" for area)
    instead of assuming the column is always volume. Returns a list of
    dicts {"R":..., "H":..., out_key:...}, same shape family as your
    existing parse_volume_csv output, just with a caller-chosen key name.

    swap_rh: set True if this export's R_texture/H_texture columns are
             confirmed (via the file's own % header + a values check, not
             assumed) to represent the opposite physical parameters from
             your uptake export -- i.e. a mismatch between separate
             COMSOL parametric-sweep tables, not a parsing error. This
             swaps R and H AFTER they're read using the header's own
             column labels, so it's a deliberate correction of a known
             sweep-setup inconsistency, not a silent guess.
    """
    entries = []
    headers = None
    for raw_line in text.replace("\r", "").split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("%"):
            header_text = line[1:].strip()
            if re.search(r"(R_texture|H_texture|H_cone|Volume|intopV|intopA|Area)", header_text, re.I):
                headers = next(csv.reader([header_text]))
            continue
        row = next(csv.reader([line]))
        vals = [parse_numeric_cell(v) for v in row]
        if len(vals) < 4:
            continue
        r_idx = find_column_index(headers, ["R_texture", "R"])
        h_idx = find_column_index(headers, ["H_texture", "H_cone", "H"])
        v_idx = find_column_index(headers, value_candidates)
        R = vals[r_idx] if r_idx >= 0 else vals[0]
        H = vals[h_idx] if h_idx >= 0 else vals[1]
        val = vals[v_idx] if v_idx >= 0 else vals[3]
        if None not in (R, H, val):
            entries.append({"R": R, "H": H, out_key: val})
    return canonicalize_geometry_orientation(entries)


def parse_volume_csv(text: str):
    """Pivoted volume .csv export -> list of {R, H, V_unit} dicts.
    Now a thin wrapper around parse_scalar_csv() -- unchanged output,
    so anything already calling this keeps working as-is."""
    return parse_scalar_csv(text, ["Volume", "intopV", "value"], "V_unit")


def parse_area_csv(text: str):
    """Pivoted area .csv export -> list of {R, H, intopA} dicts.
    Same file format as parse_volume_csv, just pointed at the area column."""
    return parse_scalar_csv(text, ["intopA", "Area", "value"], "intopA")


# ---- Geometry keys / interpolation ----
def geo_key(R, H): return (R, H)
def geo_label(R, H):
    # routed through normalize_label so every label is in canonical
    # NFKC form (µ -> μ, etc.) the moment it's created -- this is the
    # earliest possible point to apply the fix, before the label ever
    # reaches series/master_pipeline/any plot
    return normalize_label(f"R={fmt_r(R)} H={fmt_h(H)}")
def interp_at(points, t_query):
    """points: list of (t, n) tuples, sorted by t."""
    if not points:
        return None
    ts = [p[0] for p in points]
    ns = [p[1] for p in points]
    return float(np.interp(t_query, ts, ns))

# ---- Build series from raw uptake rows ----
def build_series(uptake_rows):
    series = {}
    for R, H, t, n in uptake_rows:
        key = geo_key(R, H)
        series.setdefault(key, {"R": R, "H": H, "label": geo_label(R, H), "points": []})
        series[key]["points"].append((t, n))
    for g in series.values():
        g["points"].sort(key=lambda p: p[0])
    all_r = sorted({g["R"] for g in series.values()})
    all_h = sorted({g["H"] for g in series.values()})
    return series, all_r, all_h


# ---- NEW: assemble one FAMILIES[family_name] entry ----
def build_family_entry(uptake_txt, volume_csv_txt=None, area_csv_txt=None,
                        param_names=("R", "H")):
    """
    Takes the same raw text your existing parsers already accept (COMSOL
    .txt uptake export, plus optional volume/area .csv exports) and
    returns exactly the dict shape master_pipeline.py's FAMILIES expects:

        {"series": ..., "vol_map": ..., "area_map": ..., "param_names": ...}

    Exports are cleaned into the physical R/H coordinate system using the
    editable RADIUS_SWEEP_UM and HEIGHT_SWEEP_UM constants above.
    """
    uptake_rows = parse_comsol_txt(uptake_txt)
    series, all_r, all_h = build_series(uptake_rows)

    vol_map = {}
    if volume_csv_txt:
        volume_entries = parse_volume_csv(volume_csv_txt)
        vol_map = {geo_key(entry["R"], entry["H"]): entry["V_unit"]
                   for entry in volume_entries}

    area_map = {}
    if area_csv_txt:
        area_entries = parse_area_csv(area_csv_txt)
        area_map = {geo_key(entry["R"], entry["H"]): entry["intopA"]
                    for entry in area_entries}

    return {
        "series": series,
        "vol_map": vol_map,
        "area_map": area_map,
        "param_names": param_names,
    }


# ---- Ranking + crossover (same logic as `ranking` / `presetRankings` / `rankingStable`) ----
def rank_at(series, vol_map, area_map, t_process):
    rows = []
    for key, g in series.items():
        n_t = interp_at(g["points"], t_process)
        n_eq = g["points"][-1][1] if g["points"] else None
        rows.append({
            "key": key, "label": g["label"], "R": g["R"], "H": g["H"],
            "n_t": n_t, "n_eq": n_eq,
            "V_unit": vol_map.get(key), "intopA": area_map.get(key),
            "pct": (n_t / n_eq * 100) if (n_t and n_eq) else None,
        })
    rows.sort(key=lambda r: r["n_t"] if r["n_t"] is not None else -np.inf, reverse=True)
    return rows
def check_stability(series, preset_times=PRESET_TIMES):
    orders = {}
    for t in preset_times:
        ranked = sorted(series.keys(),
                         key=lambda k: interp_at(series[k]["points"], t) or -np.inf,
                         reverse=True)
        orders[t] = ranked
    stable = len(set(tuple(o) for o in orders.values())) == 1
    return stable, orders


# ---- Example: assembling FAMILIES for master_pipeline.py ----
# from master_pipeline import FAMILIES
#
# FAMILIES["cone"] = build_family_entry(
#     uptake_txt=open("data/cone_uptake.txt").read(),
#     volume_csv_txt=open("data/cone_volume.csv").read(),
#     area_csv_txt=open("data/cone_area.csv").read(),
# )
# FAMILIES["groove"] = build_family_entry(
#     uptake_txt=open("data/groove_uptake.txt").read(),
#     volume_csv_txt=open("data/groove_volume.csv").read(),
#     area_csv_txt=open("data/groove_area.csv").read(),
#     param_names=("width", "depth"),  # if groove's native params differ from R/H
# )
