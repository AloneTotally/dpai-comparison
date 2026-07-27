"""
viz_utils.py

Shared utilities for all figure-generation code in this project:
  - leadership_timeline_report (single-family figures)
  - master_pipeline.py / plots.py (cross-family analysis)

These used to be defined separately in each file, which is how the µ/μ
Unicode fix regressed when master_pipeline.py was written -- its own
normalize_label() only did .strip(), silently dropping the actual fix.
Centralizing here means there's exactly one implementation to fix, once,
for every script in the project.
"""

import unicodedata


def normalize_label(s):
    """
    Normalize any string with Unicode NFKC normalization. This folds
    "looks the same, different codepoint" character pairs onto a single
    canonical form -- e.g. MICRO SIGN (µ, U+00B5) and GREEK SMALL LETTER MU
    (μ, U+03BC) both become U+03BC. This is general on purpose: it covers
    the µm issue without hand-typing which specific characters to swap,
    and it'll also catch any other similar-looking-character mismatch that
    shows up later (different unit symbols, full-width characters from a
    copy-paste, etc.) across ANY geometry family, not just cones.
    """
    if s is None:
        return s
    return unicodedata.normalize("NFKC", s).strip()


# Auto-assigning, persistent color registry. A key gets the next unused
# palette color the first time it's seen, and keeps it for the rest of the
# session/script -- across every figure that imports this module. No
# manual {label: color} dict to maintain by hand as new geometries or
# families get added.
_COLOR_REGISTRY = {}


def get_color(key, palette):
    """
    key: anything hashable identifying a distinct thing to color.
         - For single-family figures: a normalized geometry label string,
           e.g. "R=20.0 μm H=35 μm".
         - For cross-family figures: use a (family, geometry_key) tuple
           instead of a bare geometry_key -- different families can have
           colliding native keys (e.g. cone (20, 35) and a hypothetical
           groove (20, 35)), so the family must be part of the identity
           to avoid two unrelated geometries silently sharing a color.
    palette: list of colors to draw from (e.g. R_COLORS).
    """
    if isinstance(key, str):
        key = normalize_label(key)
    if key not in _COLOR_REGISTRY:
        _COLOR_REGISTRY[key] = palette[len(_COLOR_REGISTRY) % len(palette)]
    return _COLOR_REGISTRY[key]


def reset_color_registry():
    """Call this if you want a fresh color assignment (e.g. starting a
    completely new analysis session and don't want colors carried over
    from an earlier, unrelated set of figures)."""
    _COLOR_REGISTRY.clear()
