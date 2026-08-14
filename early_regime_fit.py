"""
early_regime_fit.py

Two-regime characterization of n(t) uptake curves:
  - Early regime (diffusion-controlled): n(t) ~ k * sqrt(t)
  - Late regime (capacity-controlled):    n(t) -> n_eq

Produces, per (geometry, R, H) config:
    k        : early-time sqrt(t) coefficient (slope through origin)
    t_cross  : first *persistent* post-fit deviation from the sqrt(t) fit
    n_eq     : equilibrium capacity (max/asymptotic n observed)
    r2_early : goodness of fit of the sqrt(t) model in the early window,
               for flagging configs where the collapse assumption fails
    q_ratio_anchor : n(t)/[k*sqrt(t)] at a chosen comparison time
    alpha_anchor   : local log-log uptake exponent at that comparison time

Expects a long-format DataFrame with columns:
    geometry, R, H, t, n
(matches the convention already used in master_pipeline.py / viz_utils.py)

Usage:
    from early_regime_fit import compute_regime_summary, plot_collapse_check

    summary = compute_regime_summary(df)
    summary.to_csv("regime_summary.csv", index=False)

    # sanity-check the sqrt(t) collapse before trusting k as a ranking metric
    plot_collapse_check(df, geometry="recess", R=20, H=35)
"""

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Core fitting
# ----------------------------------------------------------------------

def fit_sqrt_k(t, n, t_min=None, t_max=None, method="median_ratio"):
    """
    Fit n(t) ~= k * sqrt(t), forced through the origin, using data with
    t_min <= t <= t_max (when given). Returns (k, r2) where r2 is computed
    only over the fitted window.

    ``method="median_ratio"`` estimates k as median(n/sqrt(t)). It gives
    every early-time sample equal influence and is less likely than a
    full-window least-squares fit to let the later, gently bending part of a
    curve pull the reference slope down. ``method="least_squares"`` retains
    the previous through-origin least-squares calculation.
    """
    t = np.asarray(t, dtype=float)
    n = np.asarray(n, dtype=float)

    if t_min is not None or t_max is not None:
        mask = np.ones(len(t), dtype=bool)
        if t_min is not None:
            mask &= t >= t_min
        if t_max is not None:
            mask &= t <= t_max
        t, n = t[mask], n[mask]

    # drop t=0 (sqrt undefined issue is fine, sqrt(0)=0, but keep at least
    # a couple of nonzero points to fit)
    if len(t) < 3:
        return np.nan, np.nan

    x = np.sqrt(t)
    if not np.any(x > 0):
        return np.nan, np.nan
    if method == "median_ratio":
        valid = x > 0
        k = float(np.median(n[valid] / x[valid]))
    elif method == "least_squares":
        denom = np.sum(x * x)
        if denom == 0:
            return np.nan, np.nan
        k = np.sum(x * n) / denom
    else:
        raise ValueError("method must be 'median_ratio' or 'least_squares'")

    n_pred = k * x
    ss_res = np.sum((n - n_pred) ** 2)
    ss_tot = np.sum((n - np.mean(n)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return k, r2


def scaling_ratio(t, n, k):
    """Return q(t)/k = n(t)/(k*sqrt(t)); pure sqrt(t) uptake equals 1."""
    t = np.asarray(t, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return n / (k * np.sqrt(t))


def value_at_time(t, values, t_query):
    """Linearly interpolate a time-series value without extrapolation."""
    t = np.asarray(t, dtype=float)
    values = np.asarray(values, dtype=float)
    order = np.argsort(t)
    t, values = t[order], values[order]
    if t_query < t[0] or t_query > t[-1]:
        return np.nan
    return float(np.interp(t_query, t, values))


def local_log_slope(t, n, t_low, t_high):
    """Estimate d(log n)/d(log t) over a bracket around an anchor time.

    The diffusion-controlled prediction is 0.5. This is deliberately a
    local diagnostic: it says whether the data retain sqrt(t) scaling near
    the anchor, without assigning any curvature to a specific mechanism.
    """
    n_low = value_at_time(t, n, t_low)
    n_high = value_at_time(t, n, t_high)
    if not (np.isfinite(n_low) and np.isfinite(n_high) and n_low > 0 and n_high > 0):
        return np.nan
    return float(np.log(n_high / n_low) / np.log(t_high / t_low))


def find_persistent_deviation(t, n, k, t_start, tol=0.05, consecutive=3):
    """First sustained same-direction departure from k*sqrt(t).

    Only points strictly after ``t_start`` are eligible. A transient residual
    cannot be called a crossover: ``consecutive`` later samples must all be
    beyond ``tol`` with the same sign. Returns NaN if no such departure occurs.
    """
    t = np.asarray(t, dtype=float)
    n = np.asarray(n, dtype=float)
    order = np.argsort(t)
    t, n = t[order], n[order]
    if np.isnan(k) or consecutive < 1:
        return np.nan

    ratio = scaling_ratio(t, n, k)
    deviation = ratio - 1.0
    candidates = np.flatnonzero(t > t_start)
    for index in candidates:
        end = index + consecutive
        if end > len(t):
            break
        window = deviation[index:end]
        if np.all(window > tol) or np.all(window < -tol):
            return float(t[index])
    return np.nan


def find_t_cross(t, n, k, tol=0.05, min_points=4, n_eq=None, floor_frac=0.05):
    """
    Walk forward in time and find the first t where the sqrt(t) prediction
    (using k fit from the full early window) deviates from the observed n
    by more than `tol` fractional error. That point is t_cross: the
    diffusion-controlled -> capacity-controlled crossover.

    n_eq / floor_frac : guards against false triggers at very small t, where
    both n and n_pred are near zero and tiny absolute noise produces huge
    fractional error. Points are only checked once n_pred exceeds
    floor_frac * n_eq (default 5% of equilibrium capacity). If n_eq isn't
    passed, it's estimated from max(n) in the series.

    Returns np.nan if the deviation never exceeds tol (i.e. sqrt(t) holds
    over the whole observed range -- rare, usually means t range too short).
    """
    t = np.asarray(t, dtype=float)
    n = np.asarray(n, dtype=float)
    order = np.argsort(t)
    t, n = t[order], n[order]

    if np.isnan(k):
        return np.nan

    if n_eq is None:
        n_eq = np.max(n)

    n_pred = k * np.sqrt(t)
    floor = floor_frac * n_eq

    # avoid divide-by-zero at t=0
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_err = np.abs(n - n_pred) / np.where(n_pred == 0, np.nan, n_pred)

    for i in range(min_points, len(t)):
        if n_pred[i] < floor:
            continue  # too close to t=0 for fractional error to be meaningful
        if frac_err[i] > tol:
            return t[i]

    return np.nan


def get_n_eq(t, n, frac_of_max_t=0.9):
    """
    Equilibrium capacity estimate: mean of n over the last (1 - frac_of_max_t)
    fraction of the time series (i.e. the plateau region), falling back to
    the single max observed t if the series is short.
    """
    t = np.asarray(t, dtype=float)
    n = np.asarray(n, dtype=float)
    order = np.argsort(t)
    t, n = t[order], n[order]

    cutoff = t.max() * frac_of_max_t
    tail = n[t >= cutoff]
    if len(tail) == 0:
        return n[-1]
    return float(np.mean(tail))


# ----------------------------------------------------------------------
# Batch summary
# ----------------------------------------------------------------------

def compute_regime_summary(df, group_cols=("geometry", "R", "H"),
                            t_col="t", n_col="n",
                            early_window_t=500, early_window_start_t=5,
                            fit_method="median_ratio", tol=0.05,
                            anchor_t=2000, anchor_bracket=(1000, 3000),
                            persistent_points=3):
    """
    Compute k, r2_early, t_cross, n_eq for every (geometry, R, H) group.

    early_window_start_t / early_window_t : fixed, independently selected
                      early-time fitting window. Defaults to 5--500 s: the
                      lower bound excludes the documented solver startup
                      transient, and the upper bound stays independent of
                      the 2,000 s comparison anchor.
    fit_method      : "median_ratio" (default) or "least_squares"; passed to
                      fit_sqrt_k().
    tol             : fractional deviation from sqrt(t) used for the
                      persistent-departure diagnostic (default 5%).
    anchor_t        : process/comparison time at which q_ratio_anchor is
                      reported (default 2,000 s).
    anchor_bracket  : (t_low, t_high) used for alpha_anchor, the local
                      log-log exponent. Pure sqrt(t) uptake has alpha=0.5.
    persistent_points : same-direction post-fit points required before
                      assigning t_cross. This is a descriptive departure
                      time, not a claimed capacity crossover.
    """
    rows = []

    # dropna=False: pandas' groupby silently discards any row whose
    # grouping columns contain NaN by default. That's exactly the kind
    # of silent-NaN failure this codebase already hard-stops on
    # elsewhere (see master_pipeline.build_master_table's missing-SA/V
    # check) -- made explicit here so a NaN R/H (e.g. an untextured
    # baseline row) fails loudly downstream instead of just disappearing.
    for keys, g in df.groupby(list(group_cols), dropna=False):
        g = g.sort_values(t_col)
        t = g[t_col].values
        n = g[n_col].values

        t_max_fit = early_window_t
        k, r2 = fit_sqrt_k(t, n, t_min=early_window_start_t,
                           t_max=t_max_fit, method=fit_method)
        n_eq = get_n_eq(t, n)
        t_cross = find_persistent_deviation(
            t, n, k, t_start=t_max_fit, tol=tol,
            consecutive=persistent_points,
        )
        q_ratio_anchor = value_at_time(t, scaling_ratio(t, n, k), anchor_t)
        alpha_anchor = local_log_slope(t, n, *anchor_bracket)

        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update(dict(
            k=k, r2_early=r2, t_cross=t_cross, n_eq=n_eq,
            t_fit_start=early_window_start_t, t_fit_window=t_max_fit,
            fit_method=fit_method, anchor_t=anchor_t,
            q_ratio_anchor=q_ratio_anchor,
            deviation_anchor=q_ratio_anchor - 1 if np.isfinite(q_ratio_anchor) else np.nan,
            alpha_anchor=alpha_anchor,
        ))
        rows.append(row)

    summary = pd.DataFrame(rows)
    return summary.sort_values(list(group_cols)).reset_index(drop=True)


# ----------------------------------------------------------------------
# Validity check: does the sqrt(t) collapse actually hold?
# ----------------------------------------------------------------------

def collapse_validity_report(summary, r2_threshold=0.98):
    """
    Flags configs where the sqrt(t) early-time model is a poor fit
    (r2_early < r2_threshold). If a large fraction of a geometry family
    fails this, the k/t_cross framework should NOT be trusted for that
    family without revisiting the fit window.
    """
    summary = summary.copy()
    summary["collapse_ok"] = summary["r2_early"] >= r2_threshold

    by_family = (
        summary.groupby("geometry")["collapse_ok"]
        .agg(["mean", "sum", "count"])
        .rename(columns={"mean": "frac_ok", "sum": "n_ok", "count": "n_total"})
    )
    return summary, by_family


# ----------------------------------------------------------------------
# Plotting: visual sanity check before trusting the framework
# ----------------------------------------------------------------------

def plot_collapse_check(df, geometry, R, H, t_col="t", n_col="n",
                         t_min_fit=5, t_max_fit=500,
                         fit_method="median_ratio", tol=0.05,
                         persistent_points=3, ax=None):
    """
    Plot n(t) vs sqrt(t) for one specific config, overlay the fitted line,
    and mark t_cross. A good sqrt(t) collapse shows the data lying on a
    straight line through the origin for the early window, then bending
    away (upward saturation) after t_cross.
    """
    import matplotlib.pyplot as plt

    g = df[
        (df["geometry"] == geometry)
        & np.isclose(df["R"], R, rtol=1e-6, atol=1e-12)
        & np.isclose(df["H"], H, rtol=1e-6, atol=1e-12)
    ]
    g = g.sort_values(t_col)
    t = g[t_col].values
    n = g[n_col].values

    if len(t) == 0:
        available = df.loc[df["geometry"] == geometry, ["R", "H"]].drop_duplicates()
        raise ValueError(
            f"No rows found for geometry={geometry!r}, R={R}, H={H}.\n"
            f"Available geometries: {sorted(df['geometry'].unique())}\n"
            f"Available (R, H) for {geometry!r}:\n{available.to_string(index=False)}"
        )

    k, r2 = fit_sqrt_k(t, n, t_min=t_min_fit, t_max=t_max_fit,
                       method=fit_method)
    t_cross = find_persistent_deviation(
        t, n, k, t_start=t_max_fit, tol=tol, consecutive=persistent_points,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4.5))

    x = np.sqrt(t)
    ax.scatter(x, n, s=14, label="data", color="black")
    x_line = np.linspace(0, x.max(), 100)
    ax.plot(x_line, k * x_line, "--", color="tab:red",
            label=f"k*sqrt(t) fit (k={k:.3e}, r2={r2:.4f})")

    if not np.isnan(t_cross):
        ax.axvline(np.sqrt(t_cross), color="tab:blue", linestyle=":",
                   label=f"t_cross = {t_cross:.0f} s")

    ax.set_xlabel("sqrt(t)  [s^0.5]")
    ax.set_ylabel("n(t)  [mol]")
    ax.set_title(f"{geometry}  R={R}um  H={H}um")
    ax.legend(fontsize=8)
    return ax


def plot_scaling_ratio(df, geometry, R, H, t_col="t", n_col="n",
                       t_min_fit=5, t_max_fit=500,
                       fit_method="median_ratio", anchor_t=2000, ax=None):
    """Plot n(t)/(k*sqrt(t)) so exact sqrt(t) scaling is a horizontal line.

    This is the preferred visual diagnostic for a fixed-time kinetics
    comparison. It exposes any curvature directly rather than encoding it as
    a first residual crossing against a fit that includes the anchor time.
    """
    import matplotlib.pyplot as plt

    g = df[
        (df["geometry"] == geometry)
        & np.isclose(df["R"], R, rtol=1e-6, atol=1e-12)
        & np.isclose(df["H"], H, rtol=1e-6, atol=1e-12)
    ].sort_values(t_col)
    if len(g) == 0:
        raise ValueError(f"No rows found for geometry={geometry!r}, R={R}, H={H}.")

    t = g[t_col].to_numpy()
    n = g[n_col].to_numpy()
    k, _ = fit_sqrt_k(t, n, t_min=t_min_fit, t_max=t_max_fit,
                      method=fit_method)
    ratio = scaling_ratio(t, n, k)
    # The startup transient is deliberately excluded from both the fit and
    # the visual diagnostic; plotting it would dominate the y-range while
    # conveying no useful information about the diffusion regime.
    mask = np.isfinite(ratio) & (t >= t_min_fit)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(t[mask], ratio[mask], "o-", ms=3.5, lw=1.2, color="black")
    ax.axhline(1, color="tab:red", ls="--", label=r"ideal $k\sqrt{t}$")
    ax.axhspan(.95, 1.05, color="tab:green", alpha=.12, label="±5% band")
    ax.axvline(anchor_t, color="tab:blue", ls=":", label=f"anchor = {anchor_t:g} s")
    ax.set_xscale("log")
    ax.set_xlabel("t (s, log scale)")
    ax.set_ylabel(r"$n(t) / [k\sqrt{t}]$")
    ax.set_title(f"{geometry.replace('_', ' ')}: R={R * 1e6:g} μm, H={H * 1e6:g} μm")
    ax.grid(True, linewidth=.3, alpha=.4)
    ax.legend(fontsize=8)
    return ax


# ----------------------------------------------------------------------
# Glue: convert nested {(geometry, (R,H)): {"R":..,"H":..,"points":[(t,n),...]}}
# dicts (as produced by the existing viz_utils / master_pipeline output)
# into the long-format DataFrame this module expects.
# ----------------------------------------------------------------------

def nested_dict_to_long_df(data_dict, geometry_name=None):
    """
    Convert one family's nested dict (keys are (geometry, (R,H)) tuples,
    values hold 'R', 'H', 'points') into a long-format DataFrame with
    columns: geometry, R, H, t, n.

    geometry_name : if the dict's outer key's first element isn't a clean
                     geometry string (or you want to override it), pass it
                     explicitly here. Otherwise it's read from the key.
    """
    rows = []
    for key, val in data_dict.items():
        geom = geometry_name if geometry_name is not None else key[0]
        R = val["R"]
        H = val["H"]
        for t, n in val["points"]:
            rows.append(dict(geometry=geom, R=R, H=H, t=t, n=n))
    return pd.DataFrame(rows)


def merge_family_dicts(*dicts_with_names):
    """
    Combine multiple families' nested dicts into a single long-format df.
    Usage:
        df = merge_family_dicts(
            (groove_dict, "groove"),
            (recess_dict, "recess"),
            (cone_dict, "cone"),
        )
    """
    frames = []
    for d, name in dicts_with_names:
        frames.append(nested_dict_to_long_df(d, geometry_name=name))
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------
# NEW: fold the untextured (flat film) baseline into this same framework.
#
# Why this exists: an earlier ad hoc diagnostic (compute_alpha_series /
# check_flat_alpha in preprocessing.py) computed alpha(t) via raw
# two-point finite differences between consecutive exported timestamps.
# That is NOT the same calculation this module uses for every textured
# family -- fit_sqrt_k uses a fixed 5-500s window, and t_cross comes from
# find_persistent_deviation (a sustained, 3-consecutive-point departure
# from that fit), not a single-point local slope. A t_cross or alpha_anchor
# for the flat film is only comparable to the textured families' numbers
# if it is produced by the SAME functions. This helper does that by
# routing the flat film through nested_dict_to_long_df exactly like any
# other geometry, rather than adding a parallel calculation path.
# ----------------------------------------------------------------------

def add_untextured_baseline(df, baseline_points, geometry_name="untextured"):
    """
    df               : existing long-format DataFrame (geometry, R, H, t, n)
                        for the textured families, e.g. from
                        merge_family_dicts(...).
    baseline_points   : the "points" list from
                        preprocessing.build_untextured_baseline(...)["points"],
                        i.e. [(t, n), ...] for the flat film.
    geometry_name     : label used in the "geometry" column. R and H are
                        set to 0.0, not NaN -- "zero texture" is a coherent
                        physical value here, it can't collide with any real
                        swept family (no family samples R=0 or H=0), and it
                        avoids a real bug: pandas' groupby drops NaN-valued
                        grouping keys by default, which silently discarded
                        this entire row from compute_regime_summary the
                        first time this used np.nan. See that function's
                        dropna=False comment for the second half of the fix.

    Returns a new concatenated DataFrame; does not mutate df. Pass the
    result straight into compute_regime_summary() to get k, r2_early,
    t_cross, alpha_anchor for the flat film computed by the exact same
    fit_sqrt_k / find_persistent_deviation / local_log_slope logic as
    every textured family -- directly comparable, not a separate metric.
    """
    flat_dict = {
        (geometry_name, (0.0, 0.0)): {
            "R": 0.0, "H": 0.0, "points": list(baseline_points),
        }
    }
    df_flat = nested_dict_to_long_df(flat_dict, geometry_name=geometry_name)
    return pd.concat([df, df_flat], ignore_index=True)


if __name__ == "__main__":
    # Minimal smoke test with synthetic data: n(t) = k*sqrt(t) capped at n_eq
    rng = np.random.default_rng(0)
    t_vals = np.linspace(0, 20000, 60)

    def synth(k, n_eq):
        n = k * np.sqrt(t_vals)
        n = np.minimum(n, n_eq)
        n += rng.normal(0, n_eq * 0.005, size=n.shape)  # tiny noise
        return n

    rows = []
    configs = [
        ("recess", 20, 5, 3.0e-6, 1.76e-4),
        ("recess", 20, 35, 6.5e-6, 1.10e-4),
        ("groove", 20, 35, 5.0e-6, 3.0e-5),  # low n_eq -> saturates fast
    ]
    for geom, R, H, k_true, n_eq_true in configs:
        n = synth(k_true, n_eq_true)
        for tt, nn in zip(t_vals, n):
            rows.append(dict(geometry=geom, R=R, H=H, t=tt, n=nn))

    df_test = pd.DataFrame(rows)
    summary = compute_regime_summary(df_test, early_window_t=2000)
    print(summary)