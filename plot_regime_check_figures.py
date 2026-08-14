# """Build Figure 2 and Figure 2b: validity checks for the 2,000 s metric.

# The figures use an early, independent 5--500 s fit to ``n = k sqrt(t)``.
# The crossover is the first of three consecutive output points after the fit
# window that lies outside a +/-5% band.  Figure 2 overlays each fitted curve
# within its family; Figure 2b reports the two diagnostics at 2,000 s for every
# configuration.

# Run from the repository root:
#     python plot_regime_check_figures.py
# """

# from pathlib import Path

# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# import early_regime_fit as erf
# import preprocessing as pp
# from matplotlib.lines import Line2D


# DATA_DIR = Path("data")
# FIGURE_DIR = Path("figures")
# ANCHOR_S = 2_000
# FIT_WINDOW_S = (5, 500)
# FAMILIES = {
#     "Cylinder recess": "cylinder-recessed",
#     "Square recess": "square-recessed",
#     "Cone recess": "cone-recessed",
#     "Square pillar": "square-pillars",
#     "Edge": "edges",
#     "Cylindrical pillar": "cylindrical-pillars",
#     "Groove": "grooves",
# }


# def load_curves(data_dir=DATA_DIR):
#     """Load every supplied uptake sweep into early_regime_fit's long format."""
#     frames = []
#     for family, stem in FAMILIES.items():
#         entry = pp.build_family_entry((data_dir / f"{stem}-uptake.txt").read_text())
#         series = {(family, key): value for key, value in entry["series"].items()}
#         frames.append(erf.nested_dict_to_long_df(series, geometry_name=family))
#     return pd.concat(frames, ignore_index=True)


# def compute_summary(curves):
#     """Return the per-configuration diagnostics used by both figures."""
#     summary = erf.compute_regime_summary(
#         curves,
#         early_window_start_t=FIT_WINDOW_S[0],
#         early_window_t=FIT_WINDOW_S[1],
#         fit_method="median_ratio",
#         tol=0.05,
#         anchor_t=ANCHOR_S,
#         anchor_bracket=(1_000, 3_000),
#         persistent_points=3,
#     )
#     summary["R_um"] = summary["R"] * 1e6
#     summary["H_um"] = summary["H"] * 1e6
#     summary["departs_by_anchor"] = summary["t_cross"] <= ANCHOR_S
#     return summary


# def _config_style(R, H, radii, heights):
#     """Encode radius by colour and height by line style consistently."""
#     color = plt.get_cmap("viridis")(list(radii).index(R) / (len(radii) - 1))
#     style = ["-", "--", ":", "-.", (0, (3, 1, 1, 1))][list(heights).index(H)]
#     return color, style


# # def build_figure_2(curves, summary):
# #     """Residual of simulated uptake vs. independent k*sqrt(t) reference, by family.

# #     Plots (n(t) - k*sqrt(t)) / (k*sqrt(t)) * 100 so departure from the diffusion-
# #     controlled regime is visible directly, rather than inferred from two nearly-
# #     overlapping curves. A +/-5% band matches the crossover-marker threshold
# #     already used in build_figure_2, so the two figures read consistently.
# #     """
# #     fig, axes = plt.subplots(2, 4, figsize=(16, 8.4), sharex=True, sharey=True)
# #     axes = axes.ravel()
# #     radii = tuple(sorted(summary["R"].unique()))
# #     heights = tuple(sorted(summary["H"].unique()))
# #     summary_index = summary.set_index(["geometry", "R", "H"])

# #     for ax, family in zip(axes, FAMILIES):
# #         subset = curves[curves["geometry"] == family]
# #         for (R, H), config in subset.groupby(["R", "H"], sort=True):
# #             config = config.sort_values("t")
# #             row = summary_index.loc[(family, R, H)]
# #             color, linestyle = _config_style(R, H, radii, heights)

# #             t = config["t"].to_numpy()
# #             n = config["n"].to_numpy()
# #             valid = t >= FIT_WINDOW_S[0]

# #             k_sqrt_t = row.k * np.sqrt(t[valid])
# #             # guard against div-by-zero at t -> 0
# #             nonzero = k_sqrt_t > 0
# #             resid_pct = np.full_like(k_sqrt_t, np.nan)
# #             resid_pct[nonzero] = (n[valid][nonzero] - k_sqrt_t[nonzero]) / k_sqrt_t[nonzero] * 100

# #             ax.plot(t[valid], resid_pct, color=color, linestyle=linestyle,
# #                     linewidth=1.05, alpha=.85)

# #             if np.isfinite(row.t_cross):
# #                 n_cross = erf.value_at_time(t, n, row.t_cross)
# #                 k_cross = row.k * np.sqrt(row.t_cross)
# #                 resid_cross = (n_cross - k_cross) / k_cross * 100 if k_cross > 0 else np.nan
# #                 ax.plot(row.t_cross, resid_cross, marker="o", ms=3.1,
# #                         markerfacecolor="white", markeredgewidth=.85,
# #                         markeredgecolor=color, zorder=4)

# #         ax.axhspan(-5, 5, color="#4c78a8", alpha=.08, zorder=0)
# #         ax.axhline(0, color="black", linewidth=.6, alpha=.6, zorder=1)
# #         ax.axvspan(*FIT_WINDOW_S, color="#999999", alpha=.06, zorder=0)
# #         ax.axvline(ANCHOR_S, color="#c23b22", linestyle="--", linewidth=1)
# #         ax.set_title(family, fontsize=11, weight="bold")
# #         ax.set_xscale("log")
# #         ax.set_xlim(5, 50_000)
# #         ax.grid(True, which="major", linewidth=.35, alpha=.4)

# #     for ax in axes[:4]:
# #         ax.set_xlabel("Time, t (s)")
# #     for ax in axes[4:7]:
# #         ax.set_xlabel("Time, t (s)")
# #     for ax in axes[::4]:
# #         ax.set_ylabel(r"Residual, $(n - k\sqrt{t})/k\sqrt{t}$ (%)")

# #     axes[-1].axis("off")

# #     radius_handles = [mpl.lines.Line2D([], [], color=_config_style(R, heights[0], radii, heights)[0],
# #                                        lw=2, label=f"R = {R * 1e6:g} µm") for R in radii]
# #     height_handles = [mpl.lines.Line2D([], [], color="black", linestyle=_config_style(radii[0], H, radii, heights)[1],
# #                                        lw=1.5, label=f"H = {H * 1e6:g} µm") for H in heights]
# #     reference_handles = [
# #         mpl.patches.Patch(color="#4c78a8", alpha=.15, label=r"$\pm$5% band"),
# #         mpl.lines.Line2D([], [], color="#c23b22", linestyle="--", lw=1.1,
# #                          label="2,000 s checkpoint"),
# #         mpl.lines.Line2D([], [], color="black", marker="o", markerfacecolor="white", lw=0,
# #                          label="persistent ±5% crossover"),
# #     ]
# #     fig.legend(handles=reference_handles + radius_handles + height_handles, ncol=5,
# #                loc="lower center", bbox_to_anchor=(.5, -.02), fontsize=8, frameon=False)
# #     fig.suptitle(r"Figure 2. Residual departure from independent $k\sqrt{t}$ fit", y=.99, fontsize=15)
# #     fig.text(.5, .925,
# #               "Residual = (simulated − fit) / fit, as a percentage. Shaded band marks the ±5% crossover threshold used for Figure 2b.",
# #               ha="center", fontsize=9)
# #     fig.tight_layout(rect=(0, .105, 1, .91))
# #     return fig
# def build_figure_2(curves, summary):
#     """
#     Figure 2.
#     Residual departure from an independent early-time k√t fit.
#     The emphasis is on WHEN the departure occurs rather than the
#     exact residual trajectory.
#     """

#     fig, axes = plt.subplots(
#         2, 4,
#         figsize=(16, 8.4),
#         sharex=True,
#         sharey=True,
#     )

#     axes = axes.ravel()

#     radii = tuple(sorted(summary["R"].unique()))
#     heights = tuple(sorted(summary["H"].unique()))

#     summary_index = summary.set_index(["geometry", "R", "H"])

#     for ax, family in zip(axes, FAMILIES):

#         subset = curves[curves["geometry"] == family]
#         family_summary = summary[summary["geometry"] == family]

#         # ----------------------------------------------------
#         # Plot every configuration
#         # ----------------------------------------------------

#         for (R, H), config in subset.groupby(["R", "H"], sort=True):

#             config = config.sort_values("t")

#             row = summary_index.loc[(family, R, H)]

#             color, linestyle = _config_style(
#                 R,
#                 H,
#                 radii,
#                 heights,
#             )

#             t = config["t"].to_numpy()
#             n = config["n"].to_numpy()

#             mask = t >= FIT_WINDOW_S[0]

#             t = t[mask]
#             n = n[mask]

#             fit = row.k * np.sqrt(t)

#             resid = 100 * (n - fit) / fit

#             # ------------------------------------------------
#             # Background residual curves
#             # ------------------------------------------------

#             ax.plot(
#                 t,
#                 resid,
#                 color=color,
#                 linestyle=linestyle,
#                 linewidth=0.9,
#                 alpha=0.35,
#             )

#             # ------------------------------------------------
#             # Crossover marker
#             # ------------------------------------------------

#             if np.isfinite(row.t_cross):

#                 n_cross = erf.value_at_time(
#                     config["t"].to_numpy(),
#                     config["n"].to_numpy(),
#                     row.t_cross,
#                 )

#                 fit_cross = row.k * np.sqrt(row.t_cross)

#                 resid_cross = (
#                     (n_cross - fit_cross)
#                     / fit_cross
#                     * 100
#                 )

#                 # connector
#                 ax.vlines(
#                     row.t_cross,
#                     -5 if resid_cross < 0 else 5,
#                     resid_cross,
#                     color=color,
#                     linewidth=0.9,
#                     alpha=0.45,
#                     zorder=5,
#                 )

#                 # marker
#                 ax.plot(
#                     row.t_cross,
#                     resid_cross,
#                     marker="o",
#                     markersize=5.5,
#                     markerfacecolor="white",
#                     markeredgecolor=color,
#                     markeredgewidth=1.2,
#                     linestyle="None",
#                     zorder=8,
#                 )

#         # ----------------------------------------------------
#         # Family departure window
#         # ----------------------------------------------------

#         departures = family_summary["t_cross"].dropna()

#         if len(departures):

#             t0 = departures.min()
#             t1 = departures.max()

#             ax.axvspan(
#                 t0,
#                 t1,
#                 color="#f28e2b",
#                 alpha=0.08,
#                 zorder=0,
#             )

#             summary_text = (
#                 f"Departure window\n"
#                 f"{t0/1000:.1f}–{t1/1000:.1f} ks\n"
#                 f"median {departures.median()/1000:.1f} ks"
#             )

#             ax.text(
#                 0.03,
#                 0.97,
#                 summary_text,
#                 transform=ax.transAxes,
#                 ha="left",
#                 va="top",
#                 fontsize=8,
#                 bbox=dict(
#                     facecolor="white",
#                     alpha=0.85,
#                     edgecolor="0.8",
#                     boxstyle="round,pad=0.3",
#                 ),
#             )

#         # ----------------------------------------------------
#         # Reference guides
#         # ----------------------------------------------------

#         ax.axhspan(
#             -5,
#             5,
#             color="#4c78a8",
#             alpha=0.08,
#         )

#         ax.axhline(
#             0,
#             color="black",
#             linewidth=0.6,
#             alpha=0.7,
#         )

#         if subset["t"].min() <= ANCHOR_S:

#             ax.axvline(
#                 ANCHOR_S,
#                 color="#c23b22",
#                 linestyle="--",
#                 linewidth=1.2,
#             )

#             ax.text(
#                 ANCHOR_S,
#                 23,
#                 "2000 s",
#                 rotation=90,
#                 color="#c23b22",
#                 fontsize=7,
#                 ha="center",
#                 va="bottom",
#             )

#         ax.set_xscale("log")
#         ax.set_xlim(5, 50000)
#         ax.set_ylim(-35, 25)

#         ax.grid(
#             True,
#             which="major",
#             linewidth=0.35,
#             alpha=0.35,
#         )

#         ax.set_title(
#             family,
#             fontsize=11,
#             weight="bold",
#         )

#     # ----------------------------------------------------
#     # Labels
#     # ----------------------------------------------------

#     for ax in axes[4:7]:
#         ax.set_xlabel("Time, $t$ (s)")

#     for ax in axes[::4]:
#         ax.set_ylabel(
#             r"Residual, $(n-k\sqrt{t})/(k\sqrt{t})$ (%)"
#         )

#     axes[-1].axis("off")

#     # ----------------------------------------------------
#     # Legend
#     # ----------------------------------------------------

#     radius_handles = [
#         mpl.lines.Line2D(
#             [],
#             [],
#             color=_config_style(
#                 R,
#                 heights[0],
#                 radii,
#                 heights,
#             )[0],
#             lw=2,
#             label=f"R = {R*1e6:g} μm",
#         )
#         for R in radii
#     ]

#     height_handles = [
#         mpl.lines.Line2D(
#             [],
#             [],
#             color="black",
#             linestyle=_config_style(
#                 radii[0],
#                 H,
#                 radii,
#                 heights,
#             )[1],
#             lw=1.5,
#             label=f"H = {H*1e6:g} μm",
#         )
#         for H in heights
#     ]

#     reference_handles = [

#         mpl.patches.Patch(
#             color="#4c78a8",
#             alpha=0.15,
#             label="±5% band",
#         ),

#         mpl.patches.Patch(
#             color="#f28e2b",
#             alpha=0.15,
#             label="departure window",
#         ),

#         mpl.lines.Line2D(
#             [],
#             [],
#             color="#c23b22",
#             linestyle="--",
#             lw=1.2,
#             label="2000 s checkpoint",
#         ),

#         mpl.lines.Line2D(
#             [],
#             [],
#             marker="o",
#             markerfacecolor="white",
#             markeredgecolor="black",
#             lw=0,
#             markersize=6,
#             label="persistent ±5% departure",
#         ),
#     ]

#     fig.legend(
#         handles=(
#             radius_handles
#             + height_handles
#             + reference_handles
#         ),
#         ncol=5,
#         loc="lower center",
#         bbox_to_anchor=(0.5, -0.02),
#         frameon=False,
#         fontsize=8,
#     )

#     fig.suptitle(
#         "Figure 2. Departure from the independent $k\\sqrt{t}$ diffusion regime",
#         fontsize=15,
#         y=0.99,
#     )

#     fig.text(
#         0.5,
#         0.935,
#         "Residual curves are shown in the background. Open circles indicate the first persistent "
#         "departure beyond ±5%, while the shaded region summarises the departure-time window "
#         "for each geometry family.",
#         ha="center",
#         fontsize=9,
#     )

#     fig.tight_layout(rect=(0, 0.09, 1, 0.93))

#     return fig

# def _heatmap(ax, table, cmap, norm, fmt, title, fail_mask=None):
#     image = ax.imshow(table.to_numpy(), cmap=cmap, norm=norm, aspect="equal")
#     ax.set_xticks(range(len(table.columns)), [f"{value:g}" for value in table.columns])
#     ax.set_yticks(range(len(table.index)), [f"{value:g}" for value in table.index])
#     ax.set_title(title, fontsize=9, pad=5)
#     for row, H in enumerate(table.index):
#         for col, R in enumerate(table.columns):
#             value = table.loc[H, R]
#             ax.text(col, row, fmt.format(value), ha="center", va="center", fontsize=7,
#                     color="black" if norm(value) > .35 else "white")
#             if fail_mask is not None and fail_mask.loc[H, R]:
#                 ax.add_patch(plt.Rectangle((col - .5, row - .5), 1, 1, fill=False,
#                                            edgecolor="#b2182b", linewidth=1.55))
#     return image


# def build_figure_2b(summary):
#     """Facet q_ratio and local exponent heatmaps by geometry family."""
#     fig, axes = plt.subplots(4, 4, figsize=(15.5, 13), layout="constrained")
#     q_norm = mpl.colors.Normalize(vmin=.89, vmax=1.00)
#     alpha_norm = mpl.colors.Normalize(vmin=.42, vmax=.50)
#     q_cmap = plt.get_cmap("RdYlGn")
#     alpha_cmap = plt.get_cmap("RdYlGn")
#     images = []

#     for pair_index, family in enumerate(FAMILIES):
#         row, first_col = divmod(pair_index, 2)
#         first_col *= 2
#         family_summary = summary[summary["geometry"] == family]
#         q_table = family_summary.pivot(index="H_um", columns="R_um", values="q_ratio_anchor").sort_index(ascending=False)
#         alpha_table = family_summary.pivot(index="H_um", columns="R_um", values="alpha_anchor").sort_index(ascending=False)
#         bad_q = q_table < .95
#         bad_alpha = alpha_table < .475
#         images.append(_heatmap(axes[row, first_col], q_table, q_cmap, q_norm, "{:.3f}", r"$q_{ratio}$", bad_q))
#         images.append(_heatmap(axes[row, first_col + 1], alpha_table, alpha_cmap, alpha_norm, "{:.3f}", r"local $\alpha$", bad_alpha))
#         axes[row, first_col].set_ylabel("H (µm)")
#         axes[row, first_col + 1].set_yticklabels([])
#         axes[row, first_col].set_xlabel("R (µm)")
#         axes[row, first_col + 1].set_xlabel("R (µm)")
#         axes[row, first_col].annotate(family, xy=(1, 1), xycoords="axes fraction",
#                                       xytext=(0, 18), textcoords="offset points",
#                                       ha="center", va="bottom", fontsize=11, weight="bold")

#     axes[3, 2].axis("off")
#     axes[3, 3].axis("off")
#     q_bar = fig.colorbar(images[0], ax=[axes[r, c] for r in range(4) for c in (0, 2)], shrink=.72, pad=.015)
#     q_bar.set_label(r"$q_{ratio}=n(t)/[k\sqrt{t}]$  (ideal: 1)")
#     alpha_bar = fig.colorbar(images[1], ax=[axes[r, c] for r in range(4) for c in (1, 3)], shrink=.72, pad=.015)
#     alpha_bar.set_label(r"local $alpha=dln n/dln t$  (ideal: 0.5)")
#     fig.suptitle("Figure 2b. √t-regime diagnostics at t = 2,000 s", fontsize=15)
#     fig.text(.5, .012, "Red borders flag q_ratio < 0.950 or α < 0.475; both metrics use the same independent 5–500 s fit.",
#              ha="center", fontsize=9)
#     return fig


# def _get_color(R, R_values):
#     """Map R to color intensity: smallest R = darkest, largest R = brightest"""
#     # Handle both pandas Series and numpy arrays
#     if hasattr(R_values, 'unique'):
#         R_unique = sorted(R_values.unique())
#     else:
#         R_unique = sorted(np.unique(R_values))
    
#     if len(R_unique) == 1:
#         return 'blue'
#     norm_R = (R - R_unique[0]) / (R_unique[-1] - R_unique[0])
#     # Dark blue to light blue gradient
#     return (0.1 + 0.6 * norm_R, 0.1 + 0.6 * norm_R, 0.9 - 0.4 * norm_R)

# def _get_linestyle(H, H_values):
#     """Map H to linestyle: shallow = solid, deep = dashed"""
#     # Handle both pandas Series and numpy arrays
#     if hasattr(H_values, 'unique'):
#         H_unique = sorted(H_values.unique())
#     else:
#         H_unique = sorted(np.unique(H_values))
    
#     if len(H_unique) == 1:
#         return '-'
#     # Assuming first is shallow, last is deep
#     return '-' if H == H_unique[0] else '--'

# def build_figure_2_annotated(curves, summary):
#     """
#     Generate Figure 2: Residual departure from independent k√t fit.
    
#     Parameters:
#     -----------
#     curves : pd.DataFrame
#         DataFrame from load_curves() with columns [geometry, R, H, t, n]
#     summary : pd.DataFrame
#         DataFrame with columns [geometry, R, H, k, t_cross, ...]
    
#     Returns:
#     --------
#     fig : matplotlib.figure.Figure
#         The generated figure object
#     """
#     # Style configuration
#     plt.rcParams['font.family'] = 'serif'
#     plt.rcParams['font.serif'] = ['Times New Roman']
#     plt.rcParams['axes.labelsize'] = 10
#     plt.rcParams['xtick.labelsize'] = 9
#     plt.rcParams['ytick.labelsize'] = 9
#     plt.rcParams['legend.fontsize'] = 8
    
#     # Families in order for subplots (8 panels, last empty)
#     family_names = list(FAMILIES.keys()) + ['']  # 8th panel is empty
    
#     # Create figure with 2x4 subplot grid
#     fig, axes = plt.subplots(2, 4, figsize=(14, 8), sharey=True)
#     axes = axes.flatten()
    
#     # Set up global parameters
#     t_min, t_max = 5, 50000
    
#     # Pre-compute R and H values for each family for consistent styling
#     family_R_values = {}
#     family_H_values = {}
#     for family in FAMILIES.keys():
#         family_curves = curves[curves['geometry'] == family]
#         if not family_curves.empty:
#             family_R_values[family] = sorted(family_curves['R'].unique())
#             family_H_values[family] = sorted(family_curves['H'].unique())
    
#     def get_color(R, family):
#         """Get color for a given R value within a family"""
#         R_values = family_R_values.get(family, [])
#         if len(R_values) == 1:
#             return 'blue'
#         # Find position of R in sorted R_values
#         if R in R_values:
#             idx = R_values.index(R)
#             norm_R = idx / (len(R_values) - 1)
#         else:
#             # If R not found, use value-based normalization
#             norm_R = (R - R_values[0]) / (R_values[-1] - R_values[0])
#         # Dark blue to light blue gradient
#         return (0.1 + 0.6 * norm_R, 0.1 + 0.6 * norm_R, 0.9 - 0.4 * norm_R)
    
#     def get_linestyle(H, family):
#         """Get linestyle for a given H value within a family"""
#         H_values = family_H_values.get(family, [])
#         if len(H_values) == 1:
#             return '-'
#         # Assuming first is shallow, last is deep
#         return '-' if H == H_values[0] else '--'
    
#     # Process each family
#     for idx, family in enumerate(family_names):
#         ax = axes[idx]
        
#         # Skip empty panel
#         if not family:
#             ax.axis('off')
#             continue
        
#         # Filter data for this family
#         family_curves = curves[curves['geometry'] == family]
#         family_summary = summary[summary['geometry'] == family]
        
#         if family_curves.empty:
#             ax.set_title(family, fontweight='bold', fontsize=11)
#             ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
#                    ha='center', va='center', fontsize=10)
#             continue
        
#         # Plot each configuration
#         max_resid = 0
#         crossover_points = []
        
#         for (R, H), group in family_curves.groupby(['R', 'H']):
#             # Get k from summary
#             summary_row = family_summary[(family_summary['R'] == R) & (family_summary['H'] == H)]
#             if summary_row.empty:
#                 continue
            
#             k = summary_row['k'].iloc[0]
#             t_cross = summary_row['t_cross'].iloc[0] if 't_cross' in summary_row.columns else np.nan
            
#             # Calculate residual
#             with np.errstate(divide='ignore', invalid='ignore'):
#                 fit = k * np.sqrt(group['t'].values)
#                 resid = (group['n'].values - fit) / fit * 100
#                 resid = np.where(np.isfinite(resid), resid, np.nan)
            
#             # Get style
#             color = get_color(R, family)
#             linestyle = get_linestyle(H, family)
            
#             # Plot residual
#             ax.plot(group['t'].values, resid, 
#                    color=color, linestyle=linestyle, 
#                    linewidth=1.5, alpha=0.8,
#                    label=f'R={R*1e6:.0f}µm, H={H*1e6:.0f}µm')
            
#             # Track max residual for placement
#             if not np.isnan(resid).all():
#                 max_resid = max(max_resid, np.nanmax(np.abs(resid)))
            
#             # Get crossover point
#             if pd.notna(t_cross) and t_cross > 0:
#                 # Find residual value at crossover time
#                 idx_cross = np.argmin(np.abs(group['t'].values - t_cross))
#                 if idx_cross < len(resid) and not np.isnan(resid[idx_cross]):
#                     crossover_points.append((t_cross, resid[idx_cross], R, H))
        
#         # Add ±5% shaded band
#         ax.axhspan(-5, 5, color='lightblue', alpha=0.08, zorder=0)
        
#         # Horizontal line at y=0
#         ax.axhline(0, color='black', linewidth=0.6, zorder=1)
        
#         # Shade fit window (5-500s)
#         ax.axvspan(5, 500, color='lightgray', alpha=0.06, zorder=0)
        
#         # Vertical dashed line at 2000s
#         ax.axvline(2000, color='red', linewidth=1, linestyle='--', alpha=0.7, zorder=2)
        
#         # Mark crossover points
#         for t_cross, resid_val, R, H in crossover_points:
#             ax.plot(t_cross, resid_val, 'o', 
#                    markerfacecolor='white', markeredgecolor='blue',
#                    markeredgewidth=0.85, markersize=6, zorder=3)
        
#         # Set log x scale and limits
#         ax.set_xscale('log')
#         ax.set_xlim(t_min, t_max)
#         ax.set_ylim(-40, 40)
        
#         # Set title
#         ax.set_title(family, fontweight='bold', fontsize=11)
        
#         # ---- Data-driven annotations (simplified) ----
#         if not family_summary.empty and 't_cross' in family_summary.columns:
#             valid_summary = family_summary.dropna(subset=['t_cross'])
            
#             if valid_summary.empty:
#                 label = "[No departure ≤50,000 s]"
#             else:
#                 t_min = valid_summary['t_cross'].min()
#                 t_max = valid_summary['t_cross'].max()
#                 n_unique = valid_summary['t_cross'].nunique()
                
#                 if n_unique == 1:
#                     label = f"Departs ±5% at\nt ≈ {t_min:.0e} s"
#                 else:
#                     # Show range with configuration count
#                     n_configs = len(valid_summary)
#                     label = f"{n_configs} configs depart\n"
#                     label += f"  t range: {t_min:.0e}–{t_max:.0e} s"
#         else:
#             label = "[No departure data]"
    
#     # ---- Axis labels ----
#     # Y-axis labels for left column
#     for idx in [0, 4]:
#         axes[idx].set_ylabel('Residual, (n − k√t) / k√t (%)', fontsize=10)
    
#     # X-axis labels for bottom row
#     for idx in [4, 5, 6, 7]:
#         axes[idx].set_xlabel('Time, t (s)', fontsize=10)
    
#     # ---- Legend ----
#     # Create legend handles
#     legend_elements = []
    
#     # R handles (use first family's R values for consistent legend)
#     first_family = list(FAMILIES.keys())[0]
#     all_R = sorted(curves[curves['geometry'] == first_family]['R'].unique())
#     for r in all_R:
#         color = get_color(r, first_family)
#         legend_elements.append(
#             Line2D([0], [0], color=color, lw=2, 
#                    label=f'R = {r*1e6:.0f} µm')
#         )
    
#     # H handles (use first family's H values for consistent legend)
#     all_H = sorted(curves[curves['geometry'] == first_family]['H'].unique())
#     for h in all_H:
#         linestyle = get_linestyle(h, first_family)
#         legend_elements.append(
#             Line2D([0], [0], color='gray', lw=2, linestyle=linestyle,
#                    label=f'H = {h*1e6:.0f} µm')
#         )
    
#     # Reference handles
#     legend_elements.extend([
#         Line2D([0], [0], color='lightblue', lw=4, alpha=0.5, label='±5% band'),
#         Line2D([0], [0], color='red', lw=1, linestyle='--', label='2,000 s checkpoint'),
#         Line2D([0], [0], marker='o', color='w', markeredgecolor='blue',
#                markeredgewidth=0.85, markersize=6, label='persistent ±5% crossover')
#     ])
    
#     # Add legend
#     fig.legend(handles=legend_elements, ncol=5, loc='lower center',
#                bbox_to_anchor=(0.5, -0.02), fontsize=8, frameon=True)
    
#     # ---- Suptitle and subtitle ----
#     fig.suptitle('Figure 2. Residual departure from independent k√t fit',
#                  fontsize=14, fontweight='bold', y=0.98)
    
#     fig.text(0.5, 0.93, 
#              'Residual = (simulated − fit) / fit, as a percentage. '
#              'Annotated departure times indicate first sustained ±5% excursion. '
#              'Shaded band marks crossover threshold used for Figure 2b.',
#              ha='center', va='top', fontsize=9, style='italic')
    
#     # ---- Adjust layout ----
#     plt.tight_layout(rect=(0, 0.08, 1, 0.92))
    
#     return fig



# def _format_departure_time(t_cross):
#     """Format a crossover time compactly enough for a 5 by 5 heatmap cell."""
#     if not np.isfinite(t_cross):
#         return "—"
#     if t_cross >= 10_000:
#         return f"{t_cross / 1_000:.0f} k s"
#     return f"{t_cross:.0f} s"


# def _heatmap_text_colour(cmap, norm, value):
#     """Choose annotation text colour from the rendered cell colour."""
#     red, green, blue, _ = cmap(norm(value))
#     luminance = .2126 * red + .7152 * green + .0722 * blue
#     return "white" if luminance < .48 else "black"


# def _classification_heatmap(ax, table, cmap, norm, formatter, title):
#     """Render an annotated R-by-H grid with unobtrusive print-safe borders."""
#     image = ax.imshow(table.to_numpy(), cmap=cmap, norm=norm, aspect="equal")
#     ax.set_xticks(range(len(table.columns)), [f"{value:g}" for value in table.columns])
#     ax.set_yticks(range(len(table.index)), [f"{value:g}" for value in table.index])
#     ax.set_title(title, fontsize=10, pad=7)
#     ax.set_xticks(np.arange(-.5, len(table.columns), 1), minor=True)
#     ax.set_yticks(np.arange(-.5, len(table.index), 1), minor=True)
#     ax.grid(which="minor", color="#d6d6d6", linewidth=.7)
#     ax.tick_params(which="minor", bottom=False, left=False)

#     for row, H in enumerate(table.index):
#         for col, R in enumerate(table.columns):
#             value = table.loc[H, R]
#             colour = _heatmap_text_colour(cmap, norm, value) if np.isfinite(value) else "black"
#             ax.text(col, row, formatter(value), ha="center", va="center", fontsize=8,
#                     color=colour, weight="medium")
#     return image


# def build_figure_2_regime_diagnostics(summary):
#     """Build Figure 2 as a classification-first 2,000 s √t-regime check.

#     One row per family pairs the local exponent with its sustained ±5%
#     departure time.  The shared scales are deliberate: they make early
#     departures in the pillar families directly comparable to the safe,
#     green recess families.
#     """
#     with mpl.rc_context({"font.family": "serif",
#                          "font.serif": ["Times New Roman", "Times", "DejaVu Serif"]}):
#         fig, axes = plt.subplots(len(FAMILIES), 2, figsize=(10.8, 22),
#                                  sharex=True, sharey=True)
#         alpha_cmap = mpl.colors.LinearSegmentedColormap.from_list(
#             "regime_alpha", ["#b2182b", "#ef8a62", "#fee08b", "#66bd63", "#006837"]
#         )
#         # α=0.48 is the visual marginal point; α≈0.50 is compliant green.
#         alpha_norm = mpl.colors.TwoSlopeNorm(vmin=.42, vcenter=.48, vmax=.505)
#         # Blue means early/unsafe; green means late/safe at the anchor.
#         cross_cmap = mpl.colors.LinearSegmentedColormap.from_list(
#             "departure_time", ["#2166ac", "#67a9cf", "#f7fcb9", "#78c679", "#238b45"]
#         )
#         finite_crossings = summary.loc[np.isfinite(summary["t_cross"]), "t_cross"]
#         cross_norm = mpl.colors.LogNorm(vmin=min(500, finite_crossings.min()),
#                                         vmax=max(50_000, finite_crossings.max()))

#         alpha_image = cross_image = None
#         for row, family in enumerate(FAMILIES):
#             family_summary = summary[summary["geometry"] == family]
#             alpha_table = (family_summary.pivot(index="H_um", columns="R_um", values="alpha_anchor")
#                            .sort_index(ascending=False))
#             cross_table = (family_summary.pivot(index="H_um", columns="R_um", values="t_cross")
#                            .sort_index(ascending=False))
#             alpha_image = _classification_heatmap(
#                 axes[row, 0], alpha_table, alpha_cmap, alpha_norm, lambda value: f"{value:.3f}",
#                 r"Local exponent $\alpha$ at 2,000 s" if row == 0 else "",
#             )
#             cross_image = _classification_heatmap(
#                 axes[row, 1], cross_table, cross_cmap, cross_norm, _format_departure_time,
#                 r"Persistent ±5% departure time" if row == 0 else "",
#             )
#             axes[row, 0].set_ylabel(f"{family}\nHeight, H (µm)", fontsize=10, weight="bold")
#             axes[row, 0].tick_params(labelleft=True)
#             axes[row, 1].tick_params(labelleft=False)

#         for ax in axes[-1, :]:
#             ax.set_xlabel("Radius, R (µm)")

#         # Reserve the lower margin before adding shared colour bars.  Adjusting
#         # the axes afterward would leave the colour-bar positions stale and
#         # can overlap the final (Groove) row.
#         fig.subplots_adjust(top=.955, bottom=.12, hspace=.42, wspace=.12)
#         alpha_bar = fig.colorbar(alpha_image, ax=axes[:, 0], orientation="horizontal",
#                                  fraction=.035, pad=.035, aspect=38)
#         alpha_bar.set_label(r"$\alpha=d\ln n/d\ln t$  |  red: departed; green: compliant",
#                             fontsize=8, labelpad=3)
#         cross_bar = fig.colorbar(cross_image, ax=axes[:, 1], orientation="horizontal",
#                                  fraction=.035, pad=.035, aspect=38)
#         cross_bar.set_label("Departure time  |  blue: early/unsafe; green: late/safe",
#                             fontsize=8, labelpad=3)
#         cross_bar.set_ticks([500, 1_000, 2_000, 10_000, 50_000])
#         cross_bar.set_ticklabels(["500 s", "1 k s", "2 k s", "10 k s", "50 k s"])

#         fig.suptitle("Figure 2. Early-time regime diagnostics at t = 2,000 s",
#                      fontsize=16, weight="bold", y=.996)
#         fig.text(.5, .978,
#                  "Green α cells remain close to √t diffusion control. Blue departure-time cells leave the ±5% residual band before 2,000 s.",
#                  ha="center", fontsize=10)
#     return fig


# def main():
#     FIGURE_DIR.mkdir(exist_ok=True)
#     curves = load_curves()
#     summary = compute_summary(curves)
#     summary.to_csv(FIGURE_DIR / "Fig2_regime_metrics_t2000.csv", index=False)

#     # Figure 2 is classification-first; Figure 2b retains the detailed q-ratio view.
#     figures_to_save = [
#         (build_figure_2_regime_diagnostics(summary), "Fig2_early_time_regime_diagnostics"),
#         (build_figure_2b(summary), "Fig2b_regime_metrics_t2000"),
#     ]
    
#     for figure, stem in figures_to_save:
#         for suffix in ("png", "pdf"):
#             figure.savefig(FIGURE_DIR / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
#         plt.close(figure)

# if __name__ == "__main__":
#     main()

"""Build Figure 2 and Figure 2b: validity checks for the 2,000 s metric.

The figures use an early, independent 5--500 s fit to ``n = k sqrt(t)``.
The crossover is the first of three consecutive output points after the fit
window that lies outside a +/-5% band.  Figure 2 overlays each fitted curve
within its family; Figure 2b reports the two diagnostics at 2,000 s for every
configuration.

Run from the repository root:
    python plot_regime_check_figures.py
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import early_regime_fit as erf
import preprocessing as pp
from matplotlib.lines import Line2D


DATA_DIR = Path("data")
FIGURE_DIR = Path("figures")
ANCHOR_S = 2_000
FIT_WINDOW_S = (5, 500)

# Display name -> data file stem. Updated to match the current section
# naming: point recesses (cone/cylinder/square), extruded recesses
# (groove/edge), and additive pillars (square/cylindrical).
FAMILIES = {
    "Conical recess": "cone-recessed",
    "Cylindrical recess": "cylinder-recessed",
    "Square recess": "square-recessed",
    "Rectangular groove": "grooves",
    "Triangular edge recess": "edges",
    "Square pillar": "square-pillars",
    "Cylindrical pillar": "cylindrical-pillars",
}

# Category grouping used to order and label panels across figures. Every
# family in FAMILIES must appear in exactly one category list below --
# _ordered_families() checks this so a renamed/added family can't silently
# fall out of the grouped layouts.
CATEGORIES = {
    "Subtractive point recesses": ["Conical recess", "Cylindrical recess", "Square recess"],
    "Subtractive extruded recesses": ["Rectangular groove", "Triangular edge recess"],
    "Additive pillars": ["Square pillar", "Cylindrical pillar"],
}


def _ordered_families():
    """Flatten CATEGORIES into reading order and sanity-check against FAMILIES."""
    ordered = [family for members in CATEGORIES.values() for family in members]
    if set(ordered) != set(FAMILIES):
        missing = set(FAMILIES) - set(ordered)
        extra = set(ordered) - set(FAMILIES)
        raise ValueError(
            f"CATEGORIES and FAMILIES are out of sync. "
            f"In FAMILIES but not categorized: {missing or 'none'}. "
            f"In CATEGORIES but not in FAMILIES: {extra or 'none'}."
        )
    return ordered


def load_curves(data_dir=DATA_DIR):
    """Load every supplied uptake sweep into early_regime_fit's long format."""
    frames = []
    for family, stem in FAMILIES.items():
        entry = pp.build_family_entry((data_dir / f"{stem}-uptake.txt").read_text())
        series = {(family, key): value for key, value in entry["series"].items()}
        frames.append(erf.nested_dict_to_long_df(series, geometry_name=family))
    return pd.concat(frames, ignore_index=True)


def compute_summary(curves):
    """Return the per-configuration diagnostics used by both figures."""
    summary = erf.compute_regime_summary(
        curves,
        early_window_start_t=FIT_WINDOW_S[0],
        early_window_t=FIT_WINDOW_S[1],
        fit_method="median_ratio",
        tol=0.05,
        anchor_t=ANCHOR_S,
        anchor_bracket=(1_000, 3_000),
        persistent_points=3,
    )
    summary["R_um"] = summary["R"] * 1e6
    summary["H_um"] = summary["H"] * 1e6
    summary["departs_by_anchor"] = summary["t_cross"] <= ANCHOR_S
    return summary


def _config_style(R, H, radii, heights):
    """Encode radius by colour and height by line style consistently."""
    color = plt.get_cmap("viridis")(list(radii).index(R) / (len(radii) - 1))
    style = ["-", "--", ":", "-.", (0, (3, 1, 1, 1))][list(heights).index(H)]
    return color, style


def build_figure_2(curves, summary):
    """
    Figure 2.
    Residual departure from an independent early-time k√t fit.
    The emphasis is on WHEN the departure occurs rather than the
    exact residual trajectory. Panels are ordered by texture category
    (point recesses, extruded recesses, additive pillars) so families
    that behave alike sit next to each other.
    """

    ordered_families = _ordered_families()
    family_to_category = {f: cat for cat, members in CATEGORIES.items() for f in members}

    fig, axes = plt.subplots(
        2, 4,
        figsize=(16, 8.4),
        sharex=True,
        sharey=True,
    )

    axes = axes.ravel()

    radii = tuple(sorted(summary["R"].unique()))
    heights = tuple(sorted(summary["H"].unique()))

    summary_index = summary.set_index(["geometry", "R", "H"])

    for ax, family in zip(axes, ordered_families):

        subset = curves[curves["geometry"] == family]
        family_summary = summary[summary["geometry"] == family]

        # ----------------------------------------------------
        # Plot every configuration
        # ----------------------------------------------------

        for (R, H), config in subset.groupby(["R", "H"], sort=True):

            config = config.sort_values("t")

            row = summary_index.loc[(family, R, H)]

            color, linestyle = _config_style(
                R,
                H,
                radii,
                heights,
            )

            t = config["t"].to_numpy()
            n = config["n"].to_numpy()

            mask = t >= FIT_WINDOW_S[0]

            t = t[mask]
            n = n[mask]

            fit = row.k * np.sqrt(t)

            resid = 100 * (n - fit) / fit

            # ------------------------------------------------
            # Background residual curves
            # ------------------------------------------------

            ax.plot(
                t,
                resid,
                color=color,
                linestyle=linestyle,
                linewidth=0.9,
                alpha=0.35,
            )

            # ------------------------------------------------
            # Crossover marker
            # ------------------------------------------------

            if np.isfinite(row.t_cross):

                n_cross = erf.value_at_time(
                    config["t"].to_numpy(),
                    config["n"].to_numpy(),
                    row.t_cross,
                )

                fit_cross = row.k * np.sqrt(row.t_cross)

                resid_cross = (
                    (n_cross - fit_cross)
                    / fit_cross
                    * 100
                )

                # connector
                ax.vlines(
                    row.t_cross,
                    -5 if resid_cross < 0 else 5,
                    resid_cross,
                    color=color,
                    linewidth=0.9,
                    alpha=0.45,
                    zorder=5,
                )

                # marker
                ax.plot(
                    row.t_cross,
                    resid_cross,
                    marker="o",
                    markersize=5.5,
                    markerfacecolor="white",
                    markeredgecolor=color,
                    markeredgewidth=1.2,
                    linestyle="None",
                    zorder=8,
                )

        # ----------------------------------------------------
        # Family departure window
        # ----------------------------------------------------

        departures = family_summary["t_cross"].dropna()

        if len(departures):

            t0 = departures.min()
            t1 = departures.max()

            ax.axvspan(
                t0,
                t1,
                color="#f28e2b",
                alpha=0.08,
                zorder=0,
            )

            summary_text = (
                f"Departure window\n"
                f"{t0/1000:.1f}–{t1/1000:.1f} ks\n"
                f"median {departures.median()/1000:.1f} ks"
            )

            ax.text(
                0.03,
                0.97,
                summary_text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox=dict(
                    facecolor="white",
                    alpha=0.85,
                    edgecolor="0.8",
                    boxstyle="round,pad=0.3",
                ),
            )

        # ----------------------------------------------------
        # Reference guides
        # ----------------------------------------------------

        ax.axhspan(
            -5,
            5,
            color="#4c78a8",
            alpha=0.08,
        )

        ax.axhline(
            0,
            color="black",
            linewidth=0.6,
            alpha=0.7,
        )

        if subset["t"].min() <= ANCHOR_S:

            ax.axvline(
                ANCHOR_S,
                color="#c23b22",
                linestyle="--",
                linewidth=1.2,
            )

            ax.text(
                ANCHOR_S,
                23,
                "2000 s",
                rotation=90,
                color="#c23b22",
                fontsize=7,
                ha="center",
                va="bottom",
            )

        ax.set_xscale("log")
        ax.set_xlim(5, 50000)
        ax.set_ylim(-35, 25)

        ax.grid(
            True,
            which="major",
            linewidth=0.35,
            alpha=0.35,
        )

        # Panel title carries the category on the first family of each
        # group so the reader sees the grouping without a separate legend.
        category = family_to_category[family]
        is_first_in_category = family == CATEGORIES[category][0]
        title = f"{category}\n{family}" if is_first_in_category else family
        ax.set_title(
            title,
            fontsize=10 if is_first_in_category else 11,
            weight="bold",
        )

    # ----------------------------------------------------
    # Labels
    # ----------------------------------------------------

    for ax in axes[4:7]:
        ax.set_xlabel("Time, $t$ (s)")

    for ax in axes[::4]:
        ax.set_ylabel(
            r"Residual, $(n-k\sqrt{t})/(k\sqrt{t})$ (%)"
        )

    axes[-1].axis("off")

    # ----------------------------------------------------
    # Legend
    # ----------------------------------------------------

    radius_handles = [
        mpl.lines.Line2D(
            [],
            [],
            color=_config_style(
                R,
                heights[0],
                radii,
                heights,
            )[0],
            lw=2,
            label=f"R = {R*1e6:g} μm",
        )
        for R in radii
    ]

    height_handles = [
        mpl.lines.Line2D(
            [],
            [],
            color="black",
            linestyle=_config_style(
                radii[0],
                H,
                radii,
                heights,
            )[1],
            lw=1.5,
            label=f"H = {H*1e6:g} μm",
        )
        for H in heights
    ]

    reference_handles = [

        mpl.patches.Patch(
            color="#4c78a8",
            alpha=0.15,
            label="±5% band",
        ),

        mpl.patches.Patch(
            color="#f28e2b",
            alpha=0.15,
            label="departure window",
        ),

        mpl.lines.Line2D(
            [],
            [],
            color="#c23b22",
            linestyle="--",
            lw=1.2,
            label="2000 s checkpoint",
        ),

        mpl.lines.Line2D(
            [],
            [],
            marker="o",
            markerfacecolor="white",
            markeredgecolor="black",
            lw=0,
            markersize=6,
            label="persistent ±5% departure",
        ),
    ]

    fig.legend(
        handles=(
            radius_handles
            + height_handles
            + reference_handles
        ),
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        fontsize=8,
    )

    fig.suptitle(
        "Figure 2. Departure from the independent $k\\sqrt{t}$ diffusion regime",
        fontsize=15,
        y=0.99,
    )

    fig.text(
        0.5,
        0.935,
        "Residual curves are shown in the background. Open circles indicate the first persistent "
        "departure beyond ±5%, while the shaded region summarises the departure-time window "
        "for each geometry family.",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=(0, 0.09, 1, 0.93))

    return fig


def _heatmap(ax, table, cmap, norm, fmt, title, fail_mask=None):
    image = ax.imshow(table.to_numpy(), cmap=cmap, norm=norm, aspect="equal")
    ax.set_xticks(range(len(table.columns)), [f"{value:g}" for value in table.columns])
    ax.set_yticks(range(len(table.index)), [f"{value:g}" for value in table.index])
    ax.set_title(title, fontsize=9, pad=5)
    for row, H in enumerate(table.index):
        for col, R in enumerate(table.columns):
            value = table.loc[H, R]
            ax.text(col, row, fmt.format(value), ha="center", va="center", fontsize=7,
                    color="black" if norm(value) > .35 else "white")
            if fail_mask is not None and fail_mask.loc[H, R]:
                ax.add_patch(plt.Rectangle((col - .5, row - .5), 1, 1, fill=False,
                                           edgecolor="#b2182b", linewidth=1.55))
    return image


def build_figure_2b(summary):
    """Facet q_ratio and local exponent heatmaps by geometry family."""
    ordered_families = _ordered_families()

    fig, axes = plt.subplots(4, 4, figsize=(15.5, 13), layout="constrained")
    q_norm = mpl.colors.Normalize(vmin=.89, vmax=1.00)
    alpha_norm = mpl.colors.Normalize(vmin=.42, vmax=.50)
    q_cmap = plt.get_cmap("RdYlGn")
    alpha_cmap = plt.get_cmap("RdYlGn")
    images = []

    for pair_index, family in enumerate(ordered_families):
        row, first_col = divmod(pair_index, 2)
        first_col *= 2
        family_summary = summary[summary["geometry"] == family]
        q_table = family_summary.pivot(index="H_um", columns="R_um", values="q_ratio_anchor").sort_index(ascending=False)
        alpha_table = family_summary.pivot(index="H_um", columns="R_um", values="alpha_anchor").sort_index(ascending=False)
        bad_q = q_table < .95
        bad_alpha = alpha_table < .475
        images.append(_heatmap(axes[row, first_col], q_table, q_cmap, q_norm, "{:.3f}", r"$q_{ratio}$", bad_q))
        images.append(_heatmap(axes[row, first_col + 1], alpha_table, alpha_cmap, alpha_norm, "{:.3f}", r"local $\alpha$", bad_alpha))
        axes[row, first_col].set_ylabel("H (µm)")
        axes[row, first_col + 1].set_yticklabels([])
        axes[row, first_col].set_xlabel("R (µm)")
        axes[row, first_col + 1].set_xlabel("R (µm)")
        axes[row, first_col].annotate(family, xy=(1, 1), xycoords="axes fraction",
                                      xytext=(0, 18), textcoords="offset points",
                                      ha="center", va="bottom", fontsize=11, weight="bold")

    axes[3, 2].axis("off")
    axes[3, 3].axis("off")
    q_bar = fig.colorbar(images[0], ax=[axes[r, c] for r in range(4) for c in (0, 2)], shrink=.72, pad=.015)
    q_bar.set_label(r"$q_{ratio}=n(t)/[k\sqrt{t}]$  (ideal: 1)")
    alpha_bar = fig.colorbar(images[1], ax=[axes[r, c] for r in range(4) for c in (1, 3)], shrink=.72, pad=.015)
    alpha_bar.set_label(r"local $alpha=dln n/dln t$  (ideal: 0.5)")
    fig.suptitle("Figure 2b. √t-regime diagnostics at t = 2,000 s", fontsize=15)
    fig.text(.5, .012, "Red borders flag q_ratio < 0.950 or α < 0.475; both metrics use the same independent 5–500 s fit.",
             ha="center", fontsize=9)
    return fig


def _format_departure_time(t_cross):
    """Format a crossover time compactly enough for a 5 by 5 heatmap cell."""
    if not np.isfinite(t_cross):
        return "—"
    if t_cross >= 10_000:
        return f"{t_cross / 1_000:.0f} k s"
    return f"{t_cross:.0f} s"


def _heatmap_text_colour(cmap, norm, value):
    """Choose annotation text colour from the rendered cell colour."""
    red, green, blue, _ = cmap(norm(value))
    luminance = .2126 * red + .7152 * green + .0722 * blue
    return "white" if luminance < .48 else "black"


def _classification_heatmap(ax, table, cmap, norm, formatter, title):
    """Render an annotated R-by-H grid with unobtrusive print-safe borders."""
    image = ax.imshow(table.to_numpy(), cmap=cmap, norm=norm, aspect="equal")
    ax.set_xticks(range(len(table.columns)), [f"{value:g}" for value in table.columns])
    ax.set_yticks(range(len(table.index)), [f"{value:g}" for value in table.index])
    ax.set_title(title, fontsize=9, pad=6)
    ax.set_xticks(np.arange(-.5, len(table.columns), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(table.index), 1), minor=True)
    ax.grid(which="minor", color="#d6d6d6", linewidth=.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row, H in enumerate(table.index):
        for col, R in enumerate(table.columns):
            value = table.loc[H, R]
            colour = _heatmap_text_colour(cmap, norm, value) if np.isfinite(value) else "black"
            ax.text(col, row, formatter(value), ha="center", va="center", fontsize=7,
                    color=colour, weight="medium")
    return image


def build_figure_2_regime_diagnostics(summary):
    """Build Figure 2 as a classification-first 2,000 s √t-regime check.

    Two families share each row (alpha | departure time | alpha | departure
    time) instead of stacking all seven families vertically, so the whole
    figure fits a normal page instead of a 10.8 x 22 in column. Panels are
    grouped and labelled by texture category (subtractive point recesses,
    subtractive extruded recesses, additive pillars) so families that
    behave alike are visually adjacent.
    """
    ordered_families = _ordered_families()
    n_rows = -(-len(ordered_families) // 2)  # ceil division

    with mpl.rc_context({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "DejaVu Serif"]}):
        fig, axes = plt.subplots(n_rows, 4, figsize=(13, 3.15 * n_rows),
                                 sharex=True, sharey=True)
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        alpha_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "regime_alpha", ["#b2182b", "#ef8a62", "#fee08b", "#66bd63", "#006837"]
        )
        # α=0.48 is the visual marginal point; α≈0.50 is compliant green.
        alpha_norm = mpl.colors.TwoSlopeNorm(vmin=.42, vcenter=.48, vmax=.505)
        # Blue means early/unsafe; green means late/safe at the anchor.
        cross_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "departure_time", ["#2166ac", "#67a9cf", "#f7fcb9", "#78c679", "#238b45"]
        )
        finite_crossings = summary.loc[np.isfinite(summary["t_cross"]), "t_cross"]
        cross_norm = mpl.colors.LogNorm(vmin=min(500, finite_crossings.min()),
                                        vmax=max(50_000, finite_crossings.max()))

        # Slot each family into (row, pair) reading order, two per row.
        slot_of = {family: divmod(i, 2) for i, family in enumerate(ordered_families)}

        alpha_image = cross_image = None
        for family in ordered_families:
            row, pair = slot_of[family]
            col0 = pair * 2
            family_summary = summary[summary["geometry"] == family]
            alpha_table = (family_summary.pivot(index="H_um", columns="R_um", values="alpha_anchor")
                           .sort_index(ascending=False))
            cross_table = (family_summary.pivot(index="H_um", columns="R_um", values="t_cross")
                           .sort_index(ascending=False))
            alpha_image = _classification_heatmap(
                axes[row, col0], alpha_table, alpha_cmap, alpha_norm, lambda value: f"{value:.3f}",
                r"Local exponent $\alpha$ at 2,000 s" if row == 0 else "",
            )
            cross_image = _classification_heatmap(
                axes[row, col0 + 1], cross_table, cross_cmap, cross_norm, _format_departure_time,
                r"Persistent ±5% departure time" if row == 0 else "",
            )
            axes[row, col0].set_ylabel(f"{family}\nH (µm)", fontsize=8.5, weight="bold")
            axes[row, col0].tick_params(labelleft=True)
            axes[row, col0 + 1].tick_params(labelleft=False)

        # Blank any unused trailing slot (odd number of families).
        if len(ordered_families) % 2:
            last_row = (len(ordered_families) - 1) // 2
            axes[last_row, 2].axis("off")
            axes[last_row, 3].axis("off")

        for ax in axes[-1, :]:
            ax.set_xlabel("R (µm)")

        # Category divider: a light rule plus bold label above the row on
        # which each category starts. Reserve top margin before adding the
        # shared colour bars so their positions don't go stale afterward.
        fig.subplots_adjust(top=.90, bottom=.11, hspace=.62, wspace=.16)
        for category, members in CATEGORIES.items():
            first_row = slot_of[members[0]][0]
            anchor_ax = axes[first_row, 0]
            fig.canvas.draw()
            pos = anchor_ax.get_position()
            if first_row > 0:
                fig.add_artist(Line2D(
                    [0.045, 0.97], [pos.y1 + .028, pos.y1 + .028],
                    transform=fig.transFigure, color="#bbbbbb", linewidth=.9,
                ))
            fig.text(0.045, pos.y1 + .033, category,
                     fontsize=11, weight="bold", color="#333333", ha="left", va="bottom")

        alpha_bar = fig.colorbar(alpha_image, ax=axes[:, [0, 2]], orientation="horizontal",
                                 fraction=.03, pad=.06, aspect=48)
        alpha_bar.set_label(r"$\alpha=d\ln n/d\ln t$  |  red: departed; green: compliant",
                            fontsize=8, labelpad=3)
        cross_bar = fig.colorbar(cross_image, ax=axes[:, [1, 3]], orientation="horizontal",
                                 fraction=.03, pad=.06, aspect=48)
        cross_bar.set_label("Departure time  |  blue: early/unsafe; green: late/safe",
                            fontsize=8, labelpad=3)
        cross_bar.set_ticks([500, 1_000, 2_000, 10_000, 50_000])
        cross_bar.set_ticklabels(["500 s", "1 k s", "2 k s", "10 k s", "50 k s"])

        fig.suptitle("Figure 2. Early-time regime diagnostics at t = 2,000 s",
                     fontsize=15, weight="bold", y=.985)
        fig.text(.5, .935,
                 "Green α cells remain close to √t diffusion control. Blue departure-time cells leave the ±5% residual band before 2,000 s.",
                 ha="center", fontsize=9)
    return fig


def build_figure_2_report_ready(summary):
    """Create the clean, one-family-per-row report version of Figure 2."""
    ordered_families = _ordered_families()
    with mpl.rc_context({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "DejaVu Serif"]}):
        fig, axes = plt.subplots(len(ordered_families), 2, figsize=(8.6, 17.6),
                                 sharex=True, sharey=True)
        alpha_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "report_regime_alpha", ["#b2182b", "#ef8a62", "#fee08b", "#66bd63", "#006837"]
        )
        alpha_norm = mpl.colors.TwoSlopeNorm(vmin=.42, vcenter=.48, vmax=.505)
        cross_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "report_departure_time", ["#2166ac", "#67a9cf", "#f7fcb9", "#78c679", "#238b45"]
        )
        finite_crossings = summary.loc[np.isfinite(summary["t_cross"]), "t_cross"]
        cross_norm = mpl.colors.LogNorm(vmin=min(500, finite_crossings.min()),
                                        vmax=max(50_000, finite_crossings.max()))

        alpha_image = cross_image = None
        for row, family in enumerate(ordered_families):
            family_summary = summary[summary["geometry"] == family]
            alpha_table = (family_summary.pivot(index="H_um", columns="R_um", values="alpha_anchor")
                           .sort_index(ascending=False))
            cross_table = (family_summary.pivot(index="H_um", columns="R_um", values="t_cross")
                           .sort_index(ascending=False))
            alpha_image = _classification_heatmap(
                axes[row, 0], alpha_table, alpha_cmap, alpha_norm, lambda value: f"{value:.3f}",
                r"Local exponent $\alpha$ at 2,000 s" if row == 0 else "",
            )
            cross_image = _classification_heatmap(
                axes[row, 1], cross_table, cross_cmap, cross_norm, _format_departure_time,
                r"Persistent +/-5% departure time" if row == 0 else "",
            )
            axes[row, 0].set_ylabel(f"{family}\nH (um)", fontsize=9, weight="bold")
            axes[row, 0].tick_params(labelleft=True)
            axes[row, 1].tick_params(labelleft=False)

        for ax in axes[-1, :]:
            ax.set_xlabel("R (um)")

        # The colour bars live in their own bottom band, outside all panels.
        # This prevents the overlap seen when Matplotlib tries to infer their
        # location from a dense grid of shared axes.
        fig.subplots_adjust(left=.19, right=.96, top=.915, bottom=.135,
                            hspace=.48, wspace=.42)
        alpha_cax = fig.add_axes([.19, .052, .32, .012])
        cross_cax = fig.add_axes([.64, .052, .32, .012])
        alpha_bar = fig.colorbar(alpha_image, cax=alpha_cax, orientation="horizontal")
        alpha_bar.set_label(r"$\alpha=d\ln n/d\ln t$  |  red: departed; green: compliant",
                            fontsize=8, labelpad=3)
        cross_bar = fig.colorbar(cross_image, cax=cross_cax, orientation="horizontal")
        cross_bar.set_label("Departure time  |  blue: early/unsafe; green: late/safe",
                            fontsize=8, labelpad=3)
        cross_bar.set_ticks([500, 1_000, 2_000, 10_000, 50_000])
        cross_bar.set_ticklabels(["500 s", "1 k s", "2 k s", "10 k s", "50 k s"])

    return fig


def build_figure_2_compact_report(summary):
    """Create the compact report layout without competing category headings."""
    ordered_families = _ordered_families()
    n_rows = -(-len(ordered_families) // 2)
    with mpl.rc_context({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "DejaVu Serif"]}):
        fig, axes = plt.subplots(n_rows, 4, figsize=(12.8, 12.8),
                                 sharex=True, sharey=True)
        alpha_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "compact_regime_alpha", ["#b2182b", "#ef8a62", "#fee08b", "#66bd63", "#006837"]
        )
        alpha_norm = mpl.colors.TwoSlopeNorm(vmin=.42, vcenter=.48, vmax=.505)
        cross_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "compact_departure_time", ["#2166ac", "#67a9cf", "#f7fcb9", "#78c679", "#238b45"]
        )
        finite_crossings = summary.loc[np.isfinite(summary["t_cross"]), "t_cross"]
        cross_norm = mpl.colors.LogNorm(vmin=min(500, finite_crossings.min()),
                                        vmax=max(50_000, finite_crossings.max()))
        slots = {family: divmod(index, 2) for index, family in enumerate(ordered_families)}

        alpha_image = cross_image = None
        for family in ordered_families:
            row, pair = slots[family]
            first_col = pair * 2
            family_summary = summary[summary["geometry"] == family]
            alpha_table = (family_summary.pivot(index="H_um", columns="R_um", values="alpha_anchor")
                           .sort_index(ascending=False))
            cross_table = (family_summary.pivot(index="H_um", columns="R_um", values="t_cross")
                           .sort_index(ascending=False))
            alpha_image = _classification_heatmap(
                axes[row, first_col], alpha_table, alpha_cmap, alpha_norm,
                lambda value: f"{value:.3f}", r"Local exponent $\alpha$ at 2,000 s" if row == 0 else "",
            )
            cross_image = _classification_heatmap(
                axes[row, first_col + 1], cross_table, cross_cmap, cross_norm,
                _format_departure_time, r"Persistent +/-5% departure time" if row == 0 else "",
            )
            axes[row, first_col].set_ylabel(f"{family}\nH (um)", fontsize=8.5, weight="bold")
            axes[row, first_col].tick_params(labelleft=True)
            axes[row, first_col + 1].tick_params(labelleft=False)

        if len(ordered_families) % 2:
            axes[-1, 2].axis("off")
            axes[-1, 3].axis("off")
        for ax in axes[-1, :]:
            if ax.axison:
                ax.set_xlabel("R (um)")

        # A reserved lower band avoids colour-bar/tick-label collisions.
        fig.subplots_adjust(left=.105, right=.97, top=.89, bottom=.13,
                            hspace=.62, wspace=.27)
        alpha_cax = fig.add_axes([.16, .052, .30, .014])
        cross_cax = fig.add_axes([.62, .052, .30, .014])
        alpha_bar = fig.colorbar(alpha_image, cax=alpha_cax, orientation="horizontal")
        alpha_bar.set_label(r"$\alpha=d\ln n/d\ln t$  |  red: departed; green: compliant",
                            fontsize=8, labelpad=3)
        cross_bar = fig.colorbar(cross_image, cax=cross_cax, orientation="horizontal")
        cross_bar.set_label("Departure time  |  blue: early/unsafe; green: late/safe",
                            fontsize=8, labelpad=3)
        cross_bar.set_ticks([500, 1_000, 2_000, 10_000, 50_000])
        cross_bar.set_ticklabels(["500 s", "1 k s", "2 k s", "10 k s", "50 k s"])

    return fig


def main():
    FIGURE_DIR.mkdir(exist_ok=True)
    curves = load_curves()
    summary = compute_summary(curves)
    summary.to_csv(FIGURE_DIR / "Fig2_regime_metrics_t2000.csv", index=False)

    # Figure 2 is classification-first; Figure 2b retains the detailed q-ratio view.
    figures_to_save = [
        (build_figure_2_compact_report(summary), "Fig2_early_time_regime_diagnostics"),
        (build_figure_2b(summary), "Fig2b_regime_metrics_t2000"),
    ]

    for figure, stem in figures_to_save:
        for suffix in ("png", "pdf"):
            figure.savefig(FIGURE_DIR / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    main()
