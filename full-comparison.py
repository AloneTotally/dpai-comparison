"""
CO2 uptake validation study: full comparison
  - 1x1 (original) : fine mesh, treated as the accuracy REFERENCE
  - 1x1 (rerun)     : coarser mesh, re-solved after a disk-error/temp-file
                       issue. Matches the reference at equilibrium (<0.5%
                       diff from t=200s onward) but is NOT reliable for
                       early-time (t<100s) transient shape - shown for
                       context only, not used as the % deviation reference.
  - 2x2 / 4x4 / 6x6 : repeated-cell arrays, read directly from file,
                       raw "full solid" totals divided by cell count.

Geometry: square_recessed, R = 80 um, H = 35 um (stress-test case)
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Generic COMSOL table parser (last two whitespace columns = time, value)
# ---------------------------------------------------------------------------
def parse_comsol_table(path):
    times, values = [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            times.append(float(parts[-2]))
            values.append(float(parts[-1]))
    return np.array(times), np.array(values)


def dedup(t, v):
    _, first_idx = np.unique(t, return_index=True)
    order = np.sort(first_idx)
    return t[order], v[order]


# ---------------------------------------------------------------------------
# Load array files directly (raw = full-solid total; divide by cell count)
# ---------------------------------------------------------------------------
ARRAY_FILES = {
    "2x2": ("/mnt/user-data/uploads/2x2-unitcell.txt", 4),
    "4x4": ("/mnt/user-data/uploads/4x4-unitcell.txt", 16),
    "6x6": ("/mnt/user-data/uploads/6x6-unitcell.txt", 36),
}
data = {}
for label, (path, ncells) in ARRAY_FILES.items():
    t, v_raw = parse_comsol_table(path)
    t, v_raw = dedup(t, v_raw)
    data[label] = {"t": t, "v_pc": v_raw / ncells, "ncells": ncells}

# ---------------------------------------------------------------------------
# 1x1 original (fine mesh) - accuracy reference
# ---------------------------------------------------------------------------
t_1x1 = np.array([
    0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,
    5,10,100,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,
    7000,8000,9000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,
    110000,120000,130000,140000,150000,160000,170000,180000,190000,200000
])
v_1x1_ref = np.array([
    0,0,0,0,0,0,0,0,0,0,
    1.4768848503428773e-6,2.743367408882311e-6,2.7672275461419993e-6,
    2.790962969096993e-6,2.8145840705220164e-6,2.8380781401382526e-6,
    2.861423848391611e-6,2.884690327831875e-6,2.907877741441394e-6,
    2.930932556323173e-6,2.9537248809265937e-6,
    3.5995863318808934e-6,4.536797238758763e-6,
    1.3349017748564066e-5,1.3349017748564066e-5,1.8857227093232585e-5,
    2.3035322282282234e-5,2.6526891950998195e-5,2.9581221288093457e-5,
    3.2329279045385026e-5,3.484377449699324e-5,3.717052661967135e-5,
    3.9343517502033006e-5,4.1389449612175845e-5,
    5.728128069707663e-5,6.794903275618646e-5,7.541242124301404e-5,
    8.074839315827326e-5,8.466200876267099e-5,8.759170409070707e-5,
    8.984509121870374e-5,9.159120387650715e-5,9.29461666531229e-5,
    9.721607442059349e-5,9.762421485814678e-5,9.765335565429568e-5,
    9.765196250269627e-5,9.764499725326857e-5,9.764206473423971e-5,
    9.764091026453948e-5,9.764042548476082e-5,9.764023593693898e-5,
    9.764015843383777e-5,9.764013032443884e-5,9.764011660861122e-5,
    9.764010923800966e-5,9.764010593264632e-5,9.764010441939152e-5,
    9.764010371884709e-5,9.764010337657559e-5,9.764010322016993e-5,
    9.764010314570092e-5
])
t_1x1_r, v_1x1_ref_r = dedup(t_1x1, v_1x1_ref)
data["1x1_ref"] = {"t": t_1x1_r, "v_pc": v_1x1_ref_r, "ncells": 1}

# ---------------------------------------------------------------------------
# 1x1 rerun (coarser mesh, post disk-error) - shown for context only
# ---------------------------------------------------------------------------
t_1x1r, v_1x1_rerun = parse_comsol_table("/mnt/user-data/uploads/1x1-unitcell.txt")
t_1x1r, v_1x1_rerun = dedup(t_1x1r, v_1x1_rerun)
data["1x1_rerun"] = {"t": t_1x1r, "v_pc": v_1x1_rerun, "ncells": 1}

colors = {"1x1_ref": "black", "1x1_rerun": "gray",
          "2x2": "#1f77b4", "4x4": "#ff7f0e", "6x6": "#2ca02c"}
labels_disp = {"1x1_ref": "1x1 (reference, fine mesh)",
               "1x1_rerun": "1x1 (rerun, coarse mesh)",
               "2x2": "2x2", "4x4": "4x4", "6x6": "6x6"}

# ---------------------------------------------------------------------------
# FIGURE 2: per-cell uptake vs time, all 5 curves
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))
style = {
    "1x1_ref":   dict(ls="-",  lw=2.5, ms=3, marker="o", zorder=5),
    "1x1_rerun": dict(ls=":",  lw=1.5, ms=3, marker="x", zorder=4),
    "2x2":       dict(ls="-",  lw=1.5, ms=3, marker="o", zorder=3),
    "4x4":       dict(ls="--", lw=1.5, ms=6, marker="o", zorder=6,
                       markerfacecolor="none"),
    "6x6":       dict(ls="-",  lw=1.5, ms=3, marker="o", zorder=2),
}
for label in ["1x1_ref", "1x1_rerun", "2x2", "4x4", "6x6"]:
    d = data[label]
    m = d["t"] > 0
    ax.semilogx(d["t"][m], d["v_pc"][m], color=colors[label],
                label=labels_disp[label], **style[label])
ax.set_xlabel("Time (s)")
ax.set_ylabel("Uptake per unit cell (mol)")
ax.set_title("Per-unit-cell uptake vs time: all models")
ax.legend(fontsize=9)
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("/home/claude/fig2_full_comparison.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# FIGURE 3: % deviation from 1x1 REFERENCE (fine mesh) vs time
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))
ref_t, ref_v = data["1x1_ref"]["t"], data["1x1_ref"]["v_pc"]
for label in ["1x1_rerun", "2x2", "4x4", "6x6"]:
    d = data[label]
    common = np.array(sorted(set(ref_t) & set(d["t"])))
    common = common[common > 0]
    ref_at = ref_v[np.searchsorted(ref_t, common)]
    val_at = d["v_pc"][np.searchsorted(d["t"], common)]
    valid = ref_at != 0
    pct = 100 * (val_at[valid] - ref_at[valid]) / ref_at[valid]
    ax.semilogx(common[valid], pct, marker="o", ms=3, lw=1.5,
                color=colors[label], label=labels_disp[label])
ax.axhline(0, color="k", lw=1, ls="--")
ax.set_xlabel("Time (s)")
ax.set_ylabel("% deviation from 1x1 reference (fine mesh)")
ax.set_title("Relative deviation from the fine-mesh 1x1 reference")
ax.legend(fontsize=9)
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("/home/claude/fig3_full_deviation.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# FIGURE 4: equilibrium per-cell uptake vs array size
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
order_lab = ["1x1_ref", "1x1_rerun", "2x2", "4x4", "6x6"]
eq_vals = [data[l]["v_pc"][-1] for l in order_lab]
x_pos = np.arange(len(order_lab))
ax.bar(x_pos, eq_vals, color=[colors[l] for l in order_lab])
ax.set_xticks(x_pos)
ax.set_xticklabels([labels_disp[l] for l in order_lab], rotation=20, ha="right")
ax.set_ylabel("Equilibrium uptake per unit cell (mol)")
ax.set_title("Equilibrium per-cell uptake across all models")
for xp, val in zip(x_pos, eq_vals):
    ax.text(xp, val, f"{val:.3e}", ha="center", va="bottom", fontsize=8)
fig.tight_layout()
fig.savefig("/home/claude/fig4_full_equilibrium.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print(f"{'Model':<26}{'Cells':>7}{'Eq. per-cell uptake (mol)':>28}{'% vs 1x1 ref':>16}")
ref_eq = data["1x1_ref"]["v_pc"][-1]
for label in order_lab:
    val = data[label]["v_pc"][-1]
    pct = 100 * (val - ref_eq) / ref_eq
    print(f"{labels_disp[label]:<26}{data[label]['ncells']:>7}{val:>28.6e}{pct:>15.2f}%")
