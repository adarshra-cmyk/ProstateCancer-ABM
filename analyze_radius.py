"""
analyze_radius.py
=================
Analysis script for the initial tumor radius parameter sweep.

Sweep design:
  - Radii tested: 50, 100, 150, 200, 250 µm
  - Dimensions: 2D (10 seeds/condition) and 3D (10 seeds/condition)
  - Conditions: normal androgen (8 ng/ml) and ADT (1 ng/ml)
  - Total: 5 radii × 2 dimensions × 2 conditions × 10 seeds = 200 runs

Output folders follow the naming convention:
  output_r{RADIUS}_{2D|3D}_{normal|ADT}_seed{00-09}/

Produces:
  figures/radius_summary_15months.png    — all metrics at t=15mo vs. radius
  figures/radius_temporal_*.png          — time series per metric and treatment
  figures/radius_violins_15months.png    — violin + p-value plots at t=15mo
  csv_exports/radius_cell_counts_temporal.csv
  csv_exports/radius_clustering_temporal.csv

Run from: ~/ProstateCancer-ABM/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')           # non-interactive backend for HPC
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Output directories ────────────────────────────────────────────────────────
os.makedirs("figures", exist_ok=True)
os.makedirs("csv_exports", exist_ok=True)

# ── Simulation constants ──────────────────────────────────────────────────────
MIN_PER_MONTH  = 30 * 24 * 60   # minutes in a 30-day month
FINAL_TIME_MIN = 15 * MIN_PER_MONTH  # t = 15 months in minutes
N_SEEDS        = 10              # seeds per 2D condition

# Radius values swept and their display labels
RADII       = [50, 100, 150, 200, 250]
RADII_LABEL = ["50 µm", "100 µm", "150 µm", "200 µm", "250 µm"]

# Condition definitions: display label, line color, line style
CONDITIONS = {
    "2D_normal": {"label": "2D Normal androgen", "color": "#1F77B4", "ls": "-"},
    "2D_ADT":    {"label": "2D ADT",             "color": "#FF7F0E", "ls": "-"},
    "3D_normal": {"label": "3D Normal androgen", "color": "#1F77B4", "ls": "--"},
    "3D_ADT":    {"label": "3D ADT",             "color": "#FF7F0E", "ls": "--"},
}

# Cell count metrics derived from analysis_over_time.csv
METRIC_COLS   = ["total", "S", "R", "RS"]
METRIC_LABELS = ["Total cell count", "Sensitive (S)", "Resistant (R)", "R / S ratio"]

# Clustering index columns from analysis_over_time.csv
CLUST_COLS    = ["C_avg_all", "C_avg_PTEN_normal", "C_avg_PTEN_deleted"]
CLUST_LABELS  = ["Clustering index (all)", "Clustering index (S)", "Clustering index (R)"]

# Color palette for the 5 radius values (viridis colormap)
cmap   = plt.cm.viridis
COLORS = [cmap(i/4) for i in range(5)]

# ── Helper functions ──────────────────────────────────────────────────────────

def load_csv(path):
    """
    Load a single analysis_over_time.csv and add derived columns.

    Derived columns added:
      S     — alive sensitive cell count (from Alive_PTEN_Normal)
      R     — alive resistant cell count (from Alive_PTEN_Deleted)
      total — S + R
      RS    — R / S ratio (NaN when S = 0)
      month — time in months (time_min / MIN_PER_MONTH)

    Duplicate time points are removed (keeping the last occurrence),
    and the DataFrame is sorted by time_min.
    """
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset="time_min", keep="last").sort_values("time_min").reset_index(drop=True)
    df["S"]     = df["Alive_PTEN_Normal"]
    df["R"]     = df["Alive_PTEN_Deleted"]
    df["total"] = df["S"] + df["R"]
    df["RS"]    = df["R"] / df["S"].replace(0, np.nan)
    df["month"] = df["time_min"] / MIN_PER_MONTH
    return df

def load_all_seeds(r, cond):
    """
    Load all seed CSVs for a given radius and condition.

    Parameters:
      r    — radius in µm (e.g. 50, 100)
      cond — condition string, e.g. "2D_normal" or "3D_ADT"

    Returns:
      list of DataFrames, one per successfully loaded seed.
      Missing seeds are silently skipped.
    """
    dfs = []
    for seed in range(N_SEEDS):
        df = load_csv(f"output_r{r}_{cond}_seed{seed:02d}/analysis_over_time.csv")
        if df is not None:
            dfs.append(df)
    return dfs

def compute_stats(dfs, col):
    """
    Compute mean and standard deviation of a column across multiple seed DataFrames.

    Uses the time axis of the first DataFrame as the reference grid.
    Each seed's values are re-indexed and interpolated onto this grid
    to handle minor timing differences between seeds.

    Parameters:
      dfs — list of DataFrames (one per seed)
      col — column name to aggregate

    Returns:
      (mean, sd) as numpy arrays aligned to the reference time axis,
      or (None, None) if dfs is empty.
    """
    if not dfs:
        return None, None
    ref = dfs[0]["time_min"].values
    arr = []
    for df in dfs:
        indexed = df.set_index("time_min")
        indexed = indexed[~indexed.index.duplicated(keep="last")]
        arr.append(indexed.reindex(ref).interpolate(method="index")[col].values)
    arr = np.array(arr, dtype=float)
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)

def get_final(dfs, col):
    """
    Extract the value of a column at the final timepoint (t = 15 months)
    from each seed DataFrame.

    The closest available timepoint to FINAL_TIME_MIN is used.

    Returns:
      list of scalar values, one per seed, excluding NaN values.
    """
    vals = []
    for df in dfs:
        idx = np.argmin(np.abs(df["time_min"].values - FINAL_TIME_MIN))
        v = float(df[col].iloc[idx])
        if not np.isnan(v):
            vals.append(v)
    return vals

def pval_label(p):
    """Convert a p-value to a significance string for plot annotations."""
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else: return "ns"

def add_bracket(ax, x1, x2, y, h, label, fontsize=7):
    """
    Draw a significance bracket between two positions on a matplotlib axis.

    Parameters:
      ax    — matplotlib Axes object
      x1,x2 — x positions of bracket ends
      y     — y position of the horizontal bar
      h     — height of the bracket ticks
      label — text label (e.g. "***", "ns")
    """
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.0, color="black")
    ax.text((x1+x2)/2, y+h*1.2, label, ha="center", va="bottom", fontsize=fontsize,
            fontweight="bold" if label != "ns" else "normal")

# ── Load all data ─────────────────────────────────────────────────────────────
print("Loading data...")
# data[(radius, condition)] = list of DataFrames (one per seed)
data = {}
for r in RADII:
    for cond in CONDITIONS:
        dfs = load_all_seeds(r, cond)
        data[(r, cond)] = dfs
        print(f"  r={r} {cond}: {len(dfs)} seeds loaded")

x = np.arange(len(RADII))   # integer x positions for plotting

# ── Plot 1: Summary at t=15 months vs. radius ─────────────────────────────────
print("\nGenerating summary plot (all metrics at t=15 months)...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Tumor Radius Sweep — BR Cohort at t = 15 months", fontsize=13, fontweight="bold")

for ax, col, lbl in zip(axes.flat, METRIC_COLS, METRIC_LABELS):
    for cond, info in CONDITIONS.items():
        # Compute mean ± SD at t=15 months across seeds for each radius
        means, sds = [], []
        for r in RADII:
            vals = get_final(data[(r, cond)], col)
            means.append(np.mean(vals) if vals else np.nan)
            sds.append(np.std(vals) if vals else np.nan)
        means, sds = np.array(means), np.array(sds)
        marker = "o" if "2D" in cond else "s"
        ax.plot(x, means, color=info["color"], ls=info["ls"], lw=2,
                marker=marker, label=info["label"])
        ax.fill_between(x, means-sds, means+sds, alpha=0.12, color=info["color"])

    ax.set_xticks(x)
    ax.set_xticklabels(RADII_LABEL, fontsize=9)
    ax.set_xlabel("Initial tumor radius")
    ax.set_ylabel(lbl)
    ax.set_title(lbl, fontweight="bold")
    ax.grid(True, alpha=0.3, ls=":")
    if col == "RS": ax.set_ylim(bottom=0)

handles = [Line2D([0],[0], color=v["color"], ls=v["ls"], lw=2,
                  marker="o" if "2D" in k else "s", label=v["label"])
           for k, v in CONDITIONS.items()]
fig.legend(handles=handles, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, -0.02), fontsize=9)
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.savefig("figures/radius_summary_15months.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/radius_summary_15months.png")

# ── Plot 2: Temporal plots (cell counts and clustering) ───────────────────────
print("Generating temporal plots...")
for treat_2d, treat_3d, treat_str, treat_label in [
    ("2D_normal", "3D_normal", "normal", "Normal androgen (8 ng/ml)"),
    ("2D_ADT",    "3D_ADT",    "ADT",    "ADT (1 ng/ml)"),
]:
    for section, cols, labels in [
        ("cell_counts", METRIC_COLS,  METRIC_LABELS),
        ("clustering",  CLUST_COLS,   CLUST_LABELS),
    ]:
        fig, axes = plt.subplots(1, len(cols), figsize=(6*len(cols), 6))
        if len(cols) == 1: axes = [axes]
        fig.suptitle(f"Radius Sweep — {section.replace('_',' ').title()} — {treat_label}",
                     fontsize=12, fontweight="bold")

        for ax, col, lbl in zip(axes, cols, labels):
            for i, (r, color) in enumerate(zip(RADII, COLORS)):
                # 2D: solid line with ±SD shading
                dfs_2d = data[(r, treat_2d)]
                if dfs_2d:
                    ref = dfs_2d[0]["time_min"].values
                    mean, sd = compute_stats(dfs_2d, col)
                    if mean is not None:
                        months = ref / MIN_PER_MONTH
                        ax.plot(months, mean, color=color, lw=2, label=f"r={r} µm")
                        ax.fill_between(months, mean-sd, mean+sd, alpha=0.12, color=color)
                # 3D: dashed line (10 seeds averaged)
                dfs_3d = data[(r, treat_3d)]
                if dfs_3d:
                    ref = dfs_3d[0]["time_min"].values
                    mean3, _ = compute_stats(dfs_3d, col)
                    if mean3 is not None:
                        ax.plot(ref/MIN_PER_MONTH, mean3, color=color, lw=1.5, ls="--", alpha=0.85)

            ax.set_xlabel("Time (months)"); ax.set_ylabel(lbl)
            ax.set_title(lbl, fontweight="bold"); ax.set_xlim(0, 15)
            ax.grid(True, alpha=0.3, ls=":")
            if col in ("RS",) + tuple(CLUST_COLS): ax.set_ylim(bottom=0)

        legend_els = [Line2D([0],[0], color=COLORS[i], lw=2,
                             label=f"r={RADII[i]} µm") for i in range(5)] + [
            Line2D([0],[0], color="grey", lw=2,   ls="-",  label="2D (mean±SD)"),
            Line2D([0],[0], color="grey", lw=1.5, ls="--", label="3D (mean±SD)"),
        ]
        fig.legend(handles=legend_els, loc="lower center", ncol=4,
                   bbox_to_anchor=(0.5, -0.06), fontsize=8)
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        fname = f"figures/radius_temporal_{section}_{treat_str}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fname}")

# ── Plot 3: Violin plots at t=15 months with significance brackets ─────────────
print("Generating violin plots with p-values...")
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle("Radius Sweep — Distribution at t = 15 months (BR Cohort)",
             fontsize=13, fontweight="bold")

width = 0.35
for ax, col, lbl in zip(axes.flat, METRIC_COLS, METRIC_LABELS):
    for offset, (cond_2d, cond_3d, color, clabel) in enumerate([
        ("2D_normal", "3D_normal", "#1F77B4", "Normal androgen"),
        ("2D_ADT",    "3D_ADT",    "#FF7F0E", "ADT"),
    ]):
        positions = x + (offset - 0.5) * width

        # Gather final-timepoint values per radius for 2D violin and 3D error bar
        all_vals_2d = [get_final(data[(r, cond_2d)], col) for r in RADII]
        all_vals_3d = [get_final(data[(r, cond_3d)], col) for r in RADII]

        # Draw 2D violin plots (require >1 data point per violin)
        valid = [(pos, v) for pos, v in zip(positions, all_vals_2d) if len(v) > 1]
        if valid:
            vp = ax.violinplot([v[1] for v in valid], positions=[v[0] for v in valid],
                               widths=width*0.8, showmeans=True, showmedians=True)
            for pc in vp["bodies"]:
                pc.set_facecolor(color); pc.set_alpha(0.45); pc.set_edgecolor("black")
            vp["cmeans"].set_color("black"); vp["cmedians"].set_color("white")
            for comp in ["cbars","cmaxes","cmins"]: vp[comp].set_color("grey")
            # Overlay individual seed points with jitter
            for pos, vals in valid:
                jitter = np.random.uniform(-width*0.2, width*0.2, len(vals))
                ax.scatter([pos+j for j in jitter], vals, color=color,
                           edgecolors="black", s=25, zorder=5, alpha=0.8, lw=0.5)

        # Draw 3D mean ± SD as error bars (square marker)
        for pos, vals in zip(positions, all_vals_3d):
            if vals:
                m, s = np.mean(vals), np.std(vals)
                ax.errorbar(pos, m, yerr=s, fmt="s", color=color,
                            markersize=8, capsize=4, zorder=6,
                            markeredgecolor="black", ecolor="black", lw=1.2)

    # Significance brackets: Mann-Whitney U test, normal androgen vs. ADT
    y_max = ax.get_ylim()[1]
    y_range = y_max - ax.get_ylim()[0]
    bh = y_range * 0.03
    for xi, r in enumerate(RADII):
        vn = get_final(data[(r, "2D_normal")], col)
        va = get_final(data[(r, "2D_ADT")], col)
        if len(vn) >= 2 and len(va) >= 2:
            _, p = stats.mannwhitneyu(vn, va, alternative="two-sided")
            add_bracket(ax, xi-0.5*width, xi+0.5*width,
                        y_max + y_range*0.05, bh, pval_label(p), fontsize=7)

    ax.set_ylim(top=y_max + y_range*0.25)
    ax.set_xticks(x); ax.set_xticklabels(RADII_LABEL, fontsize=9)
    ax.set_xlabel("Initial tumor radius"); ax.set_ylabel(lbl)
    ax.set_title(lbl, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y", ls=":")
    if col == "RS": ax.set_ylim(bottom=0)

legend_els = [
    Patch(facecolor="#1F77B4", alpha=0.5, label="Normal androgen 2D (violin=10 seeds)"),
    Patch(facecolor="#FF7F0E", alpha=0.5, label="ADT 2D (violin=10 seeds)"),
    Line2D([0],[0], marker="s", color="w", markerfacecolor="#1F77B4",
           markersize=10, label="Normal androgen 3D (mean±SD)"),
    Line2D([0],[0], marker="s", color="w", markerfacecolor="#FF7F0E",
           markersize=10, label="ADT 3D (mean±SD)"),
]
fig.legend(handles=legend_els, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, -0.02), fontsize=9)
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig("figures/radius_violins_15months.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/radius_violins_15months.png")

# ── Export master CSVs ────────────────────────────────────────────────────────
print("\nExporting master temporal CSVs...")
cc_rows, cl_rows = [], []

for r in RADII:
    for cond in CONDITIONS:
        dfs = data[(r, cond)]
        if not dfs:
            continue
        ref = dfs[0]["time_min"].values
        treat = "normal_androgen" if "normal" in cond else "ADT"
        dim   = "2D" if "2D" in cond else "3D"

        for col, col_label, target in [
            ("total", "Total_cells",   cc_rows),
            ("S",     "Sensitive_S",   cc_rows),
            ("R",     "Resistant_R",   cc_rows),
            ("RS",    "R_S_ratio",     cc_rows),
            ("C_avg_all",          "C_avg_all", cl_rows),
            ("C_avg_PTEN_normal",  "C_avg_S",   cl_rows),
            ("C_avg_PTEN_deleted", "C_avg_R",   cl_rows),
        ]:
            if col not in dfs[0].columns:
                continue
            mean, sd = compute_stats(dfs, col)
            if mean is None:
                continue
            for t, m, s in zip(ref, mean, sd):
                target.append({
                    "tumor_radius_um": r,
                    "treatment":       treat,
                    "dimension":       dim,
                    "n_seeds":         len(dfs),
                    "time_min":        t,
                    "month":           t / MIN_PER_MONTH,
                    "metric":          col_label,
                    "mean":            m,
                    "sd":              s,
                })

pd.DataFrame(cc_rows).to_csv("csv_exports/radius_cell_counts_temporal.csv", index=False)
pd.DataFrame(cl_rows).to_csv("csv_exports/radius_clustering_temporal.csv", index=False)
print("  Saved: csv_exports/radius_cell_counts_temporal.csv")
print("  Saved: csv_exports/radius_clustering_temporal.csv")
print("\nDone.")
