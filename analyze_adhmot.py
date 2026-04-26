"""
analyze_adhMot.py
=================
Analysis script for the adhesion+motility coupled parameter sweep.

Sweep design:
  In this sweep, adhesion and motility are varied inversely:
    adhesion_value = 0.4 × multiplier  (baseline = 0.4 µm/min)
    motility_value = 0.4 / multiplier  (baseline = 0.4 µm/min)

  Multipliers tested: 0.1×, 0.25×, 1× (baseline), 2×, 5×, 10×
  This means:
    - 0.1× → low adhesion, high motility (adhesive=0.04, speed=4.0)
    - 0.25× → low adhesion, moderate motility
    - 1× → baseline (adhesion=0.4, speed=0.4)
    - 2× → moderate adhesion, low motility
    - 10× → high adhesion, very low motility (adhesion=4.0, speed=0.04)

  2D: 10 seeds per condition (mean ± SD)
  3D: 1 seed per condition (single data point — compute limited)

Output folders:
  2D: output_adhMot{MULT}_{2D_normal|2D_ADT}_seed{00-09}/
  3D: output_adhMot{MULT}_{3D_normal|3D_ADT}_seed00/
  Baseline (1x): output_{2D_normal|2D_ADT|3D_normal|3D_ADT}_seed{00-09}/

Produces:
  figures/adhMot_summary_15months.png
  figures/adhMot_temporal_{cell_counts,clustering}_{normal,ADT}.png
  figures/adhMot_violins_15months.png
  figures/adhMot_clustering_summary_15months.png  (sigmoid fit)
  csv_exports/adhMot_cell_counts_temporal.csv
  csv_exports/adhMot_clustering_temporal.csv

Run from: ~/ProstateCancer-ABM/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore")

os.makedirs("figures", exist_ok=True)
os.makedirs("csv_exports", exist_ok=True)

# ── Simulation constants ──────────────────────────────────────────────────────
MIN_PER_MONTH  = 30 * 24 * 60
FINAL_TIME_MIN = 15 * MIN_PER_MONTH
N_SEEDS        = 10

# Multipliers and their string representations in folder names
# "baseline" maps to the standard output_2D_normal_seed* folders
ADHESION_MULTS = [0.1, 0.25, 1.0, 2.0, 5.0, 10.0]
MULT_STRS      = ["0p1", "0p25", "baseline", "2p0", "5p0", "10p0"]

# Actual adhesion and motility values for each multiplier
ADHESION_VALS  = [round(0.4 * m, 4) for m in ADHESION_MULTS]
MOTILITY_VALS  = [round(0.4 / m, 4) for m in ADHESION_MULTS]

# X-axis tick labels with adhesion/motility direction indicators
MULT_LABELS = ["0.1x\nadh↓ mot↑", "0.25x\nadh↓ mot↑", "1x\nbaseline",
               "2x\nadh↑ mot↓", "5x\nadh↑ mot↓", "10x\nadh↑ mot↓"]

# Condition style definitions
CONDITIONS_2D = {
    "2D_normal": {"label": "2D Normal androgen", "color": "#1F77B4", "ls": "-"},
    "2D_ADT":    {"label": "2D ADT",             "color": "#FF7F0E", "ls": "-"},
}
CONDITIONS_3D = {
    "3D_normal": {"label": "3D Normal androgen (n=1)", "color": "#1F77B4", "ls": "--"},
    "3D_ADT":    {"label": "3D ADT (n=1)",             "color": "#FF7F0E", "ls": "--"},
}

# Metrics to analyze
METRIC_COLS   = ["total", "S", "R", "RS"]
METRIC_LABELS = ["Total cell count", "Sensitive (S)", "Resistant (R)", "R / S ratio"]
CLUST_COLS    = ["C_avg_all", "C_avg_PTEN_normal", "C_avg_PTEN_deleted"]
CLUST_LABELS  = ["Clustering index (all)", "Clustering index (S)", "Clustering index (R)"]

# Color palette for multiplier levels
cmap   = plt.cm.viridis
COLORS = [cmap(i/5) for i in range(6)]

# ── Helper functions ──────────────────────────────────────────────────────────

def load_csv(path):
    """
    Load a single analysis_over_time.csv, add derived columns, and return DataFrame.
    Returns None if the file does not exist.
    """
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset="time_min", keep="last").sort_values("time_min").reset_index(drop=True)
    df["S"]     = df["Alive_PTEN_Normal"]
    df["R"]     = df["Alive_PTEN_Deleted"]
    df["total"] = df["S"] + df["R"]
    df["RS"]    = df["R"] / df["S"].replace(0, np.nan)
    df["month"] = df["time_min"] / MIN_PER_MONTH
    return df

def load_2d(mult_str, cond):
    """
    Load all 10 seed CSVs for a 2D condition at a given multiplier.

    Special case: mult_str == "baseline" → uses the standard output_{cond}_seed* folders
    rather than output_adhMot*_* folders.
    """
    dfs = []
    for seed in range(N_SEEDS):
        if mult_str == "baseline":
            path = f"output_{cond}_seed{seed:02d}/analysis_over_time.csv"
        else:
            path = f"output_adhMot{mult_str}_{cond}_seed{seed:02d}/analysis_over_time.csv"
        df = load_csv(path)
        if df is not None: dfs.append(df)
    return dfs

def load_3d(mult_str, cond):
    """
    Load the single-seed 3D CSV for a given multiplier and condition.
    Returns None if the file does not exist.
    """
    if mult_str == "baseline":
        path = f"output_{cond}_seed00/analysis_over_time.csv"
    else:
        path = f"output_adhMot{mult_str}_{cond}_seed00/analysis_over_time.csv"
    return load_csv(path)

def compute_stats(dfs, col):
    """
    Compute mean and standard deviation of col across seed DataFrames.
    Re-indexes each seed onto the first seed's time axis to handle
    minor timing differences.
    Returns (mean_array, sd_array) or (None, None) if dfs is empty.
    """
    if not dfs: return None, None
    ref = dfs[0]["time_min"].values
    arr = []
    for df in dfs:
        idx = df.set_index("time_min")
        idx = idx[~idx.index.duplicated(keep="last")]
        arr.append(idx.reindex(ref).interpolate(method="index")[col].values)
    arr = np.array(arr, dtype=float)
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)

def get_final(dfs_or_df, col):
    """
    Get the value of col at t=15 months.

    Accepts either:
      - a list of DataFrames (returns list of scalars, one per seed)
      - a single DataFrame (returns a single scalar)
    NaN values are excluded from the list form.
    """
    if dfs_or_df is None: return []
    if isinstance(dfs_or_df, list):
        vals = []
        for df in dfs_or_df:
            idx = np.argmin(np.abs(df["time_min"].values - FINAL_TIME_MIN))
            v = float(df[col].iloc[idx])
            if not np.isnan(v): vals.append(v)
        return vals
    else:
        df = dfs_or_df
        idx = np.argmin(np.abs(df["time_min"].values - FINAL_TIME_MIN))
        return float(df[col].iloc[idx]) if col in df.columns else np.nan

def pval_label(p):
    """Convert p-value to significance annotation string."""
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else: return "ns"

def add_bracket(ax, x1, x2, y, h, label, fontsize=7):
    """Draw a significance bracket with label on a matplotlib axis."""
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.0, color="black")
    ax.text((x1+x2)/2, y+h*1.2, label, ha="center", va="bottom", fontsize=fontsize,
            fontweight="bold" if label != "ns" else "normal")

def sigmoid(x, L, k, x0, b):
    """
    4-parameter logistic (sigmoid) function.

    Parameters:
      L  — upper asymptote (ceiling)
      k  — steepness (growth rate)
      x0 — x value at midpoint (inflection point)
      b  — lower asymptote (floor)

    Returns: b + (L - b) / (1 + exp(-k * (x - x0)))
    """
    return b + (L - b) / (1 + np.exp(-k * (x - x0)))

def fit_sigmoid(x_vals, y_vals):
    """
    Fit a sigmoid to (x_vals, y_vals) using scipy curve_fit.

    Returns (x_dense, y_fit) for plotting, or (None, None) if fit fails.
    x_dense is a 200-point array spanning the input x range.
    """
    try:
        p0 = [max(y_vals), 1.0, np.median(x_vals), min(y_vals)]
        bounds = ([0, 0, min(x_vals)-2, 0], [1.5, 10, max(x_vals)+2, 1.0])
        popt, _ = curve_fit(sigmoid, x_vals, y_vals, p0=p0,
                            bounds=bounds, maxfev=5000)
        x_dense = np.linspace(min(x_vals), max(x_vals), 200)
        return x_dense, sigmoid(x_dense, *popt)
    except Exception:
        return None, None

# ── Load all data ─────────────────────────────────────────────────────────────
print("Loading data...")
data_2d = {}  # (mult_str, cond) → list of DataFrames
data_3d = {}  # (mult_str, cond) → single DataFrame or None

for mult_str in MULT_STRS:
    for cond in CONDITIONS_2D:
        data_2d[(mult_str, cond)] = load_2d(mult_str, cond)
        n = len(data_2d[(mult_str, cond)])
        print(f"  2D {mult_str} {cond}: {n} seeds")
    for cond in CONDITIONS_3D:
        data_3d[(mult_str, cond)] = load_3d(mult_str, cond)
        ok = "OK" if data_3d[(mult_str, cond)] is not None else "MISSING"
        print(f"  3D {mult_str} {cond}: {ok}")

x = np.arange(len(ADHESION_MULTS))   # integer x positions for bar/violin plots

# ── Plot 1: Summary at t=15 months ────────────────────────────────────────────
print("\nGenerating summary plot...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Adhesion+Motility Sweep — BR Cohort at t = 15 months\n"
             "(adhesion × mult, motility ÷ mult)", fontsize=13, fontweight="bold")

for ax, col, lbl in zip(axes.flat, METRIC_COLS, METRIC_LABELS):
    # 2D lines with ±SD shading
    for cond, info in CONDITIONS_2D.items():
        means = [np.mean(get_final(data_2d[(m, cond)], col)) if get_final(data_2d[(m, cond)], col) else np.nan for m in MULT_STRS]
        sds   = [np.std(get_final(data_2d[(m, cond)], col))  if get_final(data_2d[(m, cond)], col) else np.nan for m in MULT_STRS]
        means, sds = np.array(means), np.array(sds)
        ax.plot(x, means, color=info["color"], ls=info["ls"], lw=2, marker="o", label=info["label"])
        ax.fill_between(x, means-sds, means+sds, alpha=0.12, color=info["color"])
    # 3D single points (dashed)
    for cond, info in CONDITIONS_3D.items():
        vals = [get_final(data_3d[(m, cond)], col) for m in MULT_STRS]
        vals = [v if not isinstance(v, list) else (np.mean(v) if v else np.nan) for v in vals]
        ax.plot(x, vals, color=info["color"], ls=info["ls"], lw=2, marker="s", markersize=8, label=info["label"])

    ax.set_xticks(x); ax.set_xticklabels(MULT_LABELS, fontsize=7)
    ax.set_xlabel("Adhesion/Motility multiplier"); ax.set_ylabel(lbl)
    ax.set_title(lbl, fontweight="bold")
    ax.grid(True, alpha=0.3, ls=":")
    # Dashed vertical line at baseline (index 2)
    ax.axvline(x=2, color="grey", ls=":", lw=1.5, alpha=0.6)
    if col == "RS": ax.set_ylim(bottom=0)

handles = [Line2D([0],[0], color=v["color"], ls=v["ls"], lw=2,
                  marker="o" if "2D" in k else "s", label=v["label"])
           for k, v in {**CONDITIONS_2D, **CONDITIONS_3D}.items()]
fig.legend(handles=handles, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, -0.02), fontsize=9)
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.savefig("figures/adhMot_summary_15months.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/adhMot_summary_15months.png")

# ── Plot 2: Temporal plots ────────────────────────────────────────────────────
print("Generating temporal plots...")
for treat_2d, treat_3d, treat_str, treat_label in [
    ("2D_normal", "3D_normal", "normal", "Normal androgen (8 ng/ml)"),
    ("2D_ADT",    "3D_ADT",    "ADT",    "ADT (1 ng/ml)"),
]:
    for section, cols, labels in [
        ("cell_counts", METRIC_COLS, METRIC_LABELS),
        ("clustering",  CLUST_COLS,  CLUST_LABELS),
    ]:
        fig, axes = plt.subplots(1, len(cols), figsize=(6*len(cols), 6))
        if len(cols) == 1: axes = [axes]
        fig.suptitle(f"Adhesion+Motility Sweep — {section.replace('_',' ').title()} — {treat_label}",
                     fontsize=12, fontweight="bold")

        for ax, col, lbl in zip(axes, cols, labels):
            for i, (mult_str, color) in enumerate(zip(MULT_STRS, COLORS)):
                # 2D: mean ± SD shading
                dfs = data_2d[(mult_str, treat_2d)]
                if dfs:
                    ref = dfs[0]["time_min"].values
                    mean, sd = compute_stats(dfs, col)
                    if mean is not None:
                        ax.plot(ref/MIN_PER_MONTH, mean, color=color, lw=2,
                                label=MULT_LABELS[i].split("\n")[0])
                        ax.fill_between(ref/MIN_PER_MONTH, mean-sd, mean+sd,
                                        alpha=0.12, color=color)
                # 3D: single seed dashed line
                df3 = data_3d[(mult_str, treat_3d)]
                if df3 is not None and col in df3.columns:
                    ax.plot(df3["month"], df3[col], color=color, lw=1.5, ls="--", alpha=0.85)

            ax.set_xlabel("Time (months)"); ax.set_ylabel(lbl)
            ax.set_title(lbl, fontweight="bold"); ax.set_xlim(0, 15)
            ax.grid(True, alpha=0.3, ls=":")
            if col in ("RS",) + tuple(CLUST_COLS): ax.set_ylim(bottom=0)

        legend_els = [Line2D([0],[0], color=COLORS[i], lw=2,
                             label=MULT_LABELS[i].split("\n")[0]) for i in range(6)] + [
            Line2D([0],[0], color="grey", lw=2,   ls="-",  label="2D (mean±SD)"),
            Line2D([0],[0], color="grey", lw=1.5, ls="--", label="3D (n=1)"),
        ]
        fig.legend(handles=legend_els, loc="lower center", ncol=4,
                   bbox_to_anchor=(0.5, -0.06), fontsize=8)
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        fname = f"figures/adhMot_temporal_{section}_{treat_str}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fname}")

# ── Plot 3: Violin plots at t=15 months ───────────────────────────────────────
print("Generating violin plots with significance brackets...")
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle("Adhesion+Motility Sweep — Distribution at t = 15 months (BR Cohort)",
             fontsize=13, fontweight="bold")

width = 0.35
for ax, col, lbl in zip(axes.flat, METRIC_COLS, METRIC_LABELS):
    for offset, (cond_2d, cond_3d, color) in enumerate([
        ("2D_normal", "3D_normal", "#1F77B4"),
        ("2D_ADT",    "3D_ADT",    "#FF7F0E"),
    ]):
        positions = x + (offset - 0.5) * width
        all_vals_2d = [get_final(data_2d[(m, cond_2d)], col) for m in MULT_STRS]

        # Violin plots for 2D distributions (require >1 point)
        valid = [(pos, v) for pos, v in zip(positions, all_vals_2d) if len(v) > 1]
        if valid:
            vp = ax.violinplot([v[1] for v in valid], positions=[v[0] for v in valid],
                               widths=width*0.8, showmeans=True, showmedians=True)
            for pc in vp["bodies"]:
                pc.set_facecolor(color); pc.set_alpha(0.45); pc.set_edgecolor("black")
            vp["cmeans"].set_color("black"); vp["cmedians"].set_color("white")
            for comp in ["cbars","cmaxes","cmins"]: vp[comp].set_color("grey")
            for pos, vals in valid:
                jitter = np.random.uniform(-width*0.2, width*0.2, len(vals))
                ax.scatter([pos+j for j in jitter], vals, color=color,
                           edgecolors="black", s=25, zorder=5, alpha=0.8, lw=0.5)

        # Star markers for 3D single seed values
        for pos, mult_str in zip(positions, MULT_STRS):
            df3 = data_3d[(mult_str, cond_3d)]
            v = get_final(df3, col)
            if not isinstance(v, float) or not np.isnan(v):
                ax.scatter([pos], [v], color=color, marker="*", s=150,
                           edgecolors="black", zorder=6, lw=0.8)

    # Significance brackets: Mann-Whitney U, normal vs. ADT per multiplier
    y_max = ax.get_ylim()[1]
    y_range = y_max - ax.get_ylim()[0]
    bh = y_range * 0.03
    for xi, mult_str in enumerate(MULT_STRS):
        vn = get_final(data_2d[(mult_str, "2D_normal")], col)
        va = get_final(data_2d[(mult_str, "2D_ADT")],    col)
        if len(vn) >= 2 and len(va) >= 2:
            _, p = stats.mannwhitneyu(vn, va, alternative="two-sided")
            add_bracket(ax, xi-0.5*width, xi+0.5*width,
                        y_max + y_range*0.05, bh, pval_label(p), fontsize=7)

    ax.set_ylim(top=y_max + y_range*0.25)
    ax.set_xticks(x); ax.set_xticklabels(MULT_LABELS, fontsize=7)
    ax.set_xlabel("Adhesion/Motility multiplier"); ax.set_ylabel(lbl)
    ax.set_title(lbl, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y", ls=":")
    if col == "RS": ax.set_ylim(bottom=0)
    ax.axvline(x=2, color="grey", ls=":", lw=1.5, alpha=0.6)

legend_els = [
    Patch(facecolor="#1F77B4", alpha=0.5, label="Normal androgen 2D (violin=10 seeds)"),
    Patch(facecolor="#FF7F0E", alpha=0.5, label="ADT 2D (violin=10 seeds)"),
    Line2D([0],[0], marker="*", color="w", markerfacecolor="#1F77B4", markersize=12, label="Normal androgen 3D (n=1)"),
    Line2D([0],[0], marker="*", color="w", markerfacecolor="#FF7F0E", markersize=12, label="ADT 3D (n=1)"),
]
fig.legend(handles=legend_els, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, -0.02), fontsize=9)
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig("figures/adhMot_violins_15months.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/adhMot_violins_15months.png")

# ── Plot 4: Clustering index sigmoid summary ──────────────────────────────────
print("Generating clustering vs. adhesion sigmoid plot...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Adhesion+Motility Sweep — Clustering Index at t = 15 months\n"
             "Points = data (mean±SD for 2D, single point for 3D), curves = sigmoid fit",
             fontsize=13, fontweight="bold")

# Use actual adhesion strength values on the x-axis (log scale) so the
# sigmoid fit reflects the biological relationship, not arbitrary indices
x_vals = np.array(ADHESION_VALS)

for ax, col, lbl in zip(axes, CLUST_COLS, CLUST_LABELS):
    # 2D: scatter with error bars + sigmoid fit
    for cond, info in CONDITIONS_2D.items():
        means, sds = [], []
        for mult_str in MULT_STRS:
            dfs = data_2d[(mult_str, cond)]
            vals = [get_final(df, col) for df in dfs if df is not None and col in df.columns]
            vals = [v for v in vals if not np.isnan(v)]
            means.append(np.mean(vals) if vals else np.nan)
            sds.append(np.std(vals)    if vals else np.nan)
        means, sds = np.array(means), np.array(sds)
        # Data points with error bars
        ax.errorbar(x_vals, means, yerr=sds, fmt="o", color=info["color"],
                    markersize=7, capsize=4, lw=1.2, zorder=5,
                    markeredgecolor="black", label=f"{info['label']} (data)")
        # Sigmoid fit to non-NaN means
        valid = ~np.isnan(means)
        x_fit, y_fit = fit_sigmoid(x_vals[valid], means[valid])
        if x_fit is not None:
            ax.plot(x_fit, y_fit, color=info["color"], ls="-", lw=2.5, alpha=0.85)

    # 3D: scatter + optional sigmoid
    for cond, info in CONDITIONS_3D.items():
        vals_3d = [get_final(data_3d[(m, cond)], col) for m in MULT_STRS]
        vals_3d = np.array([v if not isinstance(v, list) else np.nan for v in vals_3d])
        ax.scatter(x_vals, vals_3d, color=info["color"], marker="s",
                   s=70, zorder=5, edgecolors="black", lw=0.8, label=f"{info['label']} (n=1)")
        valid = ~np.isnan(vals_3d)
        if valid.sum() >= 4:
            x_fit, y_fit = fit_sigmoid(x_vals[valid], vals_3d[valid])
            if x_fit is not None:
                ax.plot(x_fit, y_fit, color=info["color"], ls="--", lw=2.0, alpha=0.85)

    ax.set_xscale("log")   # log scale appropriate since adhesion spans 2 orders of magnitude
    ax.set_xlabel("Adhesion strength (µm/min, log scale)", fontsize=10)
    ax.set_ylabel(lbl, fontsize=10)
    ax.set_title(lbl, fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, ls=":")
    # Vertical dotted line marking the baseline adhesion value (0.4 µm/min)
    ax.axvline(x=0.4, color="grey", ls=":", lw=1.5, alpha=0.6, label="Baseline (0.4)")

handles = [Line2D([0],[0], color=v["color"], ls=v["ls"], lw=2,
                  marker="o" if "2D" in k else "s", label=v["label"])
           for k, v in {**CONDITIONS_2D, **CONDITIONS_3D}.items()]
fig.legend(handles=handles, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, -0.04), fontsize=9)
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("figures/adhMot_clustering_summary_15months.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/adhMot_clustering_summary_15months.png")

# ── Export master temporal CSVs ───────────────────────────────────────────────
print("\nExporting master temporal CSVs...")
cc_rows, cl_rows = [], []

for mult_str in MULT_STRS:
    for cond_2d, cond_3d, treat in [
        ("2D_normal", "3D_normal", "normal_androgen"),
        ("2D_ADT",    "3D_ADT",    "ADT"),
    ]:
        # Process both 2D (list of DFs) and 3D (single DF) in one loop
        for dim, dfs_or_df in [("2D", data_2d[(mult_str, cond_2d)]),
                                ("3D", data_3d[(mult_str, cond_3d)])]:
            dfs = dfs_or_df if isinstance(dfs_or_df, list) else ([dfs_or_df] if dfs_or_df is not None else [])
            if not dfs: continue
            ref = dfs[0]["time_min"].values

            for col, col_label, target in [
                ("total","Total_cells",cc_rows),("S","Sensitive_S",cc_rows),
                ("R","Resistant_R",cc_rows),("RS","R_S_ratio",cc_rows),
                ("C_avg_all","C_avg_all",cl_rows),
                ("C_avg_PTEN_normal","C_avg_S",cl_rows),
                ("C_avg_PTEN_deleted","C_avg_R",cl_rows),
            ]:
                if col not in dfs[0].columns: continue
                # For multi-seed: compute mean and SD; for single seed: SD is NaN
                if len(dfs) > 1:
                    mean, sd = compute_stats(dfs, col)
                else:
                    mean = dfs[0][col].values; sd = np.full_like(mean, np.nan)
                if mean is None: continue
                mult_idx = MULT_STRS.index(mult_str)
                for t, m, s in zip(ref, mean, sd):
                    target.append({
                        "adhMot_mult":   mult_str,
                        "adhesion_val":  ADHESION_VALS[mult_idx],
                        "motility_val":  MOTILITY_VALS[mult_idx],
                        "treatment":     treat,
                        "dimension":     dim,
                        "n_seeds":       len(dfs),
                        "time_min":      t,
                        "month":         t / MIN_PER_MONTH,
                        "metric":        col_label,
                        "mean":          m,
                        "sd":            s,
                    })

pd.DataFrame(cc_rows).to_csv("csv_exports/adhMot_cell_counts_temporal.csv", index=False)
pd.DataFrame(cl_rows).to_csv("csv_exports/adhMot_clustering_temporal.csv", index=False)
print("  Saved: csv_exports/adhMot_cell_counts_temporal.csv")
print("  Saved: csv_exports/adhMot_clustering_temporal.csv")
print("\nDone.")
