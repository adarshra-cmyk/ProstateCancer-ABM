"""
analyze_adhesion.py
===================
Analysis script for the cell-cell adhesion parameter sweep.

Sweep design:
  Only adhesion is varied here (motility is held at baseline 0.4 µm/min).
  Adhesion multipliers: 0.1×, 0.25×, 1× (baseline), 2×, 5×, 10×
  Absolute adhesion values (µm/min): 0.04, 0.10, 0.40, 0.80, 2.00, 4.00

  2D: 10 seeds per adhesion × condition (mean ± SD)
  3D: 1 seed per adhesion × condition (single data point — compute limited)

Output folders:
  2D: output_adh{MULT}_{2D_normal|2D_ADT}_seed{00-09}/
  3D: output_adh{MULT}_{3D_normal|3D_ADT}_seed00/
  Baseline (1×): output_{2D_normal|2D_ADT|3D_normal|3D_ADT}_seed{00-09}/

Produces:
  figures/adhesion_summary_15months.png
  figures/adhesion_temporal_{col}.png          — per metric, normal vs. ADT
  figures/adhesion_temporal_{section}_{cond}.png — per section per treatment
  figures/adhesion_temporal_{section}_all.png  — overlay: all adh × both treatments
  figures/adhesion_violins_15months.png
  figures/adhesion_clustering_summary_15months.png  (sigmoid fit)
  adhesion_summary.csv                          — wide-format final-timepoint summary
  csv_exports/cell_counts_temporal.csv
  csv_exports/clustering_temporal.csv

Run from: ~/ProstateCancer-ABM/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')    # non-interactive backend for HPC use
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore")

os.makedirs("figures", exist_ok=True)
os.makedirs("csv_exports", exist_ok=True)

# ── Simulation constants ──────────────────────────────────────────────────────
MIN_PER_MONTH  = 30 * 24 * 60   # minutes per 30-day month
FINAL_TIME_MIN = 15 * MIN_PER_MONTH  # 648,000 min = 15 months
N_SEEDS        = 10              # seeds per 2D condition

# Adhesion multiplier values and their string representations in folder names.
# "baseline" is a special case: it points to the standard output_{cond}_seed* folders
# from the baseline simulation run rather than adhesion-specific folders.
ADHESION_MULTS = [0.1, 0.25, 1.0, 2.0, 5.0, 10.0]
MULT_STRS      = ["0p1", "0p25", "baseline", "2p0", "5p0", "10p0"]

# Absolute adhesion strength values (µm/min) for each multiplier
# Baseline adhesion = 0.4 µm/min, so values are 0.4 × multiplier
ADHESION_VALS  = [round(0.4 * m, 4) for m in ADHESION_MULTS]

# X-axis tick labels: show multiplier and absolute value for clarity
MULT_LABELS    = ["0.1x\n(0.04)", "0.25x\n(0.10)", "1x\n(0.40)\nbaseline",
                  "2x\n(0.80)", "5x\n(2.00)", "10x\n(4.00)"]

# Condition style definitions for 2D (solid lines) and 3D (dashed lines)
CONDITIONS_2D = {
    "2D_normal": {"label": "2D Normal androgen", "color": "#1F77B4", "ls": "-"},
    "2D_ADT":    {"label": "2D ADT",             "color": "#FF7F0E", "ls": "-"},
}
CONDITIONS_3D = {
    "3D_normal": {"label": "3D Normal androgen (n=1)", "color": "#1F77B4", "ls": "--"},
    "3D_ADT":    {"label": "3D ADT (n=1)",             "color": "#FF7F0E", "ls": "--"},
}

# Cell count metrics (derived from raw CSV columns)
METRIC_COLS   = ["total", "S", "R", "RS"]
METRIC_LABELS = ["Total cell count", "Sensitive (S)", "Resistant (R)", "R / S ratio"]

# Clustering index columns written by custom.cpp at each save interval.
# C_avg_all         — mean over all cells
# C_avg_PTEN_normal — mean over S cells only
# C_avg_PTEN_deleted — mean over R cells only
CLUST_COLS    = ["C_avg_all", "C_avg_PTEN_normal", "C_avg_PTEN_deleted"]
CLUST_LABELS  = ["Clustering index (all)", "Clustering index (S)", "Clustering index (R)"]

# ── Helper functions ──────────────────────────────────────────────────────────

def load_csv(path):
    """
    Load a single analysis_over_time.csv and add derived columns.

    Derived columns:
      S     — Alive_PTEN_Normal (sensitive cells)
      R     — Alive_PTEN_Deleted (resistant cells)
      total — S + R
      RS    — R / S ratio (NaN when S = 0)
      month — simulation time in months

    Duplicates are removed (keeping last), and rows are sorted by time_min.
    Returns None if the file does not exist.
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

# Alias so both sections of the script can call the same loader
load_csv_full = load_csv

def get_final_val(df, col):
    """
    Return the value of col at the final timepoint (t = 15 months).

    Finds the row with time_min closest to FINAL_TIME_MIN.
    Returns NaN if df is None.
    """
    if df is None: return np.nan
    idx = np.argmin(np.abs(df["time_min"].values - FINAL_TIME_MIN))
    return float(df[col].iloc[idx])

def load_2d_mult(mult_str, cond):
    """
    Load all 10 seed CSVs for a 2D condition at a given adhesion multiplier.

    Special case: mult_str == "baseline" uses the standard baseline output folders
    (output_{cond}_seed*) rather than adhesion-specific folders (output_adh{mult}_...).

    Returns a list of DataFrames. Missing seeds are silently skipped.
    """
    dfs = []
    for seed in range(N_SEEDS):
        if mult_str == "baseline":
            path = f"output_{cond}_seed{seed:02d}/analysis_over_time.csv"
        else:
            path = f"output_adh{mult_str}_{cond}_seed{seed:02d}/analysis_over_time.csv"
        df = load_csv(path)
        if df is not None:
            dfs.append(df)
    return dfs

def load_3d_mult(mult_str, cond):
    """
    Load the single-seed 3D CSV for a given adhesion multiplier and condition.
    Returns None if the file does not exist.
    """
    if mult_str == "baseline":
        path = f"output_{cond}_seed00/analysis_over_time.csv"
    else:
        path = f"output_adh{mult_str}_{cond}_seed00/analysis_over_time.csv"
    return load_csv(path)

def compute_stats_list(dfs, col):
    """
    Compute mean and standard deviation of col across a list of seed DataFrames.

    Uses the first DataFrame's time axis as the reference grid, then re-indexes
    and interpolates each seed onto that grid. This handles minor timing
    differences between seeds (e.g. slight save interval jitter).

    Returns (mean_array, sd_array), or (None, None) if dfs is empty.
    """
    if not dfs: return None, None
    arr = np.array([
        df.set_index("time_min")[col]
          .reindex(dfs[0]["time_min"].values)
          .interpolate(method="index").values
        for df in dfs
    ], dtype=float)
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)

def pval_label(p):
    """Convert a p-value to a significance annotation string."""
    if p < 0.001:  return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else:          return "ns"

def add_bracket(ax, x1, x2, y, h, label, fontsize=7):
    """
    Draw a significance bracket between two x positions on a matplotlib axis.

    Parameters:
      x1, x2 — bracket end positions on x-axis
      y       — y position of the horizontal bar
      h       — tick height above the bar
      label   — annotation text (e.g. "***", "ns")
    """
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.0, color="black")
    ax.text((x1+x2)/2, y+h*1.1, label, ha="center", va="bottom",
            fontsize=fontsize, fontweight="bold" if label != "ns" else "normal")

def sigmoid(x, L, k, x0, b):
    """
    4-parameter logistic (sigmoid) function.

    Parameters:
      L  — upper asymptote (ceiling of sigmoid)
      k  — steepness / growth rate
      x0 — x value at the inflection point (midpoint)
      b  — lower asymptote (floor of sigmoid)
    """
    return b + (L - b) / (1 + np.exp(-k * (x - x0)))

def fit_sigmoid(x_vals, y_vals):
    """
    Fit a sigmoid to (x_vals, y_vals) using scipy curve_fit.

    Returns (x_dense, y_fit, popt) for plotting and parameter inspection.
    x_dense is a 200-point array spanning the input x range.
    Returns (None, None, None) if the fit fails or diverges.
    """
    try:
        p0 = [max(y_vals), 1.0, np.median(x_vals), min(y_vals)]
        bounds = ([0, 0, min(x_vals)-2, 0], [1.5, 10, max(x_vals)+2, 1.0])
        popt, _ = curve_fit(sigmoid, x_vals, y_vals, p0=p0,
                            bounds=bounds, maxfev=5000)
        x_dense = np.linspace(min(x_vals), max(x_vals), 200)
        return x_dense, sigmoid(x_dense, *popt), popt
    except Exception:
        return None, None, None

# ── Load all data ─────────────────────────────────────────────────────────────
print("Loading data...")

# data_2d[(mult_str, cond)] — list of DataFrames, one per seed (up to 10)
# data_3d[(mult_str, cond)] — single DataFrame or None
data_2d = {}
data_3d = {}

for mult_str, cond in [(m, c) for m in MULT_STRS for c in CONDITIONS_2D]:
    data_2d[(mult_str, cond)] = load_2d_mult(mult_str, cond)
    n = len(data_2d[(mult_str, cond)])
    print(f"  2D {mult_str} {cond}: {n} seeds")

for mult_str, cond in [(m, c) for m in MULT_STRS for c in CONDITIONS_3D]:
    data_3d[(mult_str, cond)] = load_3d_mult(mult_str, cond)
    ok = "OK" if data_3d[(mult_str, cond)] is not None else "MISSING"
    print(f"  3D {mult_str} {cond}: {ok}")

# Integer x positions for bar/violin placement
x = np.arange(len(ADHESION_MULTS))

# ── SECTION 1: Summary — all metrics at t=15 months vs adhesion multiplier ────
print("\nGenerating summary plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Adhesion Sweep — BR Cohort at t = 15 months", fontsize=13, fontweight="bold")

for ax, col, lbl in zip(axes.flat, METRIC_COLS, METRIC_LABELS):
    # 2D: line plot of mean ± SD across 10 seeds
    for cond, info in CONDITIONS_2D.items():
        means, stds = [], []
        for mult_str in MULT_STRS:
            dfs = data_2d[(mult_str, cond)]
            vals = [get_final_val(df, col) for df in dfs]
            vals = [v for v in vals if not np.isnan(v)]
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if vals else np.nan)
        means, stds = np.array(means), np.array(stds)
        ax.plot(x, means, color=info["color"], ls=info["ls"], lw=2, marker="o",
                label=info["label"])
        ax.fill_between(x, means-stds, means+stds, alpha=0.15, color=info["color"])

    # 3D: single point per adhesion value (square marker, dashed line)
    for cond, info in CONDITIONS_3D.items():
        vals = [get_final_val(data_3d[(m, cond)], col) for m in MULT_STRS]
        ax.plot(x, vals, color=info["color"], ls=info["ls"], lw=2, marker="s",
                markersize=8, label=info["label"])

    ax.set_xticks(x)
    ax.set_xticklabels(MULT_LABELS, fontsize=8)
    ax.set_xlabel("Adhesion multiplier (absolute value)")
    ax.set_ylabel(lbl)
    ax.set_title(lbl, fontweight="bold")
    ax.grid(True, alpha=0.3, ls=":")
    # Vertical dotted line at baseline (index 2 = 1× multiplier)
    ax.axvline(x=2, color="grey", ls=":", lw=1.5, alpha=0.6)
    if col == "RS": ax.set_ylim(bottom=0)

handles = [Line2D([0],[0], color=v["color"], ls=v["ls"], lw=2,
                  marker="o" if "2D" in k else "s", label=v["label"])
           for k, v in {**CONDITIONS_2D, **CONDITIONS_3D}.items()]
fig.legend(handles=handles, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, -0.02), fontsize=9)
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.savefig("figures/adhesion_summary_15months.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/adhesion_summary_15months.png")

# ── SECTION 2: Temporal plots — all adhesion values overlaid ──────────────────
print("Generating temporal plots...")
cmap   = plt.cm.viridis
colors = [cmap(i/5) for i in range(6)]  # one color per adhesion multiplier

# First pass: simple temporal plot per metric (2 panels: normal vs. ADT)
for col, lbl in zip(METRIC_COLS, METRIC_LABELS):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Adhesion Sweep — {lbl} over time (BR Cohort)",
                 fontsize=13, fontweight="bold")

    for ax, (cond_2d, cond_3d, title) in zip(axes, [
        ("2D_normal", "3D_normal", "Normal androgen (8 ng/ml)"),
        ("2D_ADT",    "3D_ADT",    "ADT (1 ng/ml)"),
    ]):
        for i, (mult_str, color) in enumerate(zip(MULT_STRS, colors)):
            # 2D: mean ± SD shading
            dfs = data_2d[(mult_str, cond_2d)]
            if dfs:
                ref_times = dfs[0]["time_min"].values
                mean, sd = compute_stats_list(dfs, col)
                if mean is not None:
                    months = ref_times / MIN_PER_MONTH
                    ax.plot(months, mean, color=color, lw=2,
                            label=f"{MULT_LABELS[i].split(chr(10))[0]} adh")
                    ax.fill_between(months, mean-sd, mean+sd, alpha=0.12, color=color)
            # 3D: single seed dashed line
            df3 = data_3d[(mult_str, cond_3d)]
            if df3 is not None:
                ax.plot(df3["month"], df3[col], color=color, lw=1.5, ls="--", alpha=0.8)

        ax.set_xlabel("Time (months)"); ax.set_ylabel(lbl)
        ax.set_title(title, fontweight="bold")
        ax.set_xlim(0, 15); ax.grid(True, alpha=0.3, ls=":")
        if col == "RS": ax.set_ylim(bottom=0)

    legend_els = [Line2D([0],[0], color=colors[i], lw=2,
                         label=f"adh {MULT_LABELS[i].split(chr(10))[0]}")
                  for i in range(6)]
    legend_els += [
        Line2D([0],[0], color="grey", lw=2,   ls="-",  label="2D (mean±SD)"),
        Line2D([0],[0], color="grey", lw=1.5, ls="--", label="3D (n=1)"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.04), fontsize=8)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(f"figures/adhesion_temporal_{col}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/adhesion_temporal_{col}.png")

# Second pass: full temporal plots for both cell counts and clustering metrics,
# separately by treatment condition and as a combined overlay figure.
TEMPORAL_METRICS = {
    "cell_counts": {
        "cols":   ["total", "S", "R", "RS"],
        "labels": ["Total cell count", "Sensitive (S)", "Resistant (R)", "R / S ratio"],
        "title":  "Cell Counts over Time",
    },
    "clustering": {
        "cols":   ["C_avg_all", "C_avg_PTEN_normal", "C_avg_PTEN_deleted"],
        "labels": ["Clustering index (all)", "Clustering index (S)", "Clustering index (R)"],
        "title":  "Spatial Clustering Index over Time",
    },
}

# Reload all data with the full column set (including clustering columns)
# This is a reload because the initial load above may not have loaded clustering cols.
data_2d_full = {}
data_3d_full = {}
for mult_str in MULT_STRS:
    for cond in list(CONDITIONS_2D):
        dfs = []
        for seed in range(N_SEEDS):
            path = (f"output_{cond}_seed{seed:02d}/analysis_over_time.csv"
                    if mult_str == "baseline"
                    else f"output_adh{mult_str}_{cond}_seed{seed:02d}/analysis_over_time.csv")
            df = load_csv_full(path)
            if df is not None: dfs.append(df)
        data_2d_full[(mult_str, cond)] = dfs
    for cond in list(CONDITIONS_3D):
        path = (f"output_{cond}_seed00/analysis_over_time.csv"
                if mult_str == "baseline"
                else f"output_adh{mult_str}_{cond}_seed00/analysis_over_time.csv")
        data_3d_full[(mult_str, cond)] = load_csv_full(path)

cmap2   = plt.cm.viridis
colors6 = [cmap2(i/5) for i in range(6)]

for section_key, section in TEMPORAL_METRICS.items():
    cols, labels, title = section["cols"], section["labels"], section["title"]
    ncols_plot = len(cols)

    # Per-treatment figure (one panel per metric)
    for cond_2d, cond_3d, treat_label in [
        ("2D_normal", "3D_normal", "Normal androgen (8 ng/ml)"),
        ("2D_ADT",    "3D_ADT",    "ADT (1 ng/ml)"),
    ]:
        fig, axes = plt.subplots(1, ncols_plot, figsize=(6*ncols_plot, 6))
        if ncols_plot == 1: axes = [axes]
        fig.suptitle(f"Adhesion Sweep — {title}\n{treat_label} (BR Cohort)",
                     fontsize=13, fontweight="bold")

        for ax, col, lbl in zip(axes, cols, labels):
            for i, (mult_str, color) in enumerate(zip(MULT_STRS, colors6)):
                dfs = data_2d_full[(mult_str, cond_2d)]
                if dfs:
                    mean, sd = compute_stats_list(dfs, col)
                    if mean is not None:
                        months = dfs[0]["time_min"].values / MIN_PER_MONTH
                        ax.plot(months, mean, color=color, lw=2,
                                label=f"{MULT_LABELS[i].split(chr(10))[0]} adh (2D)")
                        ax.fill_between(months, mean-sd, mean+sd, alpha=0.12, color=color)
                df3 = data_3d_full[(mult_str, cond_3d)]
                if df3 is not None and col in df3.columns:
                    ax.plot(df3["month"], df3[col], color=color, lw=1.5, ls="--", alpha=0.85)

            ax.set_xlabel("Time (months)", fontsize=10); ax.set_ylabel(lbl, fontsize=10)
            ax.set_title(lbl, fontweight="bold"); ax.set_xlim(0, 15)
            ax.grid(True, alpha=0.3, ls=":")
            if col in ("RS", "C_avg_all", "C_avg_PTEN_normal", "C_avg_PTEN_deleted"):
                ax.set_ylim(bottom=0)

        legend_els = [Line2D([0],[0], color=colors6[i], lw=2,
                             label=MULT_LABELS[i].split(chr(10))[0]) for i in range(6)] + [
            Line2D([0],[0], color="grey", lw=2,   ls="-",  label="2D (mean ± SD)"),
            Line2D([0],[0], color="grey", lw=1.5, ls="--", label="3D (n=1)"),
        ]
        fig.legend(handles=legend_els, loc="lower center", ncol=4,
                   bbox_to_anchor=(0.5, -0.06), fontsize=8)
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        cond_str = "normal" if "normal" in cond_2d else "ADT"
        fname = f"figures/adhesion_temporal_{section_key}_{cond_str}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Saved: {fname}")

    # Combined overlay: 2-row figure (normal top, ADT bottom)
    fig, axes = plt.subplots(2, ncols_plot, figsize=(6*ncols_plot, 10))
    if ncols_plot == 1: axes = axes.reshape(2, 1)
    fig.suptitle(f"Adhesion Sweep — {title} — All conditions (BR Cohort)",
                 fontsize=13, fontweight="bold")

    for row_idx, (cond_2d, cond_3d, treat_label) in enumerate([
        ("2D_normal", "3D_normal", "Normal androgen"),
        ("2D_ADT",    "3D_ADT",    "ADT"),
    ]):
        for ax, col, lbl in zip(axes[row_idx], cols, labels):
            for i, (mult_str, color) in enumerate(zip(MULT_STRS, colors6)):
                dfs = data_2d_full[(mult_str, cond_2d)]
                if dfs:
                    mean, sd = compute_stats_list(dfs, col)
                    if mean is not None:
                        months = dfs[0]["time_min"].values / MIN_PER_MONTH
                        ax.plot(months, mean, color=color, lw=2)
                        ax.fill_between(months, mean-sd, mean+sd, alpha=0.10, color=color)
                df3 = data_3d_full[(mult_str, cond_3d)]
                if df3 is not None and col in df3.columns:
                    ax.plot(df3["month"], df3[col], color=color, lw=1.5, ls="--", alpha=0.85)

            ax.set_xlabel("Time (months)", fontsize=9); ax.set_ylabel(lbl, fontsize=9)
            ax.set_title(f"{treat_label} — {lbl}", fontweight="bold", fontsize=10)
            ax.set_xlim(0, 15); ax.grid(True, alpha=0.3, ls=":")
            if col in ("RS", "C_avg_all", "C_avg_PTEN_normal", "C_avg_PTEN_deleted"):
                ax.set_ylim(bottom=0)

    legend_els = [Line2D([0],[0], color=colors6[i], lw=2,
                         label=MULT_LABELS[i].split(chr(10))[0]) for i in range(6)] + [
        Line2D([0],[0], color="grey", lw=2,   ls="-",  label="2D (mean ± SD)"),
        Line2D([0],[0], color="grey", lw=1.5, ls="--", label="3D (n=1)"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.03), fontsize=8)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fname = f"figures/adhesion_temporal_{section_key}_all.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}")

print("\nAll temporal plots done!")

# ── SECTION 3: Violin plots at t=15 months with significance brackets ──────────
print("Generating violin plots...")
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle("Adhesion Sweep — Distribution at t = 15 months (2D seeds + 3D single point)",
             fontsize=13, fontweight="bold")

cond_pairs = [
    ("2D_normal", "3D_normal", "#1F77B4", "Normal androgen"),
    ("2D_ADT",    "3D_ADT",    "#FF7F0E", "ADT"),
]

for ax, col, lbl in zip(axes.flat, METRIC_COLS, METRIC_LABELS):
    width = 0.35

    for offset, (cond_2d, cond_3d, color, clabel) in enumerate(cond_pairs):
        positions = x + (offset - 0.5) * width

        # Collect final-timepoint values from all seeds per adhesion value
        all_vals = []
        for mult_str in MULT_STRS:
            dfs = data_2d[(mult_str, cond_2d)]
            vals = [get_final_val(df, col) for df in dfs]
            vals = [v for v in vals if not np.isnan(v)]
            all_vals.append(vals if vals else [np.nan])

        # Violin plots (require at least 2 non-NaN values)
        valid = [(pos, vals) for pos, vals in zip(positions, all_vals)
                 if not all(np.isnan(v) for v in vals)]
        if valid:
            vp = ax.violinplot([v[1] for v in valid],
                               positions=[v[0] for v in valid],
                               widths=width*0.8,
                               showmeans=True, showmedians=True)
            for pc in vp["bodies"]:
                pc.set_facecolor(color); pc.set_alpha(0.45); pc.set_edgecolor("black")
            vp["cmeans"].set_color("black")   # mean shown as black line
            vp["cmedians"].set_color("white") # median shown as white line
            for comp in ["cbars","cmaxes","cmins"]:
                vp[comp].set_color("grey")
            # Overlay individual seed points with horizontal jitter
            for pos, vals in valid:
                jitter = np.random.uniform(-width*0.2, width*0.2, len(vals))
                ax.scatter([pos+j for j in jitter], vals, color=color,
                           edgecolors="black", s=25, zorder=5, alpha=0.8, linewidths=0.5)

        # 3D single seed as a star marker
        for pos, mult_str in zip(positions, MULT_STRS):
            df3 = data_3d[(mult_str, cond_3d)]
            val3 = get_final_val(df3, col)
            if not np.isnan(val3):
                ax.scatter([pos], [val3], color=color, marker="*", s=150,
                           edgecolors="black", zorder=6, linewidths=0.8)

    ax.set_xticks(x); ax.set_xticklabels(MULT_LABELS, fontsize=8)
    ax.set_xlabel("Adhesion multiplier"); ax.set_ylabel(lbl)
    ax.set_title(lbl, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y", ls=":")
    if col == "RS": ax.set_ylim(bottom=0)
    ax.axvline(x=2, color="grey", ls=":", lw=1.5, alpha=0.6)

    # ── Significance bracket 1: Normal androgen vs. ADT at each adhesion value ─
    # Uses a two-sided Mann-Whitney U test on the 2D seed distributions.
    y_max = ax.get_ylim()[1]
    y_range = y_max - ax.get_ylim()[0]
    bracket_h = y_range * 0.03
    bracket_step = y_range * 0.12

    for xi, mult_str in enumerate(MULT_STRS):
        vals_normal = [get_final_val(df, col) for df in data_2d[(mult_str, "2D_normal")]]
        vals_adt    = [get_final_val(df, col) for df in data_2d[(mult_str, "2D_ADT")]]
        vals_normal = [v for v in vals_normal if not np.isnan(v)]
        vals_adt    = [v for v in vals_adt    if not np.isnan(v)]
        if len(vals_normal) < 2 or len(vals_adt) < 2: continue
        _, p = stats.mannwhitneyu(vals_normal, vals_adt, alternative="two-sided")
        add_bracket(ax, xi-0.5*width, xi+0.5*width,
                    y_max + bracket_step * 0.3, bracket_h, pval_label(p), fontsize=7)

    # ── Significance bracket 2: Each adhesion vs. baseline (2D normal only) ───
    # Only non-baseline adhesion values are tested; only significant results shown
    # to avoid plot clutter.
    baseline_vals = [get_final_val(df, col) for df in data_2d[("baseline", "2D_normal")]]
    baseline_vals = [v for v in baseline_vals if not np.isnan(v)]
    y_top = y_max + bracket_step * 0.3 + bracket_h * 3

    for xi, mult_str in enumerate(MULT_STRS):
        if mult_str == "baseline": continue
        vals = [get_final_val(df, col) for df in data_2d[(mult_str, "2D_normal")]]
        vals = [v for v in vals if not np.isnan(v)]
        if len(vals) < 2: continue
        _, p = stats.mannwhitneyu(baseline_vals, vals, alternative="two-sided")
        if pval_label(p) != "ns":
            ax.text(xi, y_top, f"vs base:\n{pval_label(p)}",
                    ha="center", va="bottom", fontsize=6,
                    color="#333333", style="italic")

    ax.set_ylim(top=y_top + y_range * 0.18)

from matplotlib.patches import Patch
legend_els = [
    Line2D([0],[0], color="#1F77B4", lw=6, alpha=0.5, label="Normal androgen 2D (violin=10 seeds)"),
    Line2D([0],[0], color="#FF7F0E", lw=6, alpha=0.5, label="ADT 2D (violin=10 seeds)"),
    Line2D([0],[0], marker="*", color="w", markerfacecolor="#1F77B4",
           markersize=12, label="Normal androgen 3D (n=1)"),
    Line2D([0],[0], marker="*", color="w", markerfacecolor="#FF7F0E",
           markersize=12, label="ADT 3D (n=1)"),
]
fig.legend(handles=legend_els, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, -0.02), fontsize=9)
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig("figures/adhesion_violins_15months.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/adhesion_violins_15months.png")

# ── SECTION 4: Wide-format summary CSV at t=15 months ─────────────────────────
# Each row is one adhesion × condition combination. Cell count metrics are
# stored as {col}_mean and {col}_sd columns.
rows = []
for mult_str, mult_val, mult_label in zip(MULT_STRS, ADHESION_VALS, MULT_LABELS):
    for cond in list(CONDITIONS_2D) + list(CONDITIONS_3D):
        is_3d = "3D" in cond
        row = {
            "adhesion_mult_str": mult_str,
            "adhesion_val": mult_val,
            "condition": cond,
            "n_seeds": 1 if is_3d else N_SEEDS
        }
        for col in METRIC_COLS:
            if is_3d:
                df = data_3d[(mult_str, cond)]
                row[f"{col}_mean"] = get_final_val(df, col)
                row[f"{col}_sd"]   = np.nan
            else:
                dfs = data_2d[(mult_str, cond)]
                vals = [get_final_val(df, col) for df in dfs if df is not None]
                vals = [v for v in vals if not np.isnan(v)]
                row[f"{col}_mean"] = np.mean(vals) if vals else np.nan
                row[f"{col}_sd"]   = np.std(vals)  if vals else np.nan
        rows.append(row)

pd.DataFrame(rows).to_csv("adhesion_summary.csv", index=False)
print("  Saved: adhesion_summary.csv")
print("\nDone! Figures in figures/")

# ── SECTION 5: Master temporal CSVs (cell counts + clustering) ─────────────────
# Produces two long-format CSVs with columns:
#   adhesion_mult, adhesion_val, treatment, dimension, n_seeds,
#   time_min, month, metric, mean, sd
print("\nExporting master temporal CSVs...")

cc_all = []   # cell count rows
cl_all = []   # clustering index rows

for mult_str, mult_label in zip(MULT_STRS, MULT_LABELS):
    mult_clean = mult_label.split("\n")[0]

    for cond_2d, cond_3d, treat in [
        ("2D_normal", "3D_normal", "normal_androgen"),
        ("2D_ADT",    "3D_ADT",    "ADT"),
    ]:
        # 2D: mean and SD across 10 seeds at every timepoint
        dfs = data_2d_full[(mult_str, cond_2d)]
        if dfs:
            ref_times = dfs[0]["time_min"].values
            for col, col_label, target in [
                ("total", "Total_cells", cc_all), ("S", "Sensitive_S", cc_all),
                ("R", "Resistant_R", cc_all),     ("RS", "R_S_ratio", cc_all),
                ("C_avg_all", "C_avg_all", cl_all),
                ("C_avg_PTEN_normal", "C_avg_S", cl_all),
                ("C_avg_PTEN_deleted", "C_avg_R", cl_all),
            ]:
                if col not in dfs[0].columns: continue
                mean, sd = compute_stats_list(dfs, col)
                if mean is None: continue
                for t, m, s in zip(ref_times, mean, sd):
                    target.append({
                        "adhesion_mult": mult_clean,
                        "adhesion_val":  0.4 * ADHESION_MULTS[MULT_STRS.index(mult_str)],
                        "treatment":     treat,
                        "dimension":     "2D",
                        "n_seeds":       len(dfs),
                        "time_min":      t,
                        "month":         t / MIN_PER_MONTH,
                        "metric":        col_label,
                        "mean":          m,
                        "sd":            s,
                    })

        # 3D: single seed — SD is NaN since only one replicate available
        df3 = data_3d_full[(mult_str, cond_3d)]
        if df3 is not None:
            for col, col_label, target in [
                ("total", "Total_cells", cc_all), ("S", "Sensitive_S", cc_all),
                ("R", "Resistant_R", cc_all),     ("RS", "R_S_ratio", cc_all),
                ("C_avg_all", "C_avg_all", cl_all),
                ("C_avg_PTEN_normal", "C_avg_S", cl_all),
                ("C_avg_PTEN_deleted", "C_avg_R", cl_all),
            ]:
                if col not in df3.columns: continue
                for _, row in df3.iterrows():
                    target.append({
                        "adhesion_mult": mult_clean,
                        "adhesion_val":  0.4 * ADHESION_MULTS[MULT_STRS.index(mult_str)],
                        "treatment":     treat,
                        "dimension":     "3D",
                        "n_seeds":       1,
                        "time_min":      row["time_min"],
                        "month":         row["month"],
                        "metric":        col_label,
                        "mean":          row[col],
                        "sd":            np.nan,
                    })

pd.DataFrame(cc_all).to_csv("csv_exports/cell_counts_temporal.csv", index=False)
print("  Saved: csv_exports/cell_counts_temporal.csv")

pd.DataFrame(cl_all).to_csv("csv_exports/clustering_temporal.csv", index=False)
print("  Saved: csv_exports/clustering_temporal.csv")

print("\nDone! Two CSVs in csv_exports/")

# ── SECTION 6: Clustering index vs. adhesion — sigmoid fit ────────────────────
# X-axis uses actual adhesion strength values (µm/min) on a log scale rather
# than integer indices, because adhesion spans two orders of magnitude and the
# sigmoid shape is more meaningful on a log scale.
print("\nGenerating clustering vs adhesion summary plot (sigmoid fit)...")

CLUST_COLS_ADH   = ["C_avg_all", "C_avg_PTEN_normal", "C_avg_PTEN_deleted"]
CLUST_LABELS_ADH = ["Clustering index (all)", "Clustering index (S)", "Clustering index (R)"]

x_vals = np.array(ADHESION_VALS)   # [0.04, 0.10, 0.40, 0.80, 2.00, 4.00]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Adhesion Sweep — Spatial Clustering Index at t = 15 months\n"
             "Points = data (mean ± SD), curves = sigmoid fit",
             fontsize=13, fontweight="bold")

for ax, col, lbl in zip(axes, CLUST_COLS_ADH, CLUST_LABELS_ADH):

    # 2D: error bar scatter + sigmoid fit
    for cond, info in CONDITIONS_2D.items():
        means, sds = [], []
        for mult_str in MULT_STRS:
            dfs = data_2d_full[(mult_str, cond)]
            vals = [get_final_val(df, col) for df in dfs
                    if df is not None and col in df.columns]
            vals = [v for v in vals if not np.isnan(v)]
            means.append(np.mean(vals) if vals else np.nan)
            sds.append(np.std(vals) if vals else np.nan)
        means, sds = np.array(means), np.array(sds)

        ax.errorbar(x_vals, means, yerr=sds, fmt="o", color=info["color"],
                    markersize=7, capsize=4, lw=1.2, zorder=5,
                    markeredgecolor="black", ecolor=info["color"],
                    label=f"{info['label']} (data)")

        # Fit sigmoid only to non-NaN mean values
        valid = ~np.isnan(means)
        x_fit, y_fit, _ = fit_sigmoid(x_vals[valid], means[valid])
        if x_fit is not None:
            ax.plot(x_fit, y_fit, color=info["color"], ls="-", lw=2.5, alpha=0.85)

    # 3D: scatter + optional sigmoid (requires ≥4 non-NaN points)
    for cond, info in CONDITIONS_3D.items():
        vals_3d = np.array([
            get_final_val(data_3d_full[(m, cond)], col)
            if data_3d_full[(m, cond)] is not None and col in data_3d_full[(m, cond)].columns
            else np.nan
            for m in MULT_STRS
        ])
        ax.scatter(x_vals, vals_3d, color=info["color"], marker="s",
                   s=70, zorder=5, edgecolors="black", lw=0.8,
                   label=f"{info['label']} (n=1)")
        valid = ~np.isnan(vals_3d)
        if valid.sum() >= 4:
            x_fit, y_fit, _ = fit_sigmoid(x_vals[valid], vals_3d[valid])
            if x_fit is not None:
                ax.plot(x_fit, y_fit, color=info["color"], ls="--", lw=2.0, alpha=0.85)

    ax.set_xscale("log")   # log scale: adhesion spans 0.04–4.0 µm/min (2 orders)
    ax.set_xlabel("Adhesion strength (µm/min, log scale)", fontsize=10)
    ax.set_ylabel(lbl, fontsize=10)
    ax.set_title(lbl, fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, ls=":")
    # Vertical dotted line at baseline adhesion value (0.4 µm/min)
    ax.axvline(x=0.4, color="grey", ls=":", lw=1.5, alpha=0.6, label="Baseline (0.4)")

handles = [Line2D([0],[0], color=v["color"], ls=v["ls"], lw=2,
                  marker="o" if "2D" in k else "s", label=v["label"])
           for k, v in {**CONDITIONS_2D, **CONDITIONS_3D}.items()]
fig.legend(handles=handles, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, -0.04), fontsize=9)
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("figures/adhesion_clustering_summary_15months.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/adhesion_clustering_summary_15months.png")
print("\nAll done.")
