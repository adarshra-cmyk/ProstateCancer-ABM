"""
analyze_results.py
Reads analysis_over_time.csv from all 40 PhysiCell runs and produces:
1. Temporal plots (mean ± 1 SD) for total, S, R, R/S ratio
2. Violin plots at t = 15 months  **with p-value significance brackets**
3. results_summary.csv
"""

import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
CONDITIONS = {
    "2D_normal": {"label": "2D | Normal androgen (8 ng/ml)", "color": "#1F77B4", "ls": "-"},
    "2D_ADT":    {"label": "2D | ADT (1 ng/ml)",             "color": "#FF7F0E", "ls": "-"},
    "3D_normal": {"label": "3D | Normal androgen (8 ng/ml)", "color": "#1F77B4", "ls": "--"},
    "3D_ADT":    {"label": "3D | ADT (1 ng/ml)",             "color": "#FF7F0E", "ls": "--"},
}
N_SEEDS = 10
MIN_PER_MONTH = 30 * 24 * 60
FINAL_TIME_MIN = 15 * MIN_PER_MONTH  # 648000

os.makedirs("figures", exist_ok=True)

# ── Comparisons to annotate (index pairs into cond_keys list, 0-based) ────────
# Each tuple: (i, j, label) where i < j
COMPARISONS = [
    (0, 1, "2D: Normal vs ADT"),      # 2D_normal vs 2D_ADT
    (2, 3, "3D: Normal vs ADT"),      # 3D_normal vs 3D_ADT
    (0, 2, "Normal: 2D vs 3D"),       # 2D_normal vs 3D_normal
    (1, 3, "ADT: 2D vs 3D"),          # 2D_ADT    vs 3D_ADT
]

# ── p-value formatting ────────────────────────────────────────────────────────
def pval_label(p):
    """Return significance string for a p-value."""
    if p < 0.001:
        return "***\np<0.001"
    elif p < 0.01:
        return f"**\np={p:.3f}"
    elif p < 0.05:
        return f"*\np={p:.3f}"
    else:
        return f"ns\np={p:.3f}"

def add_significance_brackets(ax, data_per_cond, comparisons, base_y_frac=0.05):
    """
    Draw bracket + p-value annotations above violin plots.

    Parameters
    ----------
    ax            : matplotlib Axes
    data_per_cond : list of lists, one per condition (positions 1..N)
    comparisons   : list of (i, j, label) tuples  (0-based indices)
    base_y_frac   : extra headroom above the data maximum before the first bracket
    """
    # Collect all finite values to find the data ceiling
    all_vals = [v for grp in data_per_cond for v in grp if np.isfinite(v)]
    if not all_vals:
        return

    y_max = max(all_vals)
    y_range = y_max - min(all_vals) if min(all_vals) != y_max else abs(y_max) or 1
    bracket_gap   = y_range * 0.08   # vertical step between stacked brackets
    bracket_h     = y_range * 0.025  # height of the bracket tips
    text_offset   = y_range * 0.01   # gap between bracket top and text

    # Sort comparisons by span width so narrower ones sit lower
    comparisons_sorted = sorted(comparisons, key=lambda c: abs(c[1] - c[0]))

    current_top = y_max + y_range * base_y_frac

    for (i, j, _) in comparisons_sorted:
        # positions are 1-based
        x1, x2 = i + 1, j + 1
        a = np.array([v for v in data_per_cond[i] if np.isfinite(v)])
        b = np.array([v for v in data_per_cond[j] if np.isfinite(v)])
        if len(a) < 2 or len(b) < 2:
            continue

        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        label = pval_label(p)

        y_bracket = current_top

        # Horizontal bar
        ax.plot([x1, x1, x2, x2],
                [y_bracket - bracket_h, y_bracket, y_bracket, y_bracket - bracket_h],
                lw=1.2, color="black")

        # Text centred above the bar
        ax.text((x1 + x2) / 2, y_bracket + text_offset, label,
                ha="center", va="bottom", fontsize=7.5,
                multialignment="center")

        current_top = y_bracket + bracket_gap + y_range * 0.10

    # Expand y-axis to accommodate all brackets + text
    ax.set_ylim(top=current_top + bracket_gap)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_condition(cond_key):
    dfs = {}
    for seed in range(N_SEEDS):
        pattern = f"output_{cond_key}_seed{seed:02d}/analysis_over_time.csv"
        files = glob.glob(pattern)
        if not files:
            print(f"  WARNING: missing {pattern}")
            continue
        df = pd.read_csv(files[0])
        df = df.drop_duplicates(subset="time_min", keep="last").sort_values("time_min").reset_index(drop=True)
        df["S"]     = df["Alive_PTEN_Normal"]
        df["R"]     = df["Alive_PTEN_Deleted"]
        df["total"] = df["S"] + df["R"]
        df["RS"]    = df["R"] / df["S"].replace(0, np.nan)
        df["month"] = df["time_min"] / MIN_PER_MONTH
        dfs[seed] = df
    return dfs

def align_to_common_times(dfs):
    if not dfs:
        return None, None
    ref_times = list(dfs.values())[0]["time_min"].values
    aligned = {}
    for seed, df in dfs.items():
        indexed = df.set_index("time_min")
        indexed = indexed[~indexed.index.duplicated(keep="last")]
        aligned[seed] = indexed.reindex(ref_times).interpolate(method="index")
    return aligned, ref_times

def compute_stats(aligned_dfs, metric_col):
    if aligned_dfs is None:
        return None, None, None
    arr = np.array([df[metric_col].values for df in aligned_dfs.values()], dtype=float)
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0), arr

def get_final_values(aligned_dfs, metric_col, ref_times):
    if aligned_dfs is None or ref_times is None:
        return []
    final_idx = np.argmin(np.abs(np.array(ref_times) - FINAL_TIME_MIN))
    return [float(df[metric_col].iloc[final_idx]) for df in aligned_dfs.values()
            if not np.isnan(df[metric_col].iloc[final_idx])]

# ── Load all data ─────────────────────────────────────────────────────────────
print("Loading simulation data...")
all_data = {}
for cond in CONDITIONS:
    print(f"  {cond}...")
    raw = load_condition(cond)
    aligned, ref_times = align_to_common_times(raw)
    all_data[cond] = {"aligned": aligned, "ref_times": ref_times}

metric_cols   = ["total", "S", "R", "RS"]
metric_labels = ["Total cell count", "Sensitive (S)", "Resistant (R)", "R / S ratio"]

# ── Plot 1: Temporal per condition ────────────────────────────────────────────
for cond, info in CONDITIONS.items():
    d = all_data[cond]
    if d["aligned"] is None:
        continue
    ref_months = np.array(d["ref_times"]) / MIN_PER_MONTH
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"BR Cohort — {info['label']}\n(mean ± 1 SD, n={len(d['aligned'])} seeds)",
                 fontsize=12, fontweight="bold")
    for ax, col, lbl in zip(axes.flat, metric_cols, metric_labels):
        mean, sd, _ = compute_stats(d["aligned"], col)
        if mean is None: continue
        ax.plot(ref_months, mean, color=info["color"], lw=2)
        ax.fill_between(ref_months, mean-sd, mean+sd, alpha=0.25, color=info["color"])
        ax.set_xlabel("Time (months)"); ax.set_ylabel(lbl)
        ax.set_title(lbl, fontweight="bold"); ax.set_xlim(0, 15)
        ax.grid(True, alpha=0.3, ls=":")
        if col == "RS": ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(f"figures/temporal_{cond}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/temporal_{cond}.png")

# ── Plot 2: All conditions overlaid ──────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("BR Cohort — All conditions (mean ± 1 SD)", fontsize=13, fontweight="bold")
for ax, col, lbl in zip(axes.flat, metric_cols, metric_labels):
    for cond, info in CONDITIONS.items():
        d = all_data[cond]
        if d["aligned"] is None: continue
        ref_months = np.array(d["ref_times"]) / MIN_PER_MONTH
        mean, sd, _ = compute_stats(d["aligned"], col)
        if mean is None: continue
        ax.plot(ref_months, mean, color=info["color"], ls=info["ls"], lw=2, label=info["label"])
        ax.fill_between(ref_months, mean-sd, mean+sd, alpha=0.1, color=info["color"])
    ax.set_xlabel("Time (months)"); ax.set_ylabel(lbl)
    ax.set_title(lbl, fontweight="bold"); ax.set_xlim(0, 15)
    ax.grid(True, alpha=0.3, ls=":")
    if col == "RS": ax.set_ylim(bottom=0)
handles = [Patch(facecolor=v["color"], label=v["label"]) for v in CONDITIONS.values()]
fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02), fontsize=9)
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig("figures/temporal_all_overlay.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/temporal_all_overlay.png")

# ── Plot 3: Violins at 15 months (with p-value brackets) ─────────────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 10))
fig.suptitle(f"BR Cohort — Distribution at t = 15 months\n"
             f"Brackets: Mann-Whitney U (two-sided)  |  * p<0.05  ** p<0.01  *** p<0.001  ns = not significant",
             fontsize=12, fontweight="bold")

cond_keys   = list(CONDITIONS.keys())
cond_colors = [CONDITIONS[c]["color"] for c in cond_keys]
cond_labels = ["2D\nNormal\nandrogen", "2D\nADT", "3D\nNormal\nandrogen", "3D\nADT"]

for ax, col, lbl in zip(axes, metric_cols, metric_labels):
    data_per_cond = []
    for cond in cond_keys:
        d = all_data[cond]
        vals = get_final_values(d["aligned"], col, d["ref_times"])
        data_per_cond.append(vals if vals else [np.nan])

    valid = [(i+1, d, c) for i, (d, c) in enumerate(zip(data_per_cond, cond_colors))
             if not all(np.isnan(v) for v in d)]
    if not valid: continue

    positions = [v[0] for v in valid]
    vdata     = [v[1] for v in valid]
    vcolors   = [v[2] for v in valid]

    parts = ax.violinplot(vdata, positions=positions, showmeans=True, showmedians=True)
    for pc, col_c in zip(parts["bodies"], vcolors):
        pc.set_facecolor(col_c); pc.set_alpha(0.5); pc.set_edgecolor("black")
    parts["cmeans"].set_color("black")
    parts["cmedians"].set_color("white")
    for comp in ["cbars","cmaxes","cmins"]:
        parts[comp].set_color("grey")

    for pos, vals, col_c in zip(positions, vdata, vcolors):
        jitter = np.random.uniform(-0.06, 0.06, len(vals))
        ax.scatter([pos+j for j in jitter], vals, color=col_c,
                   edgecolors="black", s=35, zorder=5, alpha=0.9, linewidths=0.5)

    ax.set_xticks(range(1, len(cond_keys)+1))
    ax.set_xticklabels(cond_labels, fontsize=9)
    ax.set_title(lbl, fontweight="bold"); ax.set_ylabel(lbl)
    ax.grid(True, alpha=0.3, axis="y", ls=":")

    # ── Add significance brackets ──────────────────────────────────────────
    add_significance_brackets(ax, data_per_cond, COMPARISONS)

legend_handles = [
    Patch(facecolor="#1F77B4", label="Normal androgen (8 ng/ml)"),
    Patch(facecolor="#FF7F0E", label="ADT (1 ng/ml)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, -0.01), fontsize=10, frameon=True)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("figures/violins_15months.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/violins_15months.png")

# ── Summary CSV ───────────────────────────────────────────────────────────────
rows = []
for cond in CONDITIONS:
    d = all_data[cond]
    row = {"condition": cond, "label": CONDITIONS[cond]["label"]}
    for col in metric_cols:
        vals = get_final_values(d["aligned"], col, d["ref_times"])
        row[f"{col}_mean"] = np.mean(vals) if vals else np.nan
        row[f"{col}_sd"]   = np.std(vals)  if vals else np.nan
        row[f"{col}_min"]  = np.min(vals)  if vals else np.nan
        row[f"{col}_max"]  = np.max(vals)  if vals else np.nan
    rows.append(row)
pd.DataFrame(rows).to_csv("results_summary.csv", index=False)
print("  Saved: results_summary.csv")
print("\nDone! All figures in figures/")
