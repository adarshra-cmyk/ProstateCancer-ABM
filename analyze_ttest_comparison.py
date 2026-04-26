"""
analyze_ttest_comparison.py
===========================
Generates side-by-side comparison figures showing Mann-Whitney U vs Student's
t-test p-values for every violin-plot analysis across all four sweeps:
  1. Baseline conditions (from analyze_results.py)
  2. Adhesion sweep (from analyze_adhesion.py)
  3. Adhesion+Motility sweep (from analyze_adhmot.py)
  4. Tumor radius sweep (from analyze_radius.py)

For each sweep a figure is produced where:
  - Top row of subplots  → violins annotated with Mann-Whitney U (two-sided)
  - Bottom row of subplots → same violins annotated with Student's t-test (two-sided)

Output figures (all written to figures/):
  figures/compare_ttest_mw_results.png
  figures/compare_ttest_mw_adhesion.png
  figures/compare_ttest_mw_adhmot.png
  figures/compare_ttest_mw_radius.png

A summary CSV with all p-values is also written:
  ttest_mw_pvalue_comparison.csv

Run from: ~/ProstateCancer-ABM/
"""

import os
import glob as globmod
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

os.makedirs("figures", exist_ok=True)

# ── Shared constants ──────────────────────────────────────────────────────────
MIN_PER_MONTH  = 30 * 24 * 60
FINAL_TIME_MIN = 15 * MIN_PER_MONTH

# ── Shared helpers ────────────────────────────────────────────────────────────

def load_csv(path):
    """Load a single analysis_over_time.csv and add derived columns."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = (df.drop_duplicates(subset="time_min", keep="last")
            .sort_values("time_min")
            .reset_index(drop=True))
    df["S"]     = df["Alive_PTEN_Normal"]
    df["R"]     = df["Alive_PTEN_Deleted"]
    df["total"] = df["S"] + df["R"]
    df["RS"]    = df["R"] / df["S"].replace(0, np.nan)
    df["month"] = df["time_min"] / MIN_PER_MONTH
    return df

def get_final_scalar(df, col):
    """Value of col at t≈15 months for a single DataFrame."""
    if df is None:
        return np.nan
    idx = np.argmin(np.abs(df["time_min"].values - FINAL_TIME_MIN))
    return float(df[col].iloc[idx])

def get_final_list(dfs, col):
    """List of final-timepoint values (one per seed), NaN excluded."""
    return [v for df in dfs
            for v in [get_final_scalar(df, col)]
            if not np.isnan(v)]

def pval_stars(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    return "ns"

def pval_label_full(p):
    stars = pval_stars(p)
    return f"{stars}\np={p:.3f}" if p >= 0.001 else f"{stars}\np<0.001"

def mannwhitney_p(a, b):
    """Two-sided Mann-Whitney U p-value; NaN if insufficient data."""
    a = np.array([v for v in a if np.isfinite(v)])
    b = np.array([v for v in b if np.isfinite(v)])
    if len(a) < 2 or len(b) < 2:
        return np.nan
    _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return p

def ttest_p(a, b):
    """Two-sided independent-samples Student's t-test p-value."""
    a = np.array([v for v in a if np.isfinite(v)])
    b = np.array([v for v in b if np.isfinite(v)])
    if len(a) < 2 or len(b) < 2:
        return np.nan
    _, p = stats.ttest_ind(a, b, equal_var=True)
    return p

def add_bracket(ax, x1, x2, y, h, label, fontsize=7):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.0, color="black")
    ax.text((x1+x2)/2, y+h*1.2, label, ha="center", va="bottom",
            fontsize=fontsize,
            fontweight="bold" if "ns" not in label else "normal",
            multialignment="center")

def draw_violins(ax, positions, data_per_pos, colors):
    """
    Draw violin bodies, mean/median lines, and jittered scatter points.
    Returns the axes after drawing.
    """
    valid = [(pos, d, c) for pos, d, c in zip(positions, data_per_pos, colors)
             if len(d) > 1 and not all(np.isnan(v) for v in d)]
    if not valid:
        return
    vp = ax.violinplot([v[1] for v in valid],
                       positions=[v[0] for v in valid],
                       showmeans=True, showmedians=True)
    for pc, (_, _, col) in zip(vp["bodies"], valid):
        pc.set_facecolor(col); pc.set_alpha(0.5); pc.set_edgecolor("black")
    vp["cmeans"].set_color("black")
    vp["cmedians"].set_color("white")
    for comp in ["cbars", "cmaxes", "cmins"]:
        vp[comp].set_color("grey")
    for pos, vals, col in valid:
        jitter = np.random.uniform(-0.07, 0.07, len(vals))
        ax.scatter([pos+j for j in jitter], vals, color=col,
                   edgecolors="black", s=30, zorder=5, alpha=0.85, linewidths=0.5)

def annotate_brackets(ax, bracket_specs, test_fn, pvalue_rows):
    """
    For each (x1, x2, group_a_vals, group_b_vals, label) in bracket_specs,
    run test_fn(a, b) and draw a bracket.  Appends rows to pvalue_rows list.
    """
    if not bracket_specs:
        return
    all_vals = [v for spec in bracket_specs
                for grp in [spec[2], spec[3]] for v in grp if np.isfinite(v)]
    if not all_vals:
        return
    y_top = ax.get_ylim()[1]
    y_range = y_top - ax.get_ylim()[0] or 1
    step = y_range * 0.14
    h    = y_range * 0.03

    current_y = y_top + y_range * 0.05

    for (x1, x2, a, b, row_label) in bracket_specs:
        p = test_fn(a, b)
        if not np.isnan(p):
            add_bracket(ax, x1, x2, current_y, h, pval_label_full(p), fontsize=7)
            pvalue_rows.append({**row_label, "p_value": p,
                                "stars": pval_stars(p)})
        current_y += step

    ax.set_ylim(top=current_y + step)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  BASELINE CONDITIONS  (analyze_results.py)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("1. Baseline conditions comparison")

BASE_CONDITIONS = {
    "2D_normal": {"label": "2D\nNormal\nandrogen", "color": "#1F77B4"},
    "2D_ADT":    {"label": "2D\nADT",              "color": "#FF7F0E"},
    "3D_normal": {"label": "3D\nNormal\nandrogen",  "color": "#1F77B4"},
    "3D_ADT":    {"label": "3D\nADT",               "color": "#FF7F0E"},
}
BASE_COMPARISONS = [
    (0, 1, "2D: Normal vs ADT"),
    (2, 3, "3D: Normal vs ADT"),
    (0, 2, "Normal: 2D vs 3D"),
    (1, 3, "ADT: 2D vs 3D"),
]
METRIC_COLS   = ["total", "S", "R", "RS"]
METRIC_LABELS = ["Total cell count", "Sensitive (S)", "Resistant (R)", "R / S ratio"]

# Load data
base_data = {}  # cond → list of DataFrames
for cond in BASE_CONDITIONS:
    dfs = []
    for seed in range(10):
        df = load_csv(f"output_{cond}_seed{seed:02d}/analysis_over_time.csv")
        if df is not None:
            dfs.append(df)
    base_data[cond] = dfs
    print(f"  {cond}: {len(dfs)} seeds")

cond_keys   = list(BASE_CONDITIONS.keys())
cond_colors = [BASE_CONDITIONS[c]["color"] for c in cond_keys]
cond_labels = [BASE_CONDITIONS[c]["label"] for c in cond_keys]

# Build comparison bracket specs per metric
def base_bracket_specs(metric_col):
    data_per = [get_final_list(base_data[c], metric_col) for c in cond_keys]
    specs = []
    for (i, j, label) in BASE_COMPARISONS:
        specs.append((i+1, j+1, data_per[i], data_per[j],
                      {"sweep": "baseline", "metric": metric_col,
                       "comparison": label}))
    return data_per, specs

pvalue_rows = []

fig, axes = plt.subplots(2, 4, figsize=(24, 14))
fig.suptitle(
    "Baseline Conditions — Distributions at t = 15 months\n"
    "Top row: Mann-Whitney U (two-sided)  |  Bottom row: Student's t-test (two-sided)",
    fontsize=13, fontweight="bold")

for col_idx, (col, lbl) in enumerate(zip(METRIC_COLS, METRIC_LABELS)):
    data_per, specs = base_bracket_specs(col)

    for row_idx, (test_fn, test_label) in enumerate([
        (mannwhitney_p, "Mann-Whitney U"),
        (ttest_p,       "Student's t-test"),
    ]):
        ax = axes[row_idx, col_idx]
        draw_violins(ax,
                     positions=list(range(1, len(cond_keys)+1)),
                     data_per_pos=data_per,
                     colors=cond_colors)
        ax.set_xticks(range(1, len(cond_keys)+1))
        ax.set_xticklabels(cond_labels, fontsize=8)
        ax.set_title(f"{lbl}\n[{test_label}]", fontweight="bold", fontsize=9)
        ax.set_ylabel(lbl, fontsize=8)
        ax.grid(True, alpha=0.3, axis="y", ls=":")
        if col == "RS":
            ax.set_ylim(bottom=0)

        # Add comparisons; tag rows with test name
        tagged_specs = [
            (x1, x2, a, b, {**meta, "test": test_label})
            for (x1, x2, a, b, meta) in specs
        ]
        annotate_brackets(ax, tagged_specs, test_fn, pvalue_rows)

for row_idx, label in enumerate(["Mann-Whitney U", "Student's t-test"]):
    axes[row_idx, 0].annotate(label, xy=(0, 0.5), xytext=(-0.25, 0.5),
                              xycoords="axes fraction", textcoords="axes fraction",
                              fontsize=10, fontweight="bold", rotation=90,
                              ha="center", va="center")

legend_handles = [
    Patch(facecolor="#1F77B4", alpha=0.6, label="Normal androgen (8 ng/ml)"),
    Patch(facecolor="#FF7F0E", alpha=0.6, label="ADT (1 ng/ml)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, 0.0), fontsize=10)
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig("figures/compare_ttest_mw_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/compare_ttest_mw_results.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  ADHESION SWEEP  (analyze_adhesion.py)
# ══════════════════════════════════════════════════════════════════════════════
print("\n2. Adhesion sweep comparison")

ADH_MULTS    = [0.1, 0.25, 1.0, 2.0, 5.0, 10.0]
ADH_STRS     = ["0p1", "0p25", "baseline", "2p0", "5p0", "10p0"]
ADH_LABELS   = ["0.1x\n(0.04)", "0.25x\n(0.10)", "1x\n(0.40)\nbaseline",
                "2x\n(0.80)", "5x\n(2.00)", "10x\n(4.00)"]
ADH_X        = np.arange(len(ADH_MULTS))

def load_adh_2d(mult_str, cond):
    dfs = []
    for seed in range(10):
        path = (f"output_{cond}_seed{seed:02d}/analysis_over_time.csv"
                if mult_str == "baseline"
                else f"output_adh{mult_str}_{cond}_seed{seed:02d}/analysis_over_time.csv")
        df = load_csv(path)
        if df is not None:
            dfs.append(df)
    return dfs

adh_2d = {}
for ms in ADH_STRS:
    for cond in ["2D_normal", "2D_ADT"]:
        adh_2d[(ms, cond)] = load_adh_2d(ms, cond)
        print(f"  adh {ms} {cond}: {len(adh_2d[(ms,cond)])} seeds")

WIDTH = 0.35

fig, axes = plt.subplots(2, 4, figsize=(26, 14))
fig.suptitle(
    "Adhesion Sweep — Distributions at t = 15 months (2D seeds only)\n"
    "Top row: Mann-Whitney U (two-sided)  |  Bottom row: Student's t-test (two-sided)\n"
    "Brackets compare Normal androgen vs. ADT at each adhesion level",
    fontsize=12, fontweight="bold")

for col_idx, (col, lbl) in enumerate(zip(METRIC_COLS, METRIC_LABELS)):
    normal_vals = [get_final_list(adh_2d[(ms, "2D_normal")], col) for ms in ADH_STRS]
    adt_vals    = [get_final_list(adh_2d[(ms, "2D_ADT")],    col) for ms in ADH_STRS]

    for row_idx, (test_fn, test_label) in enumerate([
        (mannwhitney_p, "Mann-Whitney U"),
        (ttest_p,       "Student's t-test"),
    ]):
        ax = axes[row_idx, col_idx]

        for xi, (ms, norm_v, adt_v) in enumerate(zip(ADH_STRS, normal_vals, adt_vals)):
            pos_n = xi - 0.5 * WIDTH
            pos_a = xi + 0.5 * WIDTH

            draw_violins(ax, [pos_n], [norm_v], ["#1F77B4"])
            draw_violins(ax, [pos_a], [adt_v],  ["#FF7F0E"])

        ax.set_xticks(ADH_X)
        ax.set_xticklabels(ADH_LABELS, fontsize=7)
        ax.set_xlabel("Adhesion multiplier (absolute value)", fontsize=8)
        ax.set_ylabel(lbl, fontsize=8)
        ax.set_title(f"{lbl}\n[{test_label}]", fontweight="bold", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y", ls=":")
        ax.axvline(x=2, color="grey", ls=":", lw=1.5, alpha=0.6)
        if col == "RS":
            ax.set_ylim(bottom=0)

        # Bracket: normal vs ADT at each adhesion level
        specs = [
            (xi - 0.5*WIDTH, xi + 0.5*WIDTH,
             norm_v, adt_v,
             {"sweep": "adhesion", "metric": col, "test": test_label,
              "comparison": f"Normal vs ADT @ adh={ms}"})
            for xi, (ms, norm_v, adt_v) in enumerate(zip(ADH_STRS, normal_vals, adt_vals))
        ]
        annotate_brackets(ax, specs, test_fn, pvalue_rows)

for row_idx, label in enumerate(["Mann-Whitney U", "Student's t-test"]):
    axes[row_idx, 0].annotate(label, xy=(0, 0.5), xytext=(-0.3, 0.5),
                              xycoords="axes fraction", textcoords="axes fraction",
                              fontsize=10, fontweight="bold", rotation=90,
                              ha="center", va="center")

legend_handles = [
    Patch(facecolor="#1F77B4", alpha=0.6, label="Normal androgen (10 seeds)"),
    Patch(facecolor="#FF7F0E", alpha=0.6, label="ADT (10 seeds)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, 0.0), fontsize=10)
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig("figures/compare_ttest_mw_adhesion.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/compare_ttest_mw_adhesion.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  ADHESION + MOTILITY SWEEP  (analyze_adhmot.py)
# ══════════════════════════════════════════════════════════════════════════════
print("\n3. Adhesion+Motility sweep comparison")

ADHMOT_MULTS  = [0.1, 0.25, 1.0, 2.0, 5.0, 10.0]
ADHMOT_STRS   = ["0p1", "0p25", "baseline", "2p0", "5p0", "10p0"]
ADHMOT_LABELS = ["0.1x\nadh↓ mot↑", "0.25x\nadh↓ mot↑", "1x\nbaseline",
                 "2x\nadh↑ mot↓", "5x\nadh↑ mot↓", "10x\nadh↑ mot↓"]
ADHMOT_X      = np.arange(len(ADHMOT_MULTS))

def load_adhmot_2d(mult_str, cond):
    dfs = []
    for seed in range(10):
        path = (f"output_{cond}_seed{seed:02d}/analysis_over_time.csv"
                if mult_str == "baseline"
                else f"output_adhMot{mult_str}_{cond}_seed{seed:02d}/analysis_over_time.csv")
        df = load_csv(path)
        if df is not None:
            dfs.append(df)
    return dfs

adhmot_2d = {}
for ms in ADHMOT_STRS:
    for cond in ["2D_normal", "2D_ADT"]:
        adhmot_2d[(ms, cond)] = load_adhmot_2d(ms, cond)
        print(f"  adhMot {ms} {cond}: {len(adhmot_2d[(ms,cond)])} seeds")

fig, axes = plt.subplots(2, 4, figsize=(26, 14))
fig.suptitle(
    "Adhesion+Motility Sweep — Distributions at t = 15 months (2D seeds only)\n"
    "Top row: Mann-Whitney U (two-sided)  |  Bottom row: Student's t-test (two-sided)\n"
    "Brackets compare Normal androgen vs. ADT at each multiplier level",
    fontsize=12, fontweight="bold")

for col_idx, (col, lbl) in enumerate(zip(METRIC_COLS, METRIC_LABELS)):
    normal_vals = [get_final_list(adhmot_2d[(ms, "2D_normal")], col) for ms in ADHMOT_STRS]
    adt_vals    = [get_final_list(adhmot_2d[(ms, "2D_ADT")],    col) for ms in ADHMOT_STRS]

    for row_idx, (test_fn, test_label) in enumerate([
        (mannwhitney_p, "Mann-Whitney U"),
        (ttest_p,       "Student's t-test"),
    ]):
        ax = axes[row_idx, col_idx]

        for xi, (norm_v, adt_v) in enumerate(zip(normal_vals, adt_vals)):
            draw_violins(ax, [xi - 0.5*WIDTH], [norm_v], ["#1F77B4"])
            draw_violins(ax, [xi + 0.5*WIDTH], [adt_v],  ["#FF7F0E"])

        ax.set_xticks(ADHMOT_X)
        ax.set_xticklabels(ADHMOT_LABELS, fontsize=7)
        ax.set_xlabel("Adhesion/Motility multiplier", fontsize=8)
        ax.set_ylabel(lbl, fontsize=8)
        ax.set_title(f"{lbl}\n[{test_label}]", fontweight="bold", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y", ls=":")
        ax.axvline(x=2, color="grey", ls=":", lw=1.5, alpha=0.6)
        if col == "RS":
            ax.set_ylim(bottom=0)

        specs = [
            (xi - 0.5*WIDTH, xi + 0.5*WIDTH, norm_v, adt_v,
             {"sweep": "adhMot", "metric": col, "test": test_label,
              "comparison": f"Normal vs ADT @ mult={ADHMOT_STRS[xi]}"})
            for xi, (norm_v, adt_v) in enumerate(zip(normal_vals, adt_vals))
        ]
        annotate_brackets(ax, specs, test_fn, pvalue_rows)

for row_idx, label in enumerate(["Mann-Whitney U", "Student's t-test"]):
    axes[row_idx, 0].annotate(label, xy=(0, 0.5), xytext=(-0.3, 0.5),
                              xycoords="axes fraction", textcoords="axes fraction",
                              fontsize=10, fontweight="bold", rotation=90,
                              ha="center", va="center")

legend_handles = [
    Patch(facecolor="#1F77B4", alpha=0.6, label="Normal androgen (10 seeds)"),
    Patch(facecolor="#FF7F0E", alpha=0.6, label="ADT (10 seeds)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, 0.0), fontsize=10)
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig("figures/compare_ttest_mw_adhmot.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/compare_ttest_mw_adhmot.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  TUMOR RADIUS SWEEP  (analyze_radius.py)
# ══════════════════════════════════════════════════════════════════════════════
print("\n4. Tumor radius sweep comparison")

RADII       = [50, 100, 150, 200, 250]
RADII_LABEL = ["50 µm", "100 µm", "150 µm", "200 µm", "250 µm"]
RADII_X     = np.arange(len(RADII))

radius_2d = {}
for r in RADII:
    for cond in ["2D_normal", "2D_ADT"]:
        dfs = []
        for seed in range(10):
            df = load_csv(f"output_r{r}_{cond}_seed{seed:02d}/analysis_over_time.csv")
            if df is not None:
                dfs.append(df)
        radius_2d[(r, cond)] = dfs
        print(f"  r={r} {cond}: {len(dfs)} seeds")

fig, axes = plt.subplots(2, 4, figsize=(26, 14))
fig.suptitle(
    "Tumor Radius Sweep — Distributions at t = 15 months (2D seeds only)\n"
    "Top row: Mann-Whitney U (two-sided)  |  Bottom row: Student's t-test (two-sided)\n"
    "Brackets compare Normal androgen vs. ADT at each radius",
    fontsize=12, fontweight="bold")

for col_idx, (col, lbl) in enumerate(zip(METRIC_COLS, METRIC_LABELS)):
    normal_vals = [get_final_list(radius_2d[(r, "2D_normal")], col) for r in RADII]
    adt_vals    = [get_final_list(radius_2d[(r, "2D_ADT")],    col) for r in RADII]

    for row_idx, (test_fn, test_label) in enumerate([
        (mannwhitney_p, "Mann-Whitney U"),
        (ttest_p,       "Student's t-test"),
    ]):
        ax = axes[row_idx, col_idx]

        for xi, (norm_v, adt_v) in enumerate(zip(normal_vals, adt_vals)):
            draw_violins(ax, [xi - 0.5*WIDTH], [norm_v], ["#1F77B4"])
            draw_violins(ax, [xi + 0.5*WIDTH], [adt_v],  ["#FF7F0E"])

        ax.set_xticks(RADII_X)
        ax.set_xticklabels(RADII_LABEL, fontsize=8)
        ax.set_xlabel("Initial tumor radius", fontsize=8)
        ax.set_ylabel(lbl, fontsize=8)
        ax.set_title(f"{lbl}\n[{test_label}]", fontweight="bold", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y", ls=":")
        if col == "RS":
            ax.set_ylim(bottom=0)

        specs = [
            (xi - 0.5*WIDTH, xi + 0.5*WIDTH, norm_v, adt_v,
             {"sweep": "radius", "metric": col, "test": test_label,
              "comparison": f"Normal vs ADT @ r={RADII[xi]}µm"})
            for xi, (norm_v, adt_v) in enumerate(zip(normal_vals, adt_vals))
        ]
        annotate_brackets(ax, specs, test_fn, pvalue_rows)

for row_idx, label in enumerate(["Mann-Whitney U", "Student's t-test"]):
    axes[row_idx, 0].annotate(label, xy=(0, 0.5), xytext=(-0.3, 0.5),
                              xycoords="axes fraction", textcoords="axes fraction",
                              fontsize=10, fontweight="bold", rotation=90,
                              ha="center", va="center")

legend_handles = [
    Patch(facecolor="#1F77B4", alpha=0.6, label="Normal androgen (10 seeds)"),
    Patch(facecolor="#FF7F0E", alpha=0.6, label="ADT (10 seeds)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, 0.0), fontsize=10)
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig("figures/compare_ttest_mw_radius.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/compare_ttest_mw_radius.png")


# ══════════════════════════════════════════════════════════════════════════════
# Summary CSV — all p-values from both tests side by side
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding p-value summary CSV...")

df_pvals = pd.DataFrame(pvalue_rows)

# Pivot so MW and t-test sit on the same row
if not df_pvals.empty:
    # Drop dupes that arise from the two test loops
    df_mw = (df_pvals[df_pvals["test"] == "Mann-Whitney U"]
             .drop(columns=["stars"])
             .rename(columns={"p_value": "p_mannwhitney"}))
    df_tt = (df_pvals[df_pvals["test"] == "Student's t-test"]
             .drop(columns=["stars"])
             .rename(columns={"p_value": "p_ttest"}))

    merge_keys = ["sweep", "metric", "comparison"]
    df_summary = pd.merge(
        df_mw.drop(columns=["test"]),
        df_tt.drop(columns=["test"]),
        on=merge_keys, how="outer"
    )
    df_summary["stars_mw"]    = df_summary["p_mannwhitney"].apply(
        lambda p: pval_stars(p) if not np.isnan(p) else "")
    df_summary["stars_ttest"] = df_summary["p_ttest"].apply(
        lambda p: pval_stars(p) if not np.isnan(p) else "")
    df_summary["concordant"] = (
        df_summary["stars_mw"].apply(lambda s: s != "ns") ==
        df_summary["stars_ttest"].apply(lambda s: s != "ns")
    )

    df_summary.to_csv("ttest_mw_pvalue_comparison.csv", index=False)
    print("  Saved: ttest_mw_pvalue_comparison.csv")

    n_total   = len(df_summary)
    n_concord = df_summary["concordant"].sum()
    print(f"\n  Concordance (both significant or both not): "
          f"{n_concord}/{n_total} comparisons ({100*n_concord/n_total:.1f}%)")

print("\nDone! All figures in figures/")
