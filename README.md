# ProstateCancer-ABM

Agent-based model of prostate cancer tumor growth built on [PhysiCell v1.14.2](http://physicell.org/), developed to study the effects of androgen deprivation therapy (ADT), cell adhesion, and motility on tumor dynamics.

## Requirements

| Component | Version |
|-----------|---------|
| PhysiCell | v1.14.2 |
| Compiler | gcc/10.2.0 |
| Python | 3.x (Anaconda) |
| Python packages | numpy, pandas, matplotlib, scipy |
| HPC | Bridges-2 RM-shared partition (SLURM) |

## Build

```bash
module load gcc/10.2.0
make clean && make
```

> **Required patch before compiling:** In `custom_modules/custom.cpp` line 662, change:
> ```cpp
> // Before:
> std::string csv_path = "output/analysis_over_time.csv";
> // After:
> std::string csv_path = PhysiCell_settings.folder + "/analysis_over_time.csv";
> ```
> Without this fix, output CSVs will be written to the wrong directory and all downstream analysis will fail.

## Running Simulations

```bash
# Baseline (40 runs)
bash run_all_sims.sh

# Parameter sweeps
bash run_sweep.sh radius 2D
bash run_sweep.sh radius 3D
bash run_sweep.sh adhesion 2D
bash run_adhesion_3D_single.sh
```

## Analysis

```bash
module load anaconda3
python analyze_results.py     # baseline
python analyze_radius.py      # radius sweep
python analyze_adhesion.py    # adhesion-only sweep
python analyze_adhmot.py      # adhesion+motility sweep
python analyze_ttest_comparison.py  # statistical comparisons
```

Figures are written to `figures/` and summary CSVs to `csv_exports/`.

## Config Directories

| Directory | Description |
|-----------|-------------|
| `configs/` | Baseline (40 runs) |
| `configs_radius/` | Tumor radius sweep (200 runs) |
| `configs_adhesion_only/` | Adhesion-only sweep (2D, 120 runs) |
| `configs_adhesion_3D_single_seed/` | Adhesion-only sweep (3D, 12 runs) |
| `configs_adhMot_3D_single_seed/` | Adhesion+Motility sweep (3D, 12 runs) |

## Notes

- Simulations were run with 28 threads (2D) and 64 threads (3D). PhysiCell does not scale linearly — 8–12 threads is generally optimal for future runs.
- Both oxygen and testosterone Dirichlet boundary conditions must have `enabled="True"` in all config XMLs.

## Citation

See `CITATION.txt` and `ALL_CITATIONS.txt` for PhysiCell attribution.
