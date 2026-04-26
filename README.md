0~# PhysiCell Prostate Cancer ABM — Reproducibility Guide

This README describes the steps required to reproduce all simulations and analyses from the prostate cancer agent-based modeling study. All code, configs, and scripts live in `~/ProstateCancer-ABM/` on Bridges-2 (PSC).

---

## Requirements

| Component | Version |
|-----------|---------|
| PhysiCell | v1.14.2 |
| Compiler | gcc/10.2.0 |
| Python | 3.x (Anaconda) |
| Python packages | numpy, pandas, matplotlib, scipy |
| HPC | Bridges-2 RM-shared partition (SLURM) |

---

## 1. Build the Binary

Load the compiler module and compile from the project root:

```bash
module load gcc/10.2.0
cd ~/ProstateCancer-ABM
make clean && make
```

The compiled binary will be at `~/ProstateCancer-ABM/project`.

**Critical patch — must be applied before compiling:**  
In `custom_modules/custom.cpp`, line 662, change:
```cpp
// BEFORE (incorrect — writes CSV to hardcoded "output/" directory):
std::string csv_path = "output/analysis_over_time.csv";

// AFTER (correct — writes CSV to the simulation's own output folder):
std::string csv_path = PhysiCell_settings.folder + "/analysis_over_time.csv";
```
Without this fix, the `analysis_over_time.csv` file will be written to the wrong directory and all downstream Python analysis will fail to find it.

---

## 2. Config Generation

All XML config files are pre-generated and stored in the following directories:

| Directory | Sweep |
|-----------|-------|
| `configs/` | Baseline (40 runs) |
| `configs_radius/` | Tumor radius sweep (200 runs) |
| `configs_adhesion_only/` | Adhesion-only sweep (2D, 120 runs) |
| `configs_adhesion_3D_single_seed/` | Adhesion-only sweep (3D, 12 runs) |
| `configs_adhMot_3D_single_seed/` | Adhesion+Motility sweep (3D, 12 runs) |

If you need to regenerate configs from scratch, the Python generation scripts are available locally. Key config parameters to verify before running:

- Both oxygen and testosterone `<Dirichlet_boundary_condition>` nodes must have `enabled="True"`
- The `<folder>` tag in each config must match the expected output folder name exactly

**Radius sweep folder name fix:** If configs were generated with `output_radius{R}_` instead of `output_r{R}_`, apply:
```bash
sed -i 's|<folder>output_radius\([0-9]*\)_|<folder>output_r\1_|g' configs_radius/settings_r*.xml
```

---

## 3. Running Simulations

### Baseline
```bash
bash run_all_sims.sh
```

### Parameter Sweeps
```bash
# Radius sweep (2D and 3D)
bash run_sweep.sh radius 2D
bash run_sweep.sh radius 3D

# Adhesion-only sweep (2D)
bash run_sweep.sh adhesion 2D

# Adhesion-only sweep (3D, single seed)
bash run_adhesion_3D_single.sh

# Adhesion+Motility sweep (3D, single seed)
for CONFIG in configs_adhMot_3D_single_seed/settings_adhMot*.xml; do
  OUTDIR=$(grep -oP '(?<=<folder>)[^<]+' $CONFIG | head -1)
  mkdir -p "$OUTDIR"
  cp config/cell_rules.csv "$OUTDIR/"
  sbatch --job-name="PCa_adhMot3D" \
         --partition=RM-shared \
         --ntasks-per-node=1 \
         --cpus-per-task=64 \
         --time=72:00:00 \
         --output="logs/$(basename $CONFIG .xml).out" \
         --wrap="module load gcc/10.2.0; export OMP_NUM_THREADS=64; ./project $CONFIG"
done
```

**Note on thread count:** The simulations here were run with 28 threads (2D) and 64 threads (3D). In practice, PhysiCell performance does not scale linearly with CPU count — 8–12 threads is generally the optimal range, and future runs are recommended to use this range instead.

---

## 4. Verifying Output Completeness

Before running analysis, verify that all expected `analysis_over_time.csv` files are present and complete (≥100 rows, final time ≥ 648,000 min):

```bash
# Example: check radius 3D outputs
for r in 50 100 150 200 250; do
  for cond in 3D_normal 3D_ADT; do
    for seed in $(seq -f "%02g" 0 9); do
      f="output_r${r}_${cond}_seed${seed}/analysis_over_time.csv"
      [ ! -f "$f" ] && echo "MISSING: r${r} $cond seed${seed}" && continue
      lines=$(wc -l < "$f"); last=$(tail -1 "$f" | cut -d',' -f1)
      { [ "$lines" -lt 100 ] || [ "$last" -lt 648000 ]; } \
        && echo "INCOMPLETE ($lines lines, t=$last): r${r} $cond seed${seed}"
    done
  done
done
```

---

## 5. Running Analysis

```bash
module load anaconda3
cd ~/ProstateCancer-ABM

python analyze_results.py    # baseline
python analyze_radius.py     # radius sweep
python analyze_adhesion.py   # adhesion-only sweep
python analyze_adhMot.py     # adhesion+motility sweep
```

All figures are written to `figures/` and master temporal CSVs to `csv_exports/`.

---

## 6. Output Structure

```
ProstateCancer-ABM/
├── project                          # compiled binary
├── config/                          # master config and cell rules
├── configs_radius/                  # radius sweep XML configs
├── configs_adhesion_only/           # adhesion sweep XML configs
├── configs_adhesion_3D_single_seed/ # adhesion 3D single-seed configs
├── configs_adhMot_3D_single_seed/   # adhMot 3D single-seed configs
├── output_{cond}_seed{NN}/          # baseline simulation outputs
├── output_r{R}_{cond}_seed{NN}/     # radius sweep outputs
├── output_adh{M}_{cond}_seed{NN}/   # adhesion sweep outputs
├── output_adhMot{M}_{cond}_seed{NN}/ # adhMot sweep outputs
├── figures/                         # all analysis figures
├── csv_exports/                     # master temporal CSVs
├── logs/                            # SLURM job output logs
├── analyze_results.py               # baseline analysis
├── analyze_radius.py                # radius sweep analysis
├── analyze_adhesion.py              # adhesion sweep analysis
├── analyze_adhMot.py                # adhesion+motility sweep analysis
├── run_all_sims.sh                  # baseline job submission
├── run_sweep.sh                     # sweep job submission
└── custom_modules/custom.cpp        # PhysiCell custom module (patched)
```

---

## 7. Required Code Edits Before Building

### 7.1 custom.cpp — CSV output path (line 662)

```cpp
// Change this:
std::string csv_path = "output/analysis_over_time.csv";

// To this:
std::string csv_path = PhysiCell_settings.folder + "/analysis_over_time.csv";
```

### 7.2 Config XMLs — Dirichlet boundary conditions

In every config XML, ensure both oxygen and testosterone Dirichlet nodes have `enabled="True"`:

```xml
<Dirichlet_boundary_condition units="mmHg" enabled="True">38</Dirichlet_boundary_condition>
<Dirichlet_boundary_condition units="ng/ml" enabled="True">8</Dirichlet_boundary_condition>
```

### 7.3 Radius sweep configs — folder name correction

```bash
sed -i 's|<folder>output_radius\([0-9]*\)_|<folder>output_r\1_|g' configs_radius/settings_r*.xml
```

### 7.4 Radius sweep — fix nested output folders if rename was applied after jobs ran

```bash
for dir in output_r*/; do
  nested=$(ls "$dir" 2>/dev/null | grep "^output_radius" | head -1)
  if [ -n "$nested" ]; then
    mv "$dir$nested"/* "$dir"
    rmdir "$dir$nested"
  fi
done
```1~
