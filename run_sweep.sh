#!/bin/bash
# =============================================================================
# run_sweep.sh
# Submits parameter sweep jobs for packing density studies.
#
# Usage:
#   bash run_sweep.sh radius          # Option 3: tumor radius sweep
#   bash run_sweep.sh adhesion_only   # Option 2a: adhesion only
#   bash run_sweep.sh adhesion_motility # Option 2b: adhesion + inverse motility
#   bash run_sweep.sh radius 2D       # Only 2D jobs for radius sweep
#   bash run_sweep.sh radius 3D       # Only 3D jobs for radius sweep
# =============================================================================

STUDY=${1:-"radius"}
DIM_FILTER=${2:-"all"}   # "2D", "3D", or "all"

EXEC="./project"
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

# Config directory and parameter values per study
if [ "$STUDY" == "radius" ]; then
    CONFIG_DIR="./configs_radius"
    PARAMS=("r50" "r100" "r150" "r200" "r250")
    PARAM_LABEL="radius"
elif [ "$STUDY" == "adhesion_only" ]; then
    CONFIG_DIR="./configs_adhesion_only"
    PARAMS=("adh0p1" "adh0p25" "adh2p0" "adh5p0" "adh10p0")
    PARAM_LABEL="adh"
elif [ "$STUDY" == "adhesion_motility" ]; then
    CONFIG_DIR="./configs_adhesion_motility"
    PARAMS=("adhMot0p1" "adhMot0p25" "adhMot2p0" "adhMot5p0" "adhMot10p0")
    PARAM_LABEL="adhMot"
else
    echo "Unknown study: $STUDY. Use: radius | adhesion_only | adhesion_motility"
    exit 1
fi

CONDITIONS=("2D_normal" "2D_ADT" "3D_normal" "3D_ADT")
TOTAL=0

for PARAM in "${PARAMS[@]}"; do
    for COND in "${CONDITIONS[@]}"; do

        # Apply dimension filter
        if [ "$DIM_FILTER" == "2D" ] && [[ "$COND" == 3D* ]]; then continue; fi
        if [ "$DIM_FILTER" == "3D" ] && [[ "$COND" == 2D* ]]; then continue; fi

        # Set walltime based on dimension
        if [[ "$COND" == 2D* ]]; then WALLTIME="2:00:00"; else WALLTIME="24:00:00"; fi

        for SEED in $(seq -f "%02g" 0 9); do
            CONFIG="${CONFIG_DIR}/settings_${PARAM}_${COND}_seed${SEED}.xml"
            OUTDIR="output_${PARAM}_${COND}_seed${SEED}"
            JOBNAME="PCa_${PARAM}_${COND}_s${SEED}"
            # Truncate job name to 20 chars (SLURM limit)
            JOBNAME="${JOBNAME:0:20}"

            if [ ! -f "$CONFIG" ]; then
                echo "WARNING: Config not found: $CONFIG — skipping"
                continue
            fi

            mkdir -p "$OUTDIR"
            cp ./config/cell_rules.csv "$OUTDIR/"

            sbatch --job-name="$JOBNAME" \
                   --partition=RM-shared \
                   --ntasks-per-node=4 \
                   --cpus-per-task=7 \
                   --time=$WALLTIME \
                   --mail-type=END,FAIL \
                   --mail-user=adarshra@seas.upenn.edu \
                   --output="${LOG_DIR}/${JOBNAME}.out" \
                   --error="${LOG_DIR}/${JOBNAME}.err" \
                   --wrap="module load gcc/10.2.0; export OMP_NUM_THREADS=4; $EXEC $CONFIG"

            echo "Submitted: $JOBNAME ($WALLTIME)"
            TOTAL=$((TOTAL + 1))
        done
    done
done

echo ""
echo "Study: $STUDY | Dim filter: $DIM_FILTER"
echo "Total jobs submitted: $TOTAL"
echo "Monitor: squeue -u aramamur"
