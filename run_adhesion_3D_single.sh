#!/bin/bash
# Submits 10 adhesion 3D single-seed jobs (5 adhesion values x 2 conditions)
# 3 day walltime, 12 OMP threads, 12 cpus-per-task, up to 8 nodes per job

cd ~/ProstateCancer-ABM
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
TOTAL=0

for CONFIG in configs_adhesion_3D_single_seed/settings_adh*.xml; do
    OUTDIR=$(grep -oP '(?<=<folder>)[^<]+' $CONFIG | head -1)
    LOGNAME="adh3D_single_$(basename $CONFIG .xml)"

    mkdir -p "$OUTDIR"
    cp config/cell_rules.csv "$OUTDIR/"

    sbatch --job-name="PCa_adh3D" \
           --partition=RM-shared \
           --ntasks-per-node=1 \
           --cpus-per-task=64 \
           --time=72:00:00 \
           --mail-type=END,FAIL \
           --mail-user=adarshra@seas.upenn.edu \
           --output="${LOG_DIR}/${LOGNAME}.out" \
           --error="${LOG_DIR}/${LOGNAME}.err" \
           --wrap="module load gcc/10.2.0; export OMP_NUM_THREADS=64; ./project $CONFIG"

    echo "Submitted: $CONFIG"
    TOTAL=$((TOTAL + 1))
done

echo ""
echo "Total jobs submitted: $TOTAL"
echo "Walltime: 72hrs (3 days) | Threads: 12 | Nodes: up to 8"
squeue -u aramamur
