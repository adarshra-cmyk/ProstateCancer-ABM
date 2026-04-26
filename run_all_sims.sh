#!/bin/bash
CONDITIONS=("2D_normal" "2D_ADT" "3D_normal" "3D_ADT")
if [ -n "$1" ]; then CONDITIONS=("$1"); fi

EXEC="./project"
CONFIG_DIR="./all_configs"
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

TOTAL=0
for COND in "${CONDITIONS[@]}"; do
  if [[ "$COND" == 2D* ]]; then WALLTIME="6:00:00"; else WALLTIME="24:00:00"; fi

  for SEED in $(seq -f "%02g" 0 9); do
    CONFIG="${CONFIG_DIR}/settings_${COND}_seed${SEED}.xml"
    OUTDIR="output_${COND}_seed${SEED}"
    JOBNAME="PCa_${COND}_s${SEED}"

    if [ ! -f "$CONFIG" ]; then echo "WARNING: Config not found: $CONFIG — skipping"; continue; fi

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
echo "Total jobs submitted: $TOTAL"
