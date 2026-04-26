#!/bin/bash
echo "=== Radius 3D Progress Check ==="
echo "Time: $(date)"
echo ""

# Check queue
running=$(squeue -u aramamur | grep -c " R ")
pending=$(squeue -u aramamur | grep -c " PD ")
echo "Jobs running: $running  |  pending: $pending"
echo ""

# Check if CSVs are being written
echo "=== CSV files written so far ==="
count=0
for r in 50 100 150 200 250; do
  for cond in 3D_normal 3D_ADT; do
    for seed in $(seq -f "%02g" 0 9); do
      f="output_r${r}_${cond}_seed${seed}/analysis_over_time.csv"
      if [ -f "$f" ]; then
        lines=$(wc -l < "$f")
        count=$((count+1))
        echo "  OK ($lines lines): r${r} $cond seed${seed}"
      fi
    done
  done
done
echo ""
echo "Total CSVs found: $count / 100"

# Check simulated time progress on a few logs
echo ""
echo "=== Simulated time progress (sample) ==="
for f in $(ls ~/ProstateCancer-ABM/logs/radius_r*_3D_*.out 2>/dev/null | head -6); do
  last=$(grep "current simulated time" $f 2>/dev/null | tail -1 | grep -oP '\d+(?= min \(max)')
  [ -n "$last" ] && pct=$(echo "scale=0; $last*100/648000" | bc) || pct="0"
  echo "  $pct%  $(basename $f)"
done
