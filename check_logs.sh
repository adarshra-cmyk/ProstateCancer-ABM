#!/bin/bash
# Check all log files for:
# 1. Multiple configs sharing a log (truncated job names)
# 2. Jobs that didn't reach 648000 min (timed out or failed)

LOG_DIR=~/ProstateCancer-ABM/logs
FINAL_TIME=648000

echo "=============================================="
echo "  Checking all log files for issues"
echo "=============================================="

TIMED_OUT=()
COMPLETED=0
MULTI_CONFIG=0

for log in $LOG_DIR/PCa_*.out; do
    # Get all configs that ran in this log
    configs=$(grep "Using config file" $log 2>/dev/null | grep -oP '(?<=Using config file )\S+(?= \.\.\.)')
    
    n_configs=$(echo "$configs" | grep -c "xml" 2>/dev/null || echo 0)
    
    if [ "$n_configs" -gt 1 ]; then
        MULTI_CONFIG=$((MULTI_CONFIG + 1))
    fi
    
    # For each config in this log, check if it reached final time
    while IFS= read -r config; do
        [ -z "$config" ] && continue
        
        # Get the last simulated time for this config's run
        # Find the section of the log corresponding to this config
        last_time=$(grep "current simulated time" $log | tail -1 | grep -oP '\d+(?= min \(max)')
        
        if [ -z "$last_time" ]; then
            echo "NO OUTPUT: $config (log: $(basename $log))"
            TIMED_OUT+=("$config")
        elif [ "$last_time" -lt "$FINAL_TIME" ]; then
            echo "TIMED OUT at ${last_time} min ($(echo "scale=0; $last_time*100/648000" | bc)%): $config"
            TIMED_OUT+=("$config")
        else
            COMPLETED=$((COMPLETED + 1))
        fi
    done <<< "$configs"
done

echo ""
echo "=============================================="
echo "  Summary"
echo "=============================================="
echo "Completed: $COMPLETED"
echo "Timed out / failed: ${#TIMED_OUT[@]}"
echo "Log files with multiple configs: $MULTI_CONFIG"
echo ""
echo "Configs to resubmit:"
for c in "${TIMED_OUT[@]}"; do
    echo "  $c"
done
