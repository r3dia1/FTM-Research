#!/bin/bash

PYTHON_SCRIPT="fusion_dann_v5.py"
BASE_PATH=".."  # 請根據實際資料位置修改

# 定義 RTT 組合
c4=("1 2 3 4")
c3=("1 2 3" "1 2 4" "1 3 4" "2 3 4")
c2=("1 2" "1 3" "1 4" "2 3" "2 4" "3 4")
c1=("1" "2" "3" "4")

all_combos=("${c4[@]}" "${c3[@]}" "${c2[@]}" "${c1[@]}")

mkdir -p logs_dual

echo "=========================================="
echo "Starting Dual Stream DANN Ablation (Fixed RSSI + Various RTT)"
echo "=========================================="

for combo in "${all_combos[@]}"; do
    combo_name=$(echo $combo | tr ' ' '_')
    echo ">>> Running with RTT: $combo"
    
    python $PYTHON_SCRIPT \
        --rtt_indices "$combo" \
        --base_path "$BASE_PATH" \
        > "logs_dual/log_${combo_name}.txt"
    
    echo "Finished. Log: logs_dual/log_${combo_name}.txt"
done

echo "=========================================="
echo "All experiments finished."
echo "=========================================="