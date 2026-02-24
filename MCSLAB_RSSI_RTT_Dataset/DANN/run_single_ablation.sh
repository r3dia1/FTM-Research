#!/bin/bash

# PYTHON_SCRIPT="dann_v4_with_script.py"
PYTHON_SCRIPT="dann_v5.py"
BASE_PATH=".."  # 請自行修改路徑

# 定義 RTT 組合
c4=("1 2 3 4")
c3=("1 2 3" "1 2 4" "1 3 4" "2 3 4")
c2=("1 2" "1 3" "1 4" "2 3" "2 4" "3 4")
c1=("1" "2" "3" "4")

all_combos=("${c4[@]}" "${c3[@]}" "${c2[@]}" "${c1[@]}")

mkdir -p logs_single

echo "=========================================="
echo "Starting Single Stream DANN Experiments"
echo "=========================================="

# -----------------------------------------------
# 模式 1: Fusion (RSSI + RTT) - 遍歷所有 RTT 組合
# -----------------------------------------------
echo ">>> Running FUSION Mode (RSSI + RTT)..."
for combo in "${all_combos[@]}"; do
    combo_name=$(echo $combo | tr ' ' '_')
    echo "  [Fusion] RTT: $combo"
    
    python $PYTHON_SCRIPT \
        --mode fusion \
        --rtt_indices "$combo" \
        --base_path "$BASE_PATH" \
        > "logs_single/log_fusion_${combo_name}.txt"
done

# -----------------------------------------------
# 模式 2: Pure RTT - 遍歷所有 RTT 組合
# -----------------------------------------------
# echo ">>> Running Pure RTT Mode..."
# for combo in "${all_combos[@]}"; do
#     combo_name=$(echo $combo | tr ' ' '_')
#     echo "  [RTT Only] RTT: $combo"
    
#     python $PYTHON_SCRIPT \
#         --mode rtt \
#         --rtt_indices "$combo" \
#         --base_path "$BASE_PATH" \
#         > "logs_single/log_rtt_${combo_name}.txt"
# done

# -----------------------------------------------
# 模式 3: Pure RSSI - 只跑一次 (因 RSSI 固定)
# -----------------------------------------------
# echo ">>> Running Pure RSSI Mode..."
# # 這裡 rtt_indices 隨便給一個即可，因為 mode=rssi 會忽略它
# python $PYTHON_SCRIPT \
#     --mode rssi \
#     --rtt_indices "1" \
#     --base_path "$BASE_PATH" \
#     > "logs_single/log_rssi_fixed.txt"

echo "=========================================="
echo "All experiments finished."
echo "=========================================="