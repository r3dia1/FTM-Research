#!/bin/bash

PYTHON_SCRIPT="dnn_baseline_script_fix.py"
# 請修改為您的資料集根目錄
# BASE_PATH="/home/mcslab/yutung/MCSLAB_RSSI_RTT_Dataset" 
BASE_PATH=".."

# 定義 RTT 組合
c4=("1 2 3 4")
# c3=("1 2 3" "1 2 4" "1 3 4" "2 3 4")
# c2=("1 2" "1 3" "1 4" "2 3" "2 4" "3 4")
# c1=("1" "2" "3" "4")

all_combos=("${c4[@]}" "${c3[@]}" "${c2[@]}" "${c1[@]}")

mkdir -p logs_dnn

echo "=========================================="
echo "Starting Baseline DNN Ablation Study"
echo "=========================================="

# -----------------------------------------------
# 1. Fusion Mode (RSSI + RTT)
# -----------------------------------------------
# echo ">>> Running FUSION Mode (RSSI + RTT)..."
# for combo in "${all_combos[@]}"; do
#     combo_name=$(echo $combo | tr ' ' '_')
#     echo "  [Fusion] RTT: $combo"
    
#     python $PYTHON_SCRIPT \
#         --mode fusion \
#         --rtt_indices "$combo" \
#         --base_path "$BASE_PATH" \
#         > "logs_dnn/log_fusion_${combo_name}.txt"
# done

# -----------------------------------------------
# 2. Pure RTT Mode
# -----------------------------------------------
# echo ">>> Running Pure RTT Mode..."
# for combo in "${all_combos[@]}"; do
#     combo_name=$(echo $combo | tr ' ' '_')
#     echo "  [RTT Only] RTT: $combo"
    
#     python $PYTHON_SCRIPT \
#         --mode rtt \
#         --rtt_indices "$combo" \
#         --base_path "$BASE_PATH" \
#         > "logs_dnn/log_rtt_${combo_name}.txt"
# done

# -----------------------------------------------
# 3. Pure RSSI Mode (Fixed)
# -----------------------------------------------
echo ">>> Running Pure RSSI Mode..."
# 這裡 rtt_indices 任意給一個即可，程式會忽略
python $PYTHON_SCRIPT \
    --mode rssi \
    --rtt_indices "1" \
    --base_path "$BASE_PATH" \
    > "logs_dnn/log_rssi_fixed.txt"

echo "=========================================="
echo "All DNN experiments finished."
echo "Check results_dnn/dnn_experiment_summary.csv"
echo "=========================================="