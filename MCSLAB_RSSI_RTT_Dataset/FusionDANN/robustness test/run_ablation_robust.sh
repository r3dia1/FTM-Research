#!/bin/bash

# 設定 Python 腳本路徑 (指向剛修改好的 DNN 程式)
PYTHON_SCRIPT="fusion_dann_robust_test.py"

# 設定資料集的 Base Path (請修改為你的實際路徑)
BASE_PATH="../.." 
TEST_PATH="../../2026_4_1/All_Data_With_RSSI_Diff_withoutNA.csv"

# 設定 DNN 實驗模式: 支援 fusion, rtt, rssi
MODE="fusion"

# ==========================================
# 定義所有組合
# ==========================================
# 1 個 AP
c1=("1" "2" "3" "4")
# 2 個 AP
c2=("1 2" "1 3" "1 4" "2 3" "2 4" "3 4")
# 3 個 AP
c3=("1 2 3" "1 2 4" "1 3 4" "2 3 4")
# 4 個 AP
c4=("1 2 3 4")

# 合併所有組合陣列 (目前預設只跑 c4，若要全跑請改成: all_combos=("${c1[@]}" "${c2[@]}" "${c3[@]}" "${c4[@]}"))
all_combos=("${c4[@]}")

# 建立 log 目錄 (存 script 的 log)
mkdir -p logs

echo "=========================================="
echo "Starting FusionDANN Ablation Study"
echo "Script: $PYTHON_SCRIPT"
echo "Mode: $MODE"
echo "Total combinations: ${#all_combos[@]}"
echo "=========================================="

for combo in "${all_combos[@]}"; do
    # 將空白轉為底線用於 log 檔名
    combo_name=$(echo $combo | tr ' ' '_')
    
    echo "Running combination: [ $combo ]"
    
    # 執行 Python，加入 --mode 參數
    python $PYTHON_SCRIPT \
        --rtt_indices "$combo" \
        --base_path "$BASE_PATH" \
        --future_csv "$TEST_PATH" \
        > "logs/log_${MODE}_${combo_name}.txt"
    
    echo "Finished [ $combo ]. Log saved to logs/log_${MODE}_${combo_name}.txt"
    echo "------------------------------------------"
done

echo "All experiments completed."