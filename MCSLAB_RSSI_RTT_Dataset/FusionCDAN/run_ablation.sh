#!/bin/bash

# 設定 Python 腳本路徑
PYTHON_SCRIPT="fusion_cdan_v2.py"
# 設定資料集的 Base Path (請修改為你的實際路徑)
BASE_PATH=".." 

# 定義所有組合
# 1 個 AP
c1=("1" "2" "3" "4")
# 2 個 AP
c2=("1 2" "1 3" "1 4" "2 3" "2 4" "3 4")
# 3 個 AP
c3=("1 2 3" "1 2 4" "1 3 4" "2 3 4")
# 4 個 AP
c4=("1 2 3 4")

# 合併所有組合陣列
all_combos=("${c1[@]}" "${c2[@]}" "${c3[@]}" "${c4[@]}")

# 建立 log 目錄 (存 script 的 log)
mkdir -p logs

echo "=========================================="
echo "Starting RTT Ablation Study"
echo "Total combinations: ${#all_combos[@]}"
echo "=========================================="

for combo in "${all_combos[@]}"; do
    # 將空白轉為底線用於 log 檔名
    combo_name=$(echo $combo | tr ' ' '_')
    
    echo "Running combination: [ $combo ]"
    
    # 執行 Python
    # nohup 可以讓他在背景跑，但這裡我們直接跑方便看進度
    # 2>&1 | tee 讓我們可以看到輸出同時存檔
    python $PYTHON_SCRIPT --rtt_indices "$combo" --base_path "$BASE_PATH" > "logs/log_${combo_name}.txt"
    
    echo "Finished [ $combo ]. Log saved to logs/log_${combo_name}.txt"
    echo "------------------------------------------"
done

echo "All experiments completed."