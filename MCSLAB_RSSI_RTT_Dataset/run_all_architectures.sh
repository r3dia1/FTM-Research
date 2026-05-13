#!/bin/bash

# ==========================================
# 1. 設定區
# ==========================================

# 資料集路徑 (請務必使用絕對路徑)
BASE_PATH="/home/mcslab/yutung/MCSLAB_RSSI_RTT_Dataset" 

# 定義所有架構列表 (根據您的實際路徑更新)
ARCHITECTURES=(
    # "DNN/dnn_baseline_script_fix.py"
    # "FusionDNN/fusion_dnn.py"
    # "FusionDNN/fusion_dnn_virtual_rtt_test2.py"
    # "DANN/dann_v5.py"
    # "FusionDANN/fusion_dann_v5.py"
    # "CDAN/cdan.py"
    # "FusionCDAN final/fusion_cdan.py"
    # "DAFI/dafi_v2.py"
    "DuGDA/DuGDA_v2.py"
    # "GeoSPA-Net/GeoSPA_Net_v2.py"
)

# 定義 RTT 組合
# c1=("1" "2" "3" "4")
# c2=("1 2" "1 3" "1 4" "2 3" "2 4" "3 4")
# c3=("1 2 3" "1 2 4" "1 3 4" "2 3 4")
c4=("1 2 3 4")
all_combos=("${c1[@]}" "${c2[@]}" "${c3[@]}" "${c4[@]}")

# ==========================================
# 2. 主程式區
# ==========================================

mkdir -p logs

echo "=========================================="
echo "Starting Unified Ablation Study (Updated Paths)"
echo "Data Path: $BASE_PATH"
echo "=========================================="

for arch_path in "${ARCHITECTURES[@]}"; do
    
    dir_name=$(dirname "$arch_path")
    script_name=$(basename "$arch_path")
    
    # 建立該架構的 log 資料夾
    arch_log_dir="logs/${dir_name}"
    mkdir -p "$arch_log_dir"

    echo ""
    echo "######################################################"
    echo "Processing Architecture: [ $dir_name ]"
    echo "Script: $script_name"
    echo "######################################################"

    if [ ! -f "$arch_path" ]; then
        echo "Error: Script $arch_path not found! Skipping..."
        continue
    fi

    # 進入該架構的資料夾
    pushd "$dir_name" > /dev/null

    # ------------------------------------------------------
    # [關鍵邏輯] 判斷是否需要跑多模式
    # 規則：如果資料夾名稱是 "DNN" 或 "DANN"，就跑 Fusion 和 RTT 兩種模式
    # ------------------------------------------------------
    MODES_TO_RUN=()
    
    if [[ "$dir_name" == "DNN" ]] || [[ "$dir_name" == "DANN" ]] || [[ "$dir_name" == "DAFI" ]] || [[ "$dir_name" == "FusionDNN" ]]; then
        MODES_TO_RUN=("fusion" "rtt")
        echo ">> Architecture type: [Baseline/Single Stream]"
        echo ">> Will run modes: [Fusion, RTT]"
    else
        # FusionDANN 和 FusionCDAN 跑預設模式
        MODES_TO_RUN=("default")
        echo ">> Architecture type: [Advanced Fusion]"
        echo ">> Will run modes: [Default]"
    fi

    # ------------------------------------------------------
    # 開始執行
    # ------------------------------------------------------
    for mode in "${MODES_TO_RUN[@]}"; do
        
        if [ "$mode" != "default" ]; then
            echo "  >> [Mode: $mode] Starting..."
        fi

        for combo in "${all_combos[@]}"; do
            
            # 準備檔名
            combo_name=$(echo $combo | tr ' ' '_')
            
            # 根據模式決定參數與 log 檔名
            if [ "$mode" == "default" ]; then
                log_file="../${arch_log_dir}/log_${combo_name}.txt"
                CMD="python $script_name --rtt_indices \"$combo\" --base_path \"$BASE_PATH\""
            else
                log_file="../${arch_log_dir}/log_${mode}_${combo_name}.txt"
                CMD="python $script_name --mode $mode --rtt_indices \"$combo\" --base_path \"$BASE_PATH\""
            fi
            
            echo "    Running combo: [$combo]..."
            eval $CMD > "$log_file" 2>&1
            
        done
    done
    
    popd > /dev/null
    echo "Finished $dir_name"

done

echo ""
echo "=========================================="
echo "All Experiments Completed."
echo "=========================================="