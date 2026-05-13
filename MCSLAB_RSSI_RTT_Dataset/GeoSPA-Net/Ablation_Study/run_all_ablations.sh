#!/bin/bash

# ==========================================
# 1. 設定區
# ==========================================

# 資料集路徑 (請務必使用絕對路徑)
BASE_PATH="/home/mcslab/yutung/MCSLAB_RSSI_RTT_Dataset" 

# 定義所有架構列表
ARCHITECTURES=(
    # "sparse baseline/sparse_baseline.py"
    # "naive densification/naive_densification.py"
    "GeoSPA_Net/GeoSPA_Net_v3.py"
)

# 定義 RTT 組合
c1=("1" "2" "3" "4")
c2=("1 2" "1 3" "1 4" "2 3" "2 4" "3 4")
c3=("1 2 3" "1 2 4" "1 3 4" "2 3 4")
c4=("1 2 3 4")

# 將想跑的組合放入 all_combos 中 (目前預設跑 c2)
all_combos=("${c1[@]}" "${c2[@]}" "${c3[@]}" "${c4[@]}")

# ==========================================
# 2. 主程式區
# ==========================================

mkdir -p logs

echo "=========================================="
echo "Starting Unified Ablation Study"
echo "Data Path: $BASE_PATH"
echo "=========================================="

for arch_path in "${ARCHITECTURES[@]}"; do
    
    dir_name=$(dirname "$arch_path")
    script_name=$(basename "$arch_path")
    arch_log_dir="logs/${dir_name}"
    mkdir -p "$arch_log_dir"

    echo ""
    echo "######################################################"
    echo "Processing Architecture: [ $dir_name ]"
    echo "######################################################"

    if [ ! -f "$arch_path" ]; then
        echo "Error: Script $arch_path not found! Skipping..."
        continue
    fi

    pushd "$dir_name" > /dev/null

    # 判斷是否需要跑多模式
    MODES_TO_RUN=("default")
    if [[ "$dir_name" == "DNN" ]] || [[ "$dir_name" == "DANN" ]]; then
        MODES_TO_RUN=("fusion" "rtt")
    fi

    # 開始執行模式
    for mode in "${MODES_TO_RUN[@]}"; do
        
        echo "  >> Running Mode: [$mode]"

        for combo in "${all_combos[@]}"; do
            combo_name=$(echo $combo | tr ' ' '_')
            
            if [ "$mode" == "default" ]; then
                log_file="../${arch_log_dir}/log_${combo_name}.txt"
                CMD="python $script_name --rtt_indices \"$combo\" --base_path \"$BASE_PATH\""
            else
                log_file="../${arch_log_dir}/log_${mode}_${combo_name}.txt"
                CMD="python $script_name --mode $mode --rtt_indices \"$combo\" --base_path \"$BASE_PATH\""
            fi
            
            echo "    - Executing combo: [$combo] -> Saving to $(basename $log_file)"
            eval $CMD > "$log_file" 2>&1
        done

        # ==========================================
        # 3. 統計計算 (Mean & STD across combos)
        # ==========================================
        # 使用 Python 內聯腳本解析剛剛產生的 log，計算該架構與模式的整體平均與標準差
        python3 -c "
import sys, glob, re, numpy as np

log_dir = sys.argv[1]
mode = sys.argv[2]
pattern = f'{log_dir}/log_{mode}_*.txt' if mode != 'default' else f'{log_dir}/log_*.txt'
files = glob.glob(pattern)

data_by_count = {}

for f in files:
    base = f.split('/')[-1].replace('.txt', '')
    prefix = f'log_{mode}_' if mode != 'default' else 'log_'
    
    # 確保不會讀到不相干的 log (例如有其他模式的檔案混入)
    if not base.startswith(prefix):
        continue
        
    combo_name = base[len(prefix):] # 取得如 '1_2' 或 '1_2_4'
    count = len(combo_name.split('_'))
    
    # 讀取 Log 檔內的 Target MDE 數值
    mdes = []
    with open(f, 'r') as file:
        for line in file:
            # 抓取如 'Tgt MDE: 1.2345' 的輸出格式
            match = re.search(r'Tgt MDE:\s*([\d.]+)', line)
            if match:
                mdes.append(float(match.group(1)))
    
    # 計算單一組合(多 Seed) 的平均，並歸類到對應的 mcAP 數量下
    if mdes:
        combo_mean = np.mean(mdes)
        if count not in data_by_count:
            data_by_count[count] = []
        data_by_count[count].append(combo_mean)

print(f'\n    >>> Statistics Summary for Mode: [{mode}] <<<')
if not data_by_count:
    print('    [!] No target MDE data found in logs.')
else:
    for count in sorted(data_by_count.keys()):
        means = data_by_count[count]
        overall_mean = np.mean(means)
        # 組合標準差 (ddof=1)
        overall_std = np.std(means, ddof=1) if len(means) > 1 else 0.0
        print(f'      mcAP Count {count}:')
        print(f'        - Total Combinations : {len(means)}')
        print(f'        - Overall Mean MDE   : {overall_mean:.4f} m')
        print(f'        - Combo-to-Combo STD : {overall_std:.4f} m')
print('')
" "../${arch_log_dir}" "$mode"

    done
    
    popd > /dev/null
    echo "Finished architecture: $dir_name"

done

echo "=========================================="
echo "All Experiments Completed."
echo "=========================================="