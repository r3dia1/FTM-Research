import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re

# ==========================================
# 1. 設定與路徑配置
# ==========================================

# 資料集根目錄 (預設為當前目錄)
BASE_DIR = "." 
OUTPUT_DIR = "final_comparison_plots_2026_2_4"

# 定義每個架構的路徑與顯示名稱
# 格式: '資料夾名稱': {'subpath': '該架構下存放npy的路徑', 'label': '圖例顯示名稱', 'color': '指定顏色'}
ARCH_CONFIG = {
    'DNN': {
        'subpath': 'results/cdf_data_2_4', 
        'label': 'DNN', 
        'color': '#d62728' # 紅色
    },
    'DANN': {
        'subpath': 'results/cdf_data_2_4', 
        'label': 'DANN (Single)', 
        'color': '#1f77b4' # 藍色
    },
    'FusionDANN': {
        'subpath': 'results/cdf_data_2_4', 
        'label': 'Fusion DANN (Dual)', 
        'color': '#2ca02c' # 綠色
    },
    'FusionCDAN': {
        'subpath': 'results/cdf_data_2_4',  # 請確認您的 CDAN 實際輸出路徑
        'label': 'Fusion CDAN', 
        'color': '#9467bd' # 紫色
    }
}

# 線條樣式對應模式
LINE_STYLES = {
    'Fusion': '-',      # 實線
    'Only_RTT': '--',   # 虛線
    'Only_RSSI': ':'    # 點線
}

# ==========================================
# 2. 檔名解析邏輯 (核心)
# ==========================================

def parse_filename(arch_name, filename):
    """
    根據不同架構的命名規則，解析出: (Mode, RTT_Count)
    回傳 None 表示解析失敗或略過
    """
    name = os.path.basename(filename).replace(".npy", "")
    
    # 先移除 seed 結尾，避免干擾數字判斷
    name_no_seed = re.sub(r'_seed\d+$', '', name)
    
    mode = "Unknown"
    rtt_list = []

    # --- 針對 DNN 的規則 ---
    if arch_name == 'DNN':
        if "DNN_Fusion_RTT" in name:
            mode = "Fusion"
            rtt_part = name_no_seed.replace("error_DNN_Fusion_RTT_", "")
            rtt_list = rtt_part.split("_")
        elif "DNN_Only_RTT" in name:
            mode = "Only_RTT"
            rtt_part = name_no_seed.replace("error_DNN_Only_RTT_", "")
            rtt_list = rtt_part.split("_")
        elif "DNN_Only_RSSI" in name:
            return ("Only_RSSI", 0) # 特殊處理

    # --- 針對 DANN 的規則 ---
    elif arch_name == 'DANN':
        if "error_Fusion_RTT" in name:
            mode = "Fusion"
            rtt_part = name_no_seed.replace("error_Fusion_RTT_", "")
            rtt_list = rtt_part.split("_")
        elif "error_Only_RTT" in name:
            mode = "Only_RTT"
            rtt_part = name_no_seed.replace("error_Only_RTT_", "")
            rtt_list = rtt_part.split("_")

    # --- 針對 FusionDANN (Dual) 的規則 ---
    elif arch_name == 'FusionDANN':
        if "FusionDANN" in name:
            mode = "Fusion"
            rtt_part = name_no_seed.replace("error_FusionDANN_", "")
            rtt_list = rtt_part.split("_")

    # --- 針對 FusionCDAN 的規則 ---
    elif arch_name == 'FusionCDAN':
        # 格式: error_1_2_3_4_seed... (Implicit Fusion)
        if "FusionCDAN" in name:
            mode = "Fusion"
            rtt_part = name_no_seed.replace("error_FusionCDAN", "")
            rtt_list = rtt_part.split("_")
        # if name.startswith("error_") and "RTT" not in name:
        #     mode = "Fusion"
        #     rtt_part = name_no_seed.replace("error_FusionCDAN", "")
        #     rtt_list = rtt_part.split("_")

    # 計算 RTT 數量
    if rtt_list:
        # 過濾掉空字串或非數字 (以防萬一)
        rtt_list = [x for x in rtt_list if x.isdigit()]
        return mode, len(rtt_list)
    
    return None

# ==========================================
# 3. 繪圖與數據處理
# ==========================================

def get_cdf(data):
    data_sorted = np.sort(data)
    p = 1. * np.arange(len(data)) / (len(data) - 1)
    return data_sorted, p

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 資料結構: plot_data[rtt_count][line_label] = [array...]
    # line_label 範例: "DNN (Fusion)"
    plot_data = {}

    print(f"Scanning directories in {BASE_DIR}...")

    # 1. 遍歷所有架構
    for arch_key, config in ARCH_CONFIG.items():
        search_path = os.path.join(BASE_DIR, arch_key, config['subpath'], "*.npy")
        files = glob.glob(search_path)
        
        print(f"  > Checking {arch_key}: Found {len(files)} files in {config['subpath']}")
        
        if len(files) == 0:
            print(f"    [Warning] No files found for {arch_key}. Check path: {search_path}")
            continue

        for f in files:
            res = parse_filename(arch_key, f)
            if res is None:
                continue
                
            mode, rtt_count = res
            
            # 建立顯示名稱 (Legend Label)
            # 例如: "DNN (Fusion)" 或 "DANN (Only RTT)"
            if mode == "Only_RSSI":
                label_key = f"{config['label']} (RSSI Only)"
                # RSSI Only 通常會畫在所有的 RTT 圖中當基準，或單獨處理
                # 這裡我們先跳過，專注於 RTT 數量的比較
                continue 
            else:
                label_key = f"{config['label']} ({mode})"

            # 初始化字典
            if rtt_count not in plot_data:
                plot_data[rtt_count] = {}
            if label_key not in plot_data[rtt_count]:
                plot_data[rtt_count][label_key] = {'data': [], 'arch': arch_key, 'mode': mode}
            
            # 讀取並存入
            try:
                arr = np.load(f)
                plot_data[rtt_count][label_key]['data'].extend(arr)
            except Exception as e:
                print(f"    Error reading {f}: {e}")

    # 2. 開始繪圖 (依照 RTT 數量 分組)
    sorted_counts = sorted(plot_data.keys())
    
    if not sorted_counts:
        print("No valid data found to plot.")
        return

    for count in sorted_counts:
        plt.figure(figsize=(10, 7))
        
        # 取得該數量下的所有線條數據
        lines = plot_data[count]
        # 排序：讓同一架構的線條靠在一起
        sorted_labels = sorted(lines.keys())
        
        print(f"Plotting RTT Count: {count} (Lines: {len(sorted_labels)})")
        
        for label in sorted_labels:
            info = lines[label]
            all_data = np.array(info['data'])
            
            # 計算 CDF
            x, y = get_cdf(all_data)
            mean_error = np.mean(all_data)
            
            # 樣式設定
            arch_key = info['arch']
            mode = info['mode']
            
            color = ARCH_CONFIG[arch_key]['color']
            linestyle = LINE_STYLES.get(mode, '-')
            
            # 最終圖例名稱包含平均誤差
            final_label = f"{label} [Mean: {mean_error:.4f}m]"
            
            plt.plot(x, y, label=final_label, color=color, linestyle=linestyle, linewidth=2, alpha=0.8)

        # 圖表修飾
        plt.xlabel('Distance Error (m)', fontsize=14)
        plt.ylabel('CDF', fontsize=14)
        plt.title(f'Performance Comparison - {count} RTT AP(s)', fontsize=16, fontweight='bold')
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        
        # 設定 X 軸範圍 (可根據需求調整，例如只看 0-5米)
        plt.xlim(0, 5)
        plt.ylim(0, 1.02)
        
        # plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)
        plt.legend(loc='lower right', fontsize=11)
        plt.tight_layout()
        
        save_path = os.path.join(OUTPUT_DIR, f"CDF_RTT_{count}_APs.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  -> Saved: {save_path}")

    print("\nAll plots generated in directory:", OUTPUT_DIR)

if __name__ == "__main__":
    main()