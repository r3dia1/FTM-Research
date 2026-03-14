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
OUTPUT_DIR = "final_comparison_plots_2026_2_4(final version)"


# 需求 1: 指定的顏色順序
COLORS = [
    '#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', 
    '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD'
]

# 需求 3: 擴充的架構與對應設定
# 請在此處自行調整 'subpath' 為您實際存放 npy 檔案的路徑
ARCH_CONFIG = {
    'DNN': {
        'subpath': 'DNN/results/cdf_data_2_4', 
        'label': 'DNN', 
        'color': COLORS[0]
    },
    'DANN': {
        'subpath': 'DANN/results/dann_v5/cdf_data_2_4(s:t 1:0.5)', 
        'label': 'DANN', 
        'color': COLORS[1]
    },
    'FusionDANN': {
        'subpath': 'FusionDANN/results/fusion_dann_v5/cdf_data_2_4 (s:t 1:0.5)', 
        'label': 'FusionDANN', 
        'color': COLORS[2]
    },
    'CDAN': {
        'subpath': 'FusionCDAN/ablation test(s:t=1:0.5)/CDAN/results/cdf_data',  # 請調整路徑
        'label': 'CDAN', 
        'color': COLORS[3]
    },
    'FusionCDAN': {
        'subpath': 'FusionCDAN/ablation test(s:t=1:0.5)/FusionCDAN/results/cdf_data(64)',  # 請調整路徑
        'label': 'FusionCDAN', 
        'color': COLORS[4]
    },
    'DAFI': {
        'subpath': 'DAFI/results/cdf_data_2_4(1:0.5)',  # 請調整路徑
        'label': 'DAFI', 
        'color': COLORS[5]
    },
    'GACDAN': {
        'subpath': 'FusionCDAN/ablation test(s:t=1:0.5)/GACDAN/results/cdf_data',  # 請調整路徑
        'label': 'GA-CDAN', 
        'color': COLORS[6]
    }
}

# ==========================================
# 2. 檔名解析邏輯 (核心優化)
# ==========================================

def parse_filename(filename):
    """
    通用解析邏輯：
    1. 過濾掉非 Fusion 的檔案 (Only_RTT, Only_RSSI)
    2. 自動提取檔名中的獨立數字並計算數量 (代表 RTT 數量)
    回傳 RTT_Count，若解析失敗或為非 Fusion 則回傳 None
    """
    name = os.path.basename(filename).replace(".npy", "")
    
    # 需求 2: 只要 Fusion 模式，直接排除其他模式
    if "Only_RTT" in name or "Only_RSSI" in name:
        return None
        
    # 移除結尾的 seed 標記 (例如 _seed42) 避免干擾數字判斷
    name_no_seed = re.sub(r'_seed\d+$', '', name)
    
    # 將檔名用底線拆分，並過濾出純數字的部分 (例如 error_Fusion_RTT_1_2 -> ['1', '2'])
    parts = name_no_seed.split('_')
    rtt_list = [p for p in parts if p.isdigit()]
    
    if rtt_list:
        return len(rtt_list)
    
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
    
    # 資料結構: plot_data[rtt_count][label] = [array...]
    plot_data = {}

    print(f"Scanning directories in {BASE_DIR}...")

    # 1. 遍歷所有架構
    for arch_key, config in ARCH_CONFIG.items():
        search_path = os.path.join(BASE_DIR, config['subpath'], "*.npy")
        print(search_path)
        files = glob.glob(search_path)
        
        print(f"  > Checking {arch_key}: Found {len(files)} files in {config['subpath']}")
        
        if len(files) == 0:
            print(f"    [Warning] No files found for {arch_key}. Check path: {search_path}")
            continue

        for f in files:
            rtt_count = parse_filename(f)
            
            # 如果回傳 None，代表該檔案不是 Fusion 模式或無法解析，直接略過
            if rtt_count is None:
                continue
                
            # 因為現在全部都是 Fusion 模式，圖例名稱可以直接用架構名稱，不用再後綴 (Fusion)
            label_key = config['label']

            # 初始化字典
            if rtt_count not in plot_data:
                plot_data[rtt_count] = {}
            if label_key not in plot_data[rtt_count]:
                plot_data[rtt_count][label_key] = {'data': [], 'arch': arch_key}
            
            # 讀取並存入
            try:
                arr = np.load(f)
                plot_data[rtt_count][label_key]['data'].extend(arr)
            except Exception as e:
                print(f"    Error reading {f}: {e}")

    # 2. 開始繪圖 (依照 RTT 數量 分組)
    sorted_counts = sorted(plot_data.keys())
    
    if not sorted_counts:
        print("No valid data found to plot. Please check your files and paths.")
        return

    for count in sorted_counts:
        plt.figure(figsize=(10, 7))
        
        # 取得該數量下的所有線條數據
        lines = plot_data[count]
        
        # 為了讓圖例的順序跟 ARCH_CONFIG 的定義順序一致，我們用原定義的 key 來排序
        # 確保顏色與模型的對應在視覺上是一致且有條理的
        sorted_labels = sorted(lines.keys(), key=lambda k: list(ARCH_CONFIG.keys()).index(lines[k]['arch']))
        
        print(f"Plotting RTT Count: {count} (Lines: {len(sorted_labels)})")
        
        for label in sorted_labels:
            info = lines[label]
            all_data = np.array(info['data'])
            
            # 計算 CDF
            x, y = get_cdf(all_data)
            mean_error = np.mean(all_data)
            
            # 樣式設定 (全部統一為實線，因為沒有 Only_RTT/RSSI 的比較了)
            arch_key = info['arch']
            color = ARCH_CONFIG[arch_key]['color']
            
            # 最終圖例名稱包含平均誤差
            final_label = f"{label} [Mean: {mean_error:.4f}m]"
            
            plt.plot(x, y, label=final_label, color=color, linestyle='-', linewidth=2, alpha=0.8)

        # 圖表修飾
        plt.xlabel('Distance Error (m)', fontsize=14)
        plt.ylabel('CDF', fontsize=14)
        plt.title(f'Performance Comparison (Fusion Mode) - {count} RTT AP(s)', fontsize=16, fontweight='bold')
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        
        # 設定 X 軸範圍 (可根據需求調整，例如只看 0-5米)
        plt.xlim(0, 5)
        plt.ylim(0, 1.02)
        
        plt.legend(loc='lower right', fontsize=11)
        plt.tight_layout()
        
        save_path = os.path.join(OUTPUT_DIR, f"CDF_Fusion_RTT_{count}_APs.png")
        plt.savefig(save_path, dpi=500)
        plt.close()
        print(f"  -> Saved: {save_path}")

    print("\nAll plots generated in directory:", OUTPUT_DIR)

if __name__ == "__main__":
    main()