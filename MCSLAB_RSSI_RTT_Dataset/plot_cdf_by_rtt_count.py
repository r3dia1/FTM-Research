import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re

# ==========================================
# 設定區
# ==========================================
# 資料來源資料夾 (請修改為您的實際路徑)
CDF_DATA_DIR = "cdf_data" 
# 輸出圖片資料夾
OUTPUT_DIR = "cdf_plots"

# 定義顏色與線條樣式，讓不同架構容易區分
STYLE_MAP = {
    'Fusion': {'color': '#1f77b4', 'linestyle': '-', 'label': 'Fusion (RSSI+RTT)'},  # 藍色實線
    'Only_RTT': {'color': '#ff7f0e', 'linestyle': '--', 'label': 'Only RTT'},         # 橘色虛線
    'Only_RSSI': {'color': '#2ca02c', 'linestyle': ':', 'label': 'Only RSSI (Fixed)'} # 綠色點線
}

def get_cdf(data):
    """計算 CDF 的 X, Y 值"""
    data_sorted = np.sort(data)
    p = 1. * np.arange(len(data)) / (len(data) - 1)
    return data_sorted, p

def parse_filename(filename):
    """
    解析檔名，回傳: (架構模式, RTT組合列表, Seed)
    範例: error_Fusion_RTT_1_2_seed42.npy -> ('Fusion', ['1', '2'], '42')
    範例: error_Only_RTT_3_seed123456.npy -> ('Only_RTT', ['3'], '123456')
    """
    # 移除副檔名和前綴
    name = os.path.basename(filename).replace(".npy", "").replace("error_", "")
    
    # 提取 Seed
    if "_seed" in name:
        base_part, seed = name.split("_seed")
    else:
        base_part, seed = name, "unknown"
        
    # 提取架構與 RTT 組合
    # 常見模式: Fusion_RTT_1_2... 或 Only_RTT_1...
    if "Fusion_RTT" in base_part:
        mode = "Fusion"
        rtt_str = base_part.replace("Fusion_RTT_", "")
    elif "Only_RTT" in base_part:
        mode = "Only_RTT"
        rtt_str = base_part.replace("Only_RTT_", "")
    elif "Only_RSSI" in base_part:
        mode = "Only_RSSI"
        rtt_str = "0" # RSSI 固定，視為 0 RTT 或特殊處理
    else:
        # 處理其他可能命名，如 DANN_Fusion...
        mode = "Unknown"
        rtt_str = base_part
    
    if mode == "Only_RSSI":
        rtt_list = []
    else:
        rtt_list = rtt_str.split("_")
        
    return mode, rtt_list, seed

def main():
    if not os.path.exists(CDF_DATA_DIR):
        print(f"Error: Directory '{CDF_DATA_DIR}' not found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 讀取所有檔案並整理數據
    # 結構: data_pool[rtt_count][combo_key][mode] = [array_seed1, array_seed2...]
    # rtt_count: 1, 2, 3, 4
    # combo_key: "1_2" (具體的 AP 組合)
    # mode: "Fusion", "Only_RTT"
    data_pool = {}
    
    files = glob.glob(os.path.join(CDF_DATA_DIR, "*.npy"))
    print(f"Found {len(files)} files.")

    for f in files:
        mode, rtt_list, seed = parse_filename(f)
        
        # 排除 RSSI Only 的檔案 (通常 RSSI Only 我們會單獨畫或當作 baseline 畫在每一張圖)
        # 這裡我們先處理有 RTT 的情況
        if mode == "Only_RSSI":
            continue

        rtt_count = len(rtt_list)
        combo_key = "_".join(rtt_list) # e.g., "1_2"
        
        if rtt_count not in data_pool:
            data_pool[rtt_count] = {}
        if combo_key not in data_pool[rtt_count]:
            data_pool[rtt_count][combo_key] = {}
        if mode not in data_pool[rtt_count][combo_key]:
            data_pool[rtt_count][combo_key][mode] = []
            
        # 讀取數據並加入 list (稍後做 merge)
        try:
            arr = np.load(f)
            data_pool[rtt_count][combo_key][mode].extend(arr)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    # 2. 開始繪圖 (針對每個 RTT Count 畫一張圖)
    sorted_counts = sorted(data_pool.keys())
    
    for count in sorted_counts:
        plt.figure(figsize=(10, 6))
        
        # 取得該數量下所有的組合 (e.g., 1_2, 1_3, 2_3...)
        combos = sorted(data_pool[count].keys())
        
        print(f"Plotting RTT Count: {count} (Combos: {combos})")
        
        for combo in combos:
            modes = data_pool[count][combo] # 該組合下的所有模式 (Fusion, Only_RTT)
            
            for mode, data in modes.items():
                # 取得樣式
                style = STYLE_MAP.get(mode, {'color': 'gray', 'linestyle': '-', 'label': mode})
                
                # 計算 CDF
                all_data = np.array(data) # 合併所有 seed 的數據
                x, y = get_cdf(all_data)
                mean_error = np.mean(all_data)
                
                # 標籤設定: 架構 + 組合 + 平均誤差
                # 為了避免圖例太長，如果是 Fusion/RTT 對比，我們可以強調 AP 組合
                label = f"{mode} [{combo.replace('_', ',')}] (Mean: {mean_error:.2f}m)"
                
                # 微調顏色：為了區分不同 combo，我們可以讓顏色稍微變化，或者使用不同的 marker
                # 這裡簡單起見，我們用預設顏色，但不同組合用深淺區分可能太複雜
                # 策略：Fusion 用實線，RTT 用虛線。顏色由 matplotlib 自動循環，或指定。
                
                # 這裡使用自定義策略：
                # 讓同一個 combo 的 Fusion 和 RTT 使用相同顏色，但線條不同
                # 透過 hash combo string 來決定顏色 (簡單做法)
                # color_idx = hash(combo) % 10
                # color = plt.cm.tab10(color_idx)
                
                plt.plot(x, y, linestyle=style['linestyle'], linewidth=2, label=label)

        plt.xlabel('Distance Error (m)', fontsize=12)
        plt.ylabel('CDF', fontsize=12)
        plt.title(f'CDF Comparison - {count} RTT AP(s)', fontsize=14)
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.xlim(0, 5) # 設定 X 軸範圍 (可根據數據調整)
        plt.ylim(0, 1.05)
        
        output_filename = os.path.join(OUTPUT_DIR, f"cdf_rtt_count_{count}.png")
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300)
        plt.close()
        print(f"Saved: {output_filename}")

    print("All plots generated.")

if __name__ == "__main__":
    main()