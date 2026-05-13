import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# ==========================================
# 1. 設定與路徑配置
# ==========================================

# 設置 BASE_DIR。若要使用自己的數據，請改回 "." 或設置實際數據路徑。
BASE_DIR = "." # 用戶原本的設定
# OUTPUT_DIR = "cdf_plots_2026_2_4(no zoom in version)"
OUTPUT_DIR = "cdf_plots_2026_3_17(no zoom in version)"

# 需求 1: 優化顏色順序以增加辨識度 (主要將紅色分配給 DuGDA)
COLORS = [
    '#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', 
    '#64B5CD', '#DA8BC3', '#8C8C8C', '#CCB974', '#937860'
]
# DuGDA 專用顏色 (鮮豔紅色)
DUGDA_COLOR = '#E31A1C'

# 需求 3: 擴充的架構與對應設定
# ARCH_CONFIG = {
#     'DNN': {
#         'subpath': 'DNN/results/cdf_data_2_4', 
#         'label': 'DNN', 
#         'color': COLORS[0] # Baseline 保留
#     },
#     'FusionDNN': {
#         'subpath': 'FusionDNN/results/cdf_data_2_4', 
#         'label': 'FusionDNN', 
#         'color': COLORS[1]
#     },
#     'DANN': {
#         'subpath': 'DANN/results/dann_v5/cdf_data_2_4(s:t 1:0.5)', 
#         'label': 'DANN', 
#         'color': COLORS[2]
#     },
#     'FusionDANN': {
#         'subpath': 'FusionDANN/results/fusion_dann_v5/cdf_data_2_4 (s:t 1:0.5)', 
#         'label': 'FusionDANN', 
#         'color': COLORS[3]
#     },
#     'CDAN': {
#         'subpath': 'FusionCDAN/ablation test(s:t=1:0.5) 2-4/CDAN/results/cdf_data',
#         'label': 'CDAN', 
#         'color': COLORS[4]
#     },
#     'FusionCDAN': {
#         'subpath': 'FusionCDAN/ablation test(s:t=1:0.5) 2-4/FusionCDAN/results/cdf_data(64)',
#         'label': 'FusionCDAN', 
#         'color': COLORS[5]
#     },
#     'DAFI': {
#         'subpath': 'DAFI/results/cdf_data_2_4(1:0.5)',
#         'label': 'DAFI', 
#         'color': COLORS[6]
#     },
#     # GACDAN (DuGDA): 修改顏色為鮮豔紅色
#     'GACDAN': {
#         'subpath': 'FusionCDAN/ablation test(s:t=1:0.5) 2-4/GACDAN/results/cdf_data',
#         'label': 'DuGDA', 
#         'color': DUGDA_COLOR
#     }
# }

ARCH_CONFIG = {
    'DNN': {
        'subpath': 'DNN/results/cdf_data_3_17', 
        'label': 'DNN', 
        'color': COLORS[0]
    },
    'FusionDNN': {
        'subpath': 'FusionDNN/results/cdf_data_3_17', 
        'label': 'FusionDNN', 
        'color': COLORS[1]
    },
    'DANN': {
        'subpath': 'DANN/results/dann_v5/cdf_data_3_17', 
        'label': 'DANN', 
        'color': COLORS[2]
    },
    'FusionDANN': {
        'subpath': 'FusionDANN/results/fusion_dann_v5/cdf_data_3_17', 
        'label': 'FusionDANN', 
        'color': COLORS[3]
    },
    'CDAN': {
        'subpath': 'CDAN/results/cdf_data_3_17',  # 請調整路徑
        'label': 'CDAN', 
        'color': COLORS[4]
    },
    'FusionCDAN': {
        'subpath': 'FusionCDAN/ablation test(s:t=1:0.5) 3-17/FusionCDAN/results/cdf_data',
        'label': 'FusionCDAN', 
        'color': COLORS[5]
    },
    'DAFI': {
        'subpath': 'DAFI/results/cdf_data_3_17',  # 請調整路徑
        'label': 'DAFI', 
        'color': COLORS[6]
    },
    'DuGDA': {
        'subpath': 'FusionCDAN/ablation test(s:t=1:0.5) 3-17/DuGDA/results/cdf_data',  # 請調整路徑
        'label': 'DuGDA', 
        'color': DUGDA_COLOR
    }
}

# ==========================================
# 2. 檔名解析邏輯 (保持不變)
# ==========================================

def parse_filename(filename):
    name = os.path.basename(filename).replace(".npy", "")
    if "Only_RTT" in name or "Only_RSSI" in name:
        return None
    name_no_seed = re.sub(r'_seed\d+$', '', name)
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

    # 1. 遍歷所有架構 (數據讀取與原程式碼相同)
    for arch_key, config in ARCH_CONFIG.items():
        search_path = os.path.join(BASE_DIR, config['subpath'], "*.npy")
        files = glob.glob(search_path)
        
        # print(f"  > Checking {arch_key}: Found {len(files)} files in {config['subpath']}")
        
        if len(files) == 0:
            print(f"    [Warning] No files found for {arch_key}. Check path: {search_path}")
            continue

        for f in files:
            rtt_count = parse_filename(f)
            if rtt_count is None:
                continue
            label_key = config['label']
            if rtt_count not in plot_data:
                plot_data[rtt_count] = {}
            if label_key not in plot_data[rtt_count]:
                plot_data[rtt_count][label_key] = {'data': [], 'arch': arch_key}
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
        fig, ax = plt.subplots(figsize=(10, 7)) # 修改 plt.figure 為 subplots，獲取 ax

        # 取得該數量下的所有線條數據
        lines = plot_data[count]
        
        # 為了讓圖例的順序跟 ARCH_CONFIG 的定義順序一致，我們用原定義的 key 來排序
        sorted_labels = sorted(lines.keys(), key=lambda k: list(ARCH_CONFIG.keys()).index(lines[k]['arch']))
        
        print(f"Plotting RTT Count: {count} (Lines: {len(sorted_labels)})")
        
        # [需求改進] 2: 設置局部放大圖 (Inset Plot)
        # 設置一個 inset axes ( loc="lower left" 結合 bbox_to_anchor 定位在右側偏上)
        # axins = inset_axes(ax, width="40%", height="40%", loc="lower left", 
        #                    bbox_to_anchor=(0.58, 0.45, 1, 1), bbox_transform=ax.transAxes)

        for label in sorted_labels:
            info = lines[label]
            all_data = np.array(info['data'])
            
            # 計算 CDF
            x, y = get_cdf(all_data)
            mean_error = np.mean(all_data)
            
            # [需求改進] 1: 設置線條辨識度
            # DuGDA (GACDAN) 用最粗的實線 (紅色)
            # Baseline (DNN) 用虛線
            linewidth = 2
            linestyle = '-'
            if label == 'DuGDA':
                linewidth = 4  # 最粗
                linestyle = '-' # 實線
            elif label == 'DNN':
                linestyle = '--' # 虛線
            
            arch_key = info['arch']
            color = ARCH_CONFIG[arch_key]['color']
            
            # 最終圖例名稱包含平均誤差
            final_label = f"{label} [Mean: {mean_error:.4f}m]"
            
            # 1. 在主圖繪製
            ax.plot(x, y, label=final_label, color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.8)
            
            # 2. 在 Inset 圖繪製 (不要加 label，不加 alpha，讓 inset 更清晰)
            # axins.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, alpha=1.0)

        # [需求改進] 2: 設置 Inset Plot 的範圍和修飾
        # x1, x2, y1, y2 = 0.3, 1.0, 0.3, 0.87 # 定義放大範圍 X=0.4m to 1.0m，Y 根據數據調整
        # axins.set_xlim(x1, x2)
        # axins.set_ylim(y1, y2)
        # axins.grid(True, linestyle='--', alpha=0.5) # Inset 添加網格
        
        # Inset 刻度設置，只顯示 0.4, 0.6, 0.8, 1.0
        # axins.set_xticks([0.4, 0.6, 0.8, 1.0])
        
        # 添加 mark_inset (在主圖標記放大區域，並連線)
        # mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5")
        # mark_inset 會回傳三個物件：放大框、連線1、連線2
        # patch, pp1, pp2 = mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
        # # 將兩條連線隱藏，只保留框框
        # pp1.set_visible(False)
        # pp2.set_visible(False)

        # 主圖修飾 (改為 ax.set_...)
        ax.set_xlabel('Distance Error (m)', fontsize=14)
        ax.set_ylabel('CDF', fontsize=14)
        # ax.set_title(f'Performance Comparison (Fusion Mode) - {count} RTT AP(s)', fontsize=16, fontweight='bold')
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        
        # 設定 X 軸範圍 (可根據需求調整，例如只看 0-5米)
        ax.set_xlim(0, 7)
        ax.set_ylim(0, 1.02)
        
        ax.legend(loc='lower right', fontsize=18)
        plt.tight_layout()
        
        save_path_png = os.path.join(OUTPUT_DIR, f"CDF_Fusion_RTT_{count}_APs.png")
        save_path_pdf = os.path.join(OUTPUT_DIR, f"CDF_Fusion_RTT_{count}_APs.pdf")
        plt.savefig(save_path_png, dpi=1000, bbox_inches='tight')
        plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"  -> Saved: {save_path_png} & {save_path_pdf}")

    print("\nAll plots generated in directory:", OUTPUT_DIR)

if __name__ == "__main__":
    main()