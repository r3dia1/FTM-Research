import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def get_cdf(data):
    """計算 CDF 的 X, Y 值"""
    data_sorted = np.sort(data)
    p = 1. * np.arange(len(data)) / (len(data) - 1)
    return data_sorted, p

def main():
    CDF_DIR = "results/cdf_data"
    
    # 搜尋所有 npy 檔
    files = glob.glob(os.path.join(CDF_DIR, "*.npy"))
    
    # 用 Dictionary 分組： Key=ComboName, Value=List of error arrays (from diff seeds)
    data_map = {}
    
    for f in files:
        filename = os.path.basename(f)
        # filename format: error_{combo}_seed{seed}.npy
        # e.g., error_1_2_seed42.npy
        parts = filename.replace("error_", "").split("_seed")
        combo_name = parts[0] # e.g., "1_2"
        
        errors = np.load(f)
        
        if combo_name not in data_map:
            data_map[combo_name] = []
        data_map[combo_name].extend(errors) # 將所有 seed 的誤差合併成一個大陣列
        
    # 開始畫圖
    plt.figure(figsize=(10, 6))
    
    # 為了圖例整潔，可以排序 combo_name (例如按長度排序: 1 AP, 2 AP...)
    sorted_combos = sorted(data_map.keys(), key=lambda x: (len(x.split('_')), x))
    
    for combo in sorted_combos:
        all_errors = np.array(data_map[combo])
        x, y = get_cdf(all_errors)
        
        # 簡單標示 AP 數量
        num_aps = len(combo.split('_'))
        label = f"{combo.replace('_', ',')} ({num_aps} APs)"
        
        # 根據 AP 數量給顏色 (可選)
        linestyle = '-'
        if num_aps == 1: linestyle = ':'
        elif num_aps == 2: linestyle = '--'
        elif num_aps == 3: linestyle = '-.'
            
        plt.plot(x, y, label=label, linestyle=linestyle, linewidth=1.5)
        
    plt.xlabel('Distance Error (m)')
    plt.ylabel('CDF')
    plt.title('CDF of Positioning Error with Different RTT Combinations')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # 圖例放外側
    plt.tight_layout()
    plt.savefig('results/cdf_comparison.png', dpi=300)
    print("CDF plot saved to results/cdf_comparison.png")

if __name__ == "__main__":
    main()