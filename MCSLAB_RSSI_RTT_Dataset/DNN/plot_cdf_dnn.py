import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def get_cdf(data):
    data_sorted = np.sort(data)
    p = 1. * np.arange(len(data)) / (len(data) - 1)
    return data_sorted, p

def main():
    CDF_DIR = "results_dnn/cdf_data"
    files = glob.glob(os.path.join(CDF_DIR, "*.npy"))
    
    data_map = {}
    
    for f in files:
        # filename e.g.: error_DNN_Fusion_RTT_1_2_seed42.npy
        parts = os.path.basename(f).replace("error_", "").split("_seed")
        combo_name = parts[0]
        
        errors = np.load(f)
        if combo_name not in data_map:
            data_map[combo_name] = []
        data_map[combo_name].extend(errors)
        
    plt.figure(figsize=(10, 6))
    
    # 排序：優先顯示 Fusion, 再顯示 RSSI, 再顯示 RTT
    sorted_combos = sorted(data_map.keys())
    
    for combo in sorted_combos:
        all_errors = np.array(data_map[combo])
        x, y = get_cdf(all_errors)
        
        mean_err = np.mean(all_errors)
        label = f"{combo.replace('DNN_', '')} (Mean: {mean_err:.2f}m)"
        
        # 簡單樣式區分
        style = '-'
        if "RSSI" in combo and "Fusion" not in combo: style = '--' # 純 RSSI
        elif "Only_RTT" in combo: style = ':' # 純 RTT
            
        plt.plot(x, y, label=label, linestyle=style)
        
    plt.xlabel('Distance Error (m)')
    plt.ylabel('CDF')
    plt.title('CDF Comparison (DNN Baseline)')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results_dnn/cdf_comparison_dnn.png', dpi=300)
    print("CDF plot saved to results_dnn/cdf_comparison_dnn.png")

if __name__ == "__main__":
    main()