import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def get_cdf(data):
    data_sorted = np.sort(data)
    p = 1. * np.arange(len(data)) / (len(data) - 1)
    return data_sorted, p

def main():
    CDF_DIR = "results_dual/cdf_data"
    files = glob.glob(os.path.join(CDF_DIR, "*.npy"))
    
    data_map = {}
    
    for f in files:
        # filename e.g.: error_Dual_FixedRSSI_RTT_1_2_seed42.npy
        parts = os.path.basename(f).replace("error_", "").split("_seed")
        combo_name = parts[0]
        
        errors = np.load(f)
        if combo_name not in data_map:
            data_map[combo_name] = []
        data_map[combo_name].extend(errors)
        
    plt.figure(figsize=(12, 8))
    
    # 排序：按照 RTT AP 數量
    sorted_combos = sorted(data_map.keys(), key=lambda x: (len(x.split('_')), x))
    
    for combo in sorted_combos:
        all_errors = np.array(data_map[combo])
        x, y = get_cdf(all_errors)
        
        rtt_part = combo.split("RTT_")[1]
        num_rtt = len(rtt_part.split('_'))
        
        label = f"RTT: {rtt_part.replace('_', ',')} (Mean: {np.mean(all_errors):.2f}m)"
        
        # 線條樣式區分
        style = '-'
        if num_rtt == 3: style = '--'
        elif num_rtt == 2: style = '-.'
        elif num_rtt == 1: style = ':'
            
        plt.plot(x, y, label=label, linestyle=style, linewidth=1.5)
        
    plt.xlabel('Distance Error (m)')
    plt.ylabel('CDF')
    plt.title('CDF Comparison: Fixed RSSI (6 Diff) + Various RTT Inputs')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results_dual/cdf_comparison_dual.png', dpi=300)
    print("Plot saved to results_dual/cdf_comparison_dual.png")

if __name__ == "__main__":
    main()