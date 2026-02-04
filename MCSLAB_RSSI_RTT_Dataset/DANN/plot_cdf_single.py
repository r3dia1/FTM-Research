import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def get_cdf(data):
    data_sorted = np.sort(data)
    p = 1. * np.arange(len(data)) / (len(data) - 1)
    return data_sorted, p

def main():
    CDF_DIR = "results_single_stream/cdf_data"
    files = glob.glob(os.path.join(CDF_DIR, "*.npy"))
    
    data_map = {}
    
    for f in files:
        # filename: error_RSSI_1_2_seed42.npy
        parts = os.path.basename(f).replace("error_", "").split("_seed")
        combo_name = parts[0] # e.g., "RSSI_1_2"
        
        errors = np.load(f)
        if combo_name not in data_map:
            data_map[combo_name] = []
        data_map[combo_name].extend(errors)
        
    plt.figure(figsize=(10, 6))
    
    # 排序：按照 RTT AP 數量 (檔名越長通常代表越多 AP，因為是 RSSI_1_2_3_4)
    sorted_combos = sorted(data_map.keys(), key=lambda x: (len(x.split('_')), x))
    
    for combo in sorted_combos:
        all_errors = np.array(data_map[combo])
        x, y = get_cdf(all_errors)
        
        rtt_parts = combo.replace("RSSI_", "").split("_")
        num_rtt = len(rtt_parts)
        label = f"RTT APs: {','.join(rtt_parts)} (Mean: {np.mean(all_errors):.2f}m)"
        
        style = '-'
        if num_rtt == 3: style = '--'
        elif num_rtt == 2: style = '-.'
        elif num_rtt == 1: style = ':'
            
        plt.plot(x, y, label=label, linestyle=style)
        
    plt.xlabel('Distance Error (m)')
    plt.ylabel('CDF')
    plt.title('CDF Comparison (Fixed 4 RSSI + Various RTT)')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results_single_stream/cdf_comparison.png', dpi=300)
    print("Plot saved.")

if __name__ == "__main__":
    main()