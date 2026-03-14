import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ==========================================
# 1. 實驗參數設定
# ==========================================

experiments_config = {
    'Pos1': {
        'single_file': './Server_Wide_20260101_075453_location_AP1.csv',
        'full_file':   './all/Server_Wide_20260101_140347.csv',
        'full_set_target_bssid': '24:29:34:e2:4c:36' # 在 Full Set 中，這顆 AP 的 BSSID
    },
    'Pos2': {
        'single_file': './Server_Wide_20260101_124356_location_AP2.csv',
        'full_file':   './all/Server_Wide_20260101_140347.csv',
        'full_set_target_bssid': '24:29:34:e1:ef:d4' # 在 Full Set 中，這顆 AP 的 BSSID
    },
    'Pos3': {
        'single_file': './Server_Wide_20260101_065323_location_AP3.csv',   # 單一 AP 的檔案路徑
        'full_file':   './all/Server_Wide_20260101_140347.csv',      # 完整 4 AP 的檔案路徑
        'full_set_target_bssid': 'b0:e4:d5:88:16:86' # 在 Full Set 中，這顆 AP 的 BSSID
    },
    'Pos4': {
        'single_file': './Server_Wide_20260101_113813_location_AP4.csv',
        'full_file':   './all/Server_Wide_20260101_140347.csv',
        'full_set_target_bssid': 'e4:5e:1b:a0:5e:85' # 在 Full Set 中，這顆 AP 的 BSSID
    }
}

REMOVE_NEGATIVE_DIST = True

# ==========================================
# 2. 資料處理
# ==========================================

def clean_and_load(filepath, rename_suffix=False):
    if not os.path.exists(filepath): return None
    df = pd.read_csv(filepath, low_memory=False)
    
    # 處理 _1 後綴
    if rename_suffix:
        rename_map = {c: c[:-2] for c in df.columns if c.endswith('_1')}
        df = df.rename(columns=rename_map)

    # 強制轉數值
    cols = ['RSSI', 'Dist_mm', 'Std_mm']
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            
    # 過濾 RTT 負值
    if REMOVE_NEGATIVE_DIST and 'Dist_mm' in df.columns:
        df = df[df['Dist_mm'] > 0]
        
    return df.dropna(subset=['RSSI', 'Dist_mm', 'Label'])

def extract_full_target(filepath, target_bssid):
    if not os.path.exists(filepath): return None
    df_full = pd.read_csv(filepath, low_memory=False)
    frames = []
    for i in range(1, 5):
        suffix = f'_{i}'
        if f'BSSID{suffix}' not in df_full.columns: continue
        cols = {f'Label': 'Label', f'BSSID{suffix}': 'BSSID', 
                f'RSSI{suffix}': 'RSSI', f'Dist_mm{suffix}': 'Dist_mm', f'Std_mm{suffix}': 'Std_mm'}
        try:
            subset = df_full[list(cols.keys())].rename(columns=cols)
            frames.append(subset)
        except KeyError: continue
        
    if not frames: return pd.DataFrame()
    df_long = pd.concat(frames, ignore_index=True)
    
    # 清洗與過濾
    df_target = df_long[df_long['BSSID'] == target_bssid].copy()
    for c in ['RSSI', 'Dist_mm', 'Std_mm']:
        df_target[c] = pd.to_numeric(df_target[c], errors='coerce')
        
    if REMOVE_NEGATIVE_DIST:
        df_target = df_target[df_target['Dist_mm'] > 0]
        
    return df_target.dropna(subset=['RSSI', 'Dist_mm', 'Label'])

# ==========================================
# 3. 核心分析：RP-wise 差異分析
# ==========================================

def analyze_rp_drift(df_single, df_full, loc_name):
    """
    計算每個 RP 點的平均值差異，並畫出二維漂移圖
    """
    # 1. 依 Label (RP) 分組計算平均
    grp_single = df_single.groupby('Label')[['RSSI', 'Dist_mm']].mean()
    grp_full = df_full.groupby('Label')[['RSSI', 'Dist_mm']].mean()
    
    # 2. 合併資料 (Inner Join 確保只比較兩邊都有的點)
    merged = grp_single.join(grp_full, lsuffix='_S', rsuffix='_F', how='inner')
    
    # 3. 計算差異 (Full - Single)
    merged['Diff_RSSI'] = merged['RSSI_F'] - merged['RSSI_S']
    merged['Diff_Dist'] = merged['Dist_mm_F'] - merged['Dist_mm_S']
    
    # 4. 繪圖：RP 漂移圖 (The Drift Plot)
    plt.figure(figsize=(10, 8))
    
    # 畫散點
    sns.scatterplot(data=merged, x='Diff_RSSI', y='Diff_Dist', s=100, alpha=0.7)
    
    # 加上中心十字線
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    # 標上 RP Label (找出差異最大的前 5 名標示)
    merged['Magnitude'] = np.sqrt(merged['Diff_RSSI']**2 + (merged['Diff_Dist']/1000)**2) # 簡單權重
    top_diff = merged.nlargest(5, 'Magnitude')
    
    for idx, row in top_diff.iterrows():
        plt.text(row['Diff_RSSI'], row['Diff_Dist'], str(idx), fontsize=12, color='red', fontweight='bold')

    plt.title(f'RP-wise Drift Analysis @ {loc_name}\n(Each dot is one Reference Point)', fontsize=14)
    plt.xlabel('RSSI Shift (dBm) [Full - Single]\n<-- S stronger | F stronger -->')
    plt.ylabel('RTT Shift (mm) [Full - Single]\n<-- S closer | F closer -->')
    plt.grid(True, alpha=0.3)
    
    filename = f"{loc_name}_RP_Drift.png"
    plt.savefig(filename, dpi=300)
    print(f"[Saved] {filename}")
    plt.close()
    
    return merged

# ==========================================
# 4. 主執行
# ==========================================

for loc_name, cfg in experiments_config.items():
    print(f"\nAnalyzing {loc_name}...")
    
    # 載入資料
    df_s = clean_and_load(cfg['single_file'], rename_suffix=True)
    df_f = extract_full_target(cfg['full_file'], cfg['full_set_target_bssid'])
    
    if df_s is None or df_f is None or df_s.empty or df_f.empty:
        print("Data missing, skipping.")
        continue
        
    # 執行 RP 分析
    drift_data = analyze_rp_drift(df_s, df_f, loc_name)
    
    # 輸出統計摘要
    print(f"--- Summary for {loc_name} ---")
    print(f"Average RSSI Shift: {drift_data['Diff_RSSI'].mean():.2f} dBm")
    print(f"Average Dist Shift: {drift_data['Diff_Dist'].mean():.2f} mm")
    print("Top 5 Drifting RPs:")
    print(drift_data.nlargest(5, 'Magnitude')[['Diff_RSSI', 'Diff_Dist']])