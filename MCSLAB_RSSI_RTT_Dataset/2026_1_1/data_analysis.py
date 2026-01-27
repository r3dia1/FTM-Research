import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy.stats import wasserstein_distance, ks_2samp

# ==========================================
# 1. 參數與座標設定
# ==========================================

# 設為 False，保留負值以供分析 (觀察測距失敗或 Error Code)
REMOVE_NEGATIVE_DIST = False  

# 您提供的精確座標映射
LABEL_TO_COORDS = {
    "1-1": (0, 0), "1-2": (-0.6, 0), "1-3": (-1.2, 0), "1-4": (-1.8, 0), "1-5": (-2.4, 0), "1-6": (-3.0, 0),"1-7": (-3.6, 0), "1-8": (-4.2, 0), "1-9": (-4.8, 0), "1-10": (-5.4, 0), "1-11": (-6.0, 0),
    "2-1": (0, 0.6), "2-11": (-6.0, 0.6),
    "3-1": (0, 1.2), "3-11": (-6.0, 1.2),
    "4-1": (0, 1.8), "4-11": (-6.0, 1.8),
    "5-1": (0, 2.4), "5-11": (-6.0, 2.4),
    "6-1": (0, 3.0), "6-2": (-0.6, 3.0), "6-3": (-1.2, 3.0), "6-4": (-1.8, 3.0), "6-5": (-2.4, 3.0),"6-6": (-3.0, 3.0), "6-7": (-3.6, 3.0), "6-8": (-4.2, 3.0), "6-9": (-4.8, 3.0), "6-10": (-5.4, 3.0), "6-11": (-6.0, 3.0),
    "7-1": (0, 3.6), "7-11": (-6.0, 3.6),
    "8-1": (0, 4.2), "8-11": (-6.0, 4.2),
    "9-1": (0, 4.8), "9-11": (-6.0, 4.8),
    "10-1": (0, 5.4), "10-11": (-6.0, 5.4),
    "11-1": (0, 6.0), "11-2": (-0.6, 6.0), "11-3": (-1.2, 6.0), "11-4": (-1.8, 6.0), "11-5": (-2.4, 6.0),"11-6": (-3.0, 6.0), "11-7": (-3.6, 6.0), "11-8": (-4.2, 6.0), "11-9": (-4.8, 6.0), "11-10": (-5.4, 6.0), "11-11": (-6.0, 6.0)
}

# 實驗設定
experiments_config = {
    'Pos1': {
        'single_file': './Server_Wide_20260101_075453_location_AP1.csv',
        'full_file':   './all/Server_Wide_20260101_140347.csv',
        'full_set_target_bssid': '24:29:34:e2:4c:36'
    },
    'Pos2': {
        'single_file': './Server_Wide_20260101_124356_location_AP2.csv',
        'full_file':   './all/Server_Wide_20260101_140347.csv',
        'full_set_target_bssid': '24:29:34:e1:ef:d4'
    },
    'Pos3': {
        'single_file': './Server_Wide_20260101_065323_location_AP3.csv',
        'full_file':   './all/Server_Wide_20260101_140347.csv',
        'full_set_target_bssid': 'b0:e4:d5:88:16:86'
    },
    'Pos4': {
        'single_file': './Server_Wide_20260101_113813_location_AP4.csv',
        'full_file':   './all/Server_Wide_20260101_140347.csv',
        'full_set_target_bssid': 'e4:5e:1b:a0:5e:85'
    }
}

# ==========================================
# 2. 資料處理函式
# ==========================================

def clean_numeric_columns(df, cols=['RSSI', 'Dist_mm', 'Std_mm']):
    """
    1. 強制將指定欄位轉為數字，無法轉型的變成 NaN
    2. 移除 NaN，但保留負值
    """
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    target_cols = [c for c in cols if c in df.columns]
    return df.dropna(subset=target_cols)

def preprocess_single_dataset(filepath):
    if not os.path.exists(filepath):
        print(f"[Error] 找不到檔案: {filepath}")
        return None

    df = pd.read_csv(filepath, low_memory=False)
    rename_map = {c: c[:-2] for c in df.columns if c.endswith('_1')}
    if rename_map: df = df.rename(columns=rename_map)
    return clean_numeric_columns(df)

def extract_ap_from_full_dataset(filepath, target_bssid):
    """
    修正版：確保 Label 欄位被正確保留
    """
    if not os.path.exists(filepath):
        print(f"[Error] 找不到檔案: {filepath}")
        return None

    df_full = pd.read_csv(filepath, low_memory=False)
    processed_frames = []
    has_label = 'Label' in df_full.columns
    
    for i in range(1, 5):
        suffix = f'_{i}'
        if f'BSSID{suffix}' not in df_full.columns: continue
            
        cols = {f'BSSID{suffix}': 'BSSID', f'RSSI{suffix}': 'RSSI', f'Dist_mm{suffix}': 'Dist_mm', f'Std_mm{suffix}': 'Std_mm'}
        try:
            subset = df_full[list(cols.keys())].rename(columns=cols)
            if has_label: subset['Label'] = df_full['Label']
            processed_frames.append(subset)
        except KeyError: continue
    
    if not processed_frames:
        print("[Error] 無法從 Full Set 提取任何資料")
        return pd.DataFrame()

    df_long = pd.concat(processed_frames, ignore_index=True)
    df_target = df_long[df_long['BSSID'] == target_bssid].copy()
    return clean_numeric_columns(df_target)

def map_coordinates(df):
    """
    將 DataFrame 中的 Label 映射到 (X, Y) 座標
    """
    if 'Label' not in df.columns: return df
    
    x_coords = []
    y_coords = []
    
    for label in df['Label']:
        if label in LABEL_TO_COORDS:
            x, y = LABEL_TO_COORDS[label]
            x_coords.append(x)
            y_coords.append(y)
        else:
            x_coords.append(np.nan)
            y_coords.append(np.nan)
            
    df['Pos_X'] = x_coords
    df['Pos_Y'] = y_coords
    return df.dropna(subset=['Pos_X', 'Pos_Y'])

# ==========================================
# 3. 分析與繪圖函式
# ==========================================

def calculate_statistical_diff(df_single, df_full, loc_name):
    """
    計算統計差異並存成 CSV
    """
    features = ['RSSI', 'Dist_mm', 'Std_mm']
    results = []

    print(f"\n--- Statistical Analysis for {loc_name} ---")

    for feature in features:
        if feature not in df_single.columns or feature not in df_full.columns: continue

        d1 = df_single[feature]
        d2 = df_full[feature]

        if REMOVE_NEGATIVE_DIST and feature in ['Dist_mm', 'Std_mm']:
            d1 = d1[d1 > 0]
            d2 = d2[d2 > 0]

        if len(d1) == 0 or len(d2) == 0: continue

        mean1, mean2 = d1.mean(), d2.mean()
        std1, std2 = d1.std(), d2.std()
        
        results.append({
            'Feature': feature,
            'Single_Mean': round(mean1, 2),
            'Full_Mean': round(mean2, 2),
            'Offset (Diff)': round(mean2 - mean1, 2),
            'Single_Std': round(std1, 2),
            'Full_Std': round(std2, 2),
            'Std_Ratio': round(std2 / std1 if std1 != 0 else np.nan, 2),
            'Wasserstein_Dist': round(wasserstein_distance(d1, d2), 2),
            'KS_P_Value': f"{ks_2samp(d1, d2).pvalue:.2e}"
        })

    res_df = pd.DataFrame(results)
    print(res_df)
    res_df.to_csv(f"{loc_name}_stats.csv", index=False)
    return res_df

def plot_kde_comparison(df_single, df_full, loc_name):
    """
    繪製 KDE 分佈比較圖
    """
    features = ['RSSI', 'Dist_mm', 'Std_mm']
    units = ['dBm', 'mm', 'mm']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Distribution Comparison @ {loc_name}', fontsize=16)
    
    for i, feature in enumerate(features):
        ax = axes[i]
        if feature not in df_single.columns or feature not in df_full.columns: continue

        d1 = df_single[feature]
        d2 = df_full[feature]

        if REMOVE_NEGATIVE_DIST and feature in ['Dist_mm', 'Std_mm']:
            d1 = d1[d1 > 0]
            d2 = d2[d2 > 0]

        if len(d1) > 0: sns.kdeplot(d1, ax=ax, label='Single', fill=True, color='blue', alpha=0.3)
        if len(d2) > 0: sns.kdeplot(d2, ax=ax, label='Full', fill=True, color='orange', alpha=0.3)
        
        ax.set_title(feature)
        ax.set_xlabel(units[i])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(f"{loc_name}_distribution.png", dpi=300)
    plt.close()
    print(f"[Image Saved] {loc_name}_distribution.png")

def plot_scatter(df_single, df_full, loc_name):
    """
    繪製 RSSI vs RTT 散點圖 (包含負值區域)
    """
    plt.figure(figsize=(10, 8))
    
    d1 = df_single[['RSSI', 'Dist_mm']].copy()
    d2 = df_full[['RSSI', 'Dist_mm']].copy()
    
    if REMOVE_NEGATIVE_DIST:
        d1 = d1[d1['Dist_mm'] > 0]
        d2 = d2[d2['Dist_mm'] > 0]
        
    plt.scatter(d2['RSSI'], d2['Dist_mm'], color='orange', label='Full Set', alpha=0.3, s=20, marker='x')
    plt.scatter(d1['RSSI'], d1['Dist_mm'], color='blue', label='Single Set', alpha=0.3, s=20, marker='o')

    plt.axhline(0, color='red', linestyle='--', alpha=0.5, label='Zero Distance')

    plt.title(f'RSSI vs RTT Scatter @ {loc_name}')
    plt.xlabel('RSSI (dBm)')
    plt.ylabel('RTT Distance (mm)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f"{loc_name}_scatter.png", dpi=300)
    plt.close()
    print(f"[Image Saved] {loc_name}_scatter.png")

def analyze_rp_drift_plot(df_single, df_full, loc_name):
    """
    RP-wise 漂移分析 (散點圖)
    """
    if 'Label' not in df_single.columns or 'Label' not in df_full.columns:
        print(f"[Skip] RP Drift Plot skipped (Missing Label)")
        return

    grp_single = df_single.groupby('Label')[['RSSI', 'Dist_mm']].mean()
    grp_full = df_full.groupby('Label')[['RSSI', 'Dist_mm']].mean()
    
    merged = grp_single.join(grp_full, lsuffix='_S', rsuffix='_F', how='inner')
    merged['Diff_RSSI'] = merged['RSSI_F'] - merged['RSSI_S']
    merged['Diff_Dist'] = merged['Dist_mm_F'] - merged['Dist_mm_S']
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=merged, x='Diff_RSSI', y='Diff_Dist', s=100, alpha=0.7)
    
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    merged['Magnitude'] = np.sqrt(merged['Diff_RSSI']**2 + (merged['Diff_Dist']/1000)**2)
    if len(merged) > 0:
        top_diff = merged.nlargest(min(5, len(merged)), 'Magnitude')
        for idx, row in top_diff.iterrows():
            plt.text(row['Diff_RSSI'], row['Diff_Dist'], str(idx), fontsize=12, color='red', fontweight='bold')

    plt.title(f'RP-wise Drift Analysis @ {loc_name}', fontsize=14)
    plt.xlabel('RSSI Shift (dBm) [Full - Single]')
    plt.ylabel('RTT Shift (mm) [Full - Single]')
    # plt.xlabel('RSSI Shift (dBm) [AP1 - AP3]')
    # plt.ylabel('RTT Shift (mm) [AP1 - AP3]')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f"{loc_name}_RP_Drift.png", dpi=300)
    plt.close()
    print(f"[Image Saved] {loc_name}_RP_Drift.png")

def plot_spatial_drift_scatter(merged_df, loc_name):
    """
    [新增] 繪製空間散點圖：在地圖上顯示漂移
    """
    plt.figure(figsize=(10, 6))
    
    x = merged_df['Pos_X']
    y = merged_df['Pos_Y']
    c = merged_df['Diff_RSSI']
    s = abs(merged_df['Diff_RTT']) / 5 + 30  # 大小縮放
    
    scatter = plt.scatter(x, y, c=c, s=s, cmap='coolwarm', alpha=0.9, edgecolors='k', vmin=-8, vmax=8)
    # plt.colorbar(scatter, label='RSSI Drift (dBm) [Full - Single]')
    plt.colorbar(scatter, label='RSSI Drift (dBm) [AP1 - AP3]')
    
    # 標註誤差大的點
    mask = (abs(merged_df['Diff_RTT']) > 1000) | (abs(merged_df['Diff_RSSI']) > 5)
    for idx, row in merged_df[mask].iterrows():
        plt.text(row['Pos_X'], row['Pos_Y']+0.15, row['Label'], fontsize=9, ha='center', color='darkred', fontweight='bold')

    plt.title(f'{loc_name} Spatial Drift Analysis\nSize = |RTT Diff|, Color = RSSI Diff', fontsize=14)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')
    
    plt.savefig(f"{loc_name}_Spatial_Drift_Scatter.png", dpi=300)
    plt.close()
    print(f"[Image Saved] {loc_name}_Spatial_Drift_Scatter.png")

# ==========================================
# 4. 主流程
# ==========================================

all_results = []

for loc_name, cfg in experiments_config.items():
    print(f"\n{'='*30}\nProcessing {loc_name}...\n{'='*30}")
    
    # 1. 讀取與前處理
    df_single = preprocess_single_dataset(cfg['single_file'])
    print(f"Single Set: {len(df_single) if df_single is not None else 0}")
    
    df_full = extract_ap_from_full_dataset(cfg['full_file'], cfg['full_set_target_bssid'])
    print(f"Full Set:   {len(df_full) if df_full is not None else 0}")
    
    if df_single is None or df_full is None or df_single.empty or df_full.empty:
        print("[Skip] Data missing")
        continue

    # 2. 基礎統計與繪圖
    calculate_statistical_diff(df_single, df_full, loc_name)
    plot_kde_comparison(df_single, df_full, loc_name)
    plot_scatter(df_single, df_full, loc_name)
    analyze_rp_drift_plot(df_single, df_full, loc_name)
    
    # 3. 空間分析 (計算 RP 平均與座標映射)
    if 'Label' in df_single.columns and 'Label' in df_full.columns:
        grp_s = df_single.groupby('Label')[['RSSI', 'Dist_mm']].mean()
        grp_f = df_full.groupby('Label')[['RSSI', 'Dist_mm']].mean()
        
        merged = grp_s.join(grp_f, lsuffix='_Single', rsuffix='_Full', how='inner')
        merged['Diff_RSSI'] = merged['RSSI_Full'] - merged['RSSI_Single']
        merged['Diff_RTT'] = merged['Dist_mm_Full'] - merged['Dist_mm_Single']
        merged = merged.reset_index()
        
        # 映射座標
        merged = map_coordinates(merged)
        merged['Position_ID'] = loc_name
        all_results.append(merged)
        
        # 繪製空間圖
        plot_spatial_drift_scatter(merged, loc_name)
    else:
        print("[Skip] Spatial Analysis skipped (Missing Label)")

# 4. 輸出總表
if all_results:
    final_df = pd.concat(all_results, ignore_index=True)
    cols = ['Position_ID', 'Label', 'Pos_X', 'Pos_Y', 'RSSI_Single', 'RSSI_Full', 'Diff_RSSI', 'Dist_mm_Single', 'Dist_mm_Full', 'Diff_RTT']
    final_df = final_df[[c for c in cols if c in final_df.columns]]
    final_df.to_csv('All_Pos_RP_Drift_Stats.csv', index=False, float_format='%.2f')
    print(f"\n[Success] Integrated stats saved to All_Pos_RP_Drift_Stats.csv")