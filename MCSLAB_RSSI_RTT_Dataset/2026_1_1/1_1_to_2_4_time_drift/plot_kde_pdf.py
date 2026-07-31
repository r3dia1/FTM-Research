import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy.stats import wasserstein_distance, ks_2samp
import matplotlib
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

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

# 設定兩個不同日期的完整檔案路徑
FILE_BASE = '.././all/All_Data_With_RSSI_Diff_withoutNA.csv'
FILE_COMP = '../../2026_2_4/All_Data_With_RSSI_Diff_withoutNA.csv'
source_date = '2026/1/1'
target_date = '2026/2/4'

# 定義要比較的 4 個 AP 及其名稱
TARGET_APS = {
    'AP1': '24:29:34:e2:4c:36',
    'AP2': '24:29:34:e1:ef:d4',
    'AP3': 'b0:e4:d5:88:16:86',
    'AP4': 'e4:5e:1b:a0:5e:85'
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

def extract_all_aps_data(filepath):
    """
    從完整檔案中提取所有 AP 資料並轉為長表格
    把所有 AP 收到的資料 concat 在一起
    仍然保留 BSSID 以供後續 AP-wised 的分析
    """
    if not os.path.exists(filepath):
        print(f"[Error] 找不到檔案: {filepath}")
        return None

    df_raw = pd.read_csv(filepath, low_memory=False)
    processed_frames = []
    
    # 假設檔案中有 1~4 個掃描槽位 (BSSID_1, BSSID_2...)
    for i in range(1, 5):
        suffix = f'_{i}'
        if f'BSSID{suffix}' not in df_raw.columns: continue
            
        cols = {
            f'BSSID{suffix}': 'BSSID', 
            f'RSSI{suffix}': 'RSSI', 
            f'Dist_mm{suffix}': 'Dist_mm', 
            f'Std_mm{suffix}': 'Std_mm'
        }
        
        subset = df_raw[list(cols.keys())].rename(columns=cols)
        if 'Label' in df_raw.columns:
            subset['Label'] = df_raw['Label']
        processed_frames.append(subset)
    
    df_long = pd.concat(processed_frames, ignore_index=True)
    return clean_numeric_columns(df_long)

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

def calculate_statistical_diff(df_s, df_t, loc_name):
    """
    計算統計差異並存成 CSV (已加入 JSD 與 KS Statistic)
    """
    features = ['RSSI', 'Dist_mm', 'Std_mm']
    results = []

    print(f"\n--- Statistical Analysis for {loc_name} ---")

    for feature in features:
        if feature not in df_s.columns or feature not in df_t.columns: continue

        d1 = df_s[feature].dropna()
        d2 = df_t[feature].dropna()

        if REMOVE_NEGATIVE_DIST and feature in ['Dist_mm', 'Std_mm']:
            d1 = d1[d1 > 0]
            d2 = d2[d2 > 0]

        if len(d1) < 2 or len(d2) < 2: continue # 避免資料不足無法計算 KDE

        mean1, mean2 = d1.mean(), d2.mean()
        std1, std2 = d1.std(), d2.std()
        
        # --- 計算 JSD ---
        min_val = min(d1.min(), d2.min())
        max_val = max(d1.max(), d2.max())
        x_grid = np.linspace(min_val, max_val, 1000) # 建立共用網格
        
        try:
            kde1 = gaussian_kde(d1)
            kde2 = gaussian_kde(d2)
            p = kde1(x_grid)
            q = kde2(x_grid)
            # 正規化為機率分佈 (總和為 1)
            p_norm = p / np.sum(p)
            q_norm = q / np.sum(q)
            jsd_val = jensenshannon(p_norm, q_norm)
        except np.linalg.LinAlgError:
            jsd_val = np.nan # 處理極端奇異矩陣的情況
            
        # --- 計算 KS Test ---
        ks_stat, ks_p = ks_2samp(d1, d2)
        
        results.append({
            'Feature': feature,
            'Source_Mean': round(mean1, 2),
            'Target_Mean': round(mean2, 2),
            'Offset (Diff)': round(mean2 - mean1, 2),
            'Source_Std': round(std1, 2),
            'Target_Std': round(std2, 2),
            'Std_Ratio': round(std2 / std1 if std1 != 0 else np.nan, 2),
            'Wasserstein_Dist': round(wasserstein_distance(d1, d2), 2),
            'JSD': round(jsd_val, 4),
            'KS_Statistic': round(ks_stat, 4)
        })

    res_df = pd.DataFrame(results)
    print(res_df)
    res_df.to_csv(f"{loc_name}_stats.csv", index=False)
    return res_df

def plot_kde_comparison(df_single, df_full, loc_name, source_date, target_date, REMOVE_NEGATIVE_DIST=True):
    """
    繪製 KDE 分佈比較圖 (優化版 - 改為上下排列並符合 IEEE 規範)
    """
    features = ['RSSI', 'Dist_mm']
    # 將特徵名稱與單位合併，讓 X 軸標籤在學術圖表上更完整
    x_labels = ['RSSI (dBm)', 'RTT Distance (mm)']
    
    # 改為 2 列 1 行，設定 figsize=(8, 10) 讓上下圖在論文的單一欄位中比例飽滿
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    
    for i, feature in enumerate(features):
        ax = axes[i]
        if feature not in df_single.columns or feature not in df_full.columns: continue

        d1 = df_single[feature]
        d2 = df_full[feature]

        if REMOVE_NEGATIVE_DIST and feature in ['Dist_mm']:
            d1 = d1[d1 > -5000]
            d2 = d2[d2 > -5000]

        # 保留你的輪廓線與 cut=0 設定
        if len(d1) > 0: 
            sns.kdeplot(d1, ax=ax, label=source_date, fill=True, color='blue', alpha=0.3, linewidth=2.5, cut=0)
        if len(d2) > 0: 
            sns.kdeplot(d2, ax=ax, label=target_date, fill=True, color='orange', alpha=0.3, linewidth=2.5, cut=0)
        
        # 設定 X 軸標籤 (套用名稱+單位)
        ax.set_xlabel(x_labels[i], fontsize=14)
        
        # IEEE 慣例：保留 Y 軸物理意義標籤，並放大字體
        ax.set_ylabel('Probability Density', fontsize=14)
        
        # 保留隱藏 Y 軸刻度數字的設定，降低視覺干擾
        ax.set_yticks([]) 

        # 放大 X 軸刻度數字
        ax.tick_params(axis='x', labelsize=12)

        # 保留圖例設定
        ax.legend(fontsize=12, loc='upper right')
        
        # 保留你的虛線網格設定
        ax.grid(True, alpha=0.4, linestyle='--')
        
    plt.tight_layout()
    # 保留 bbox_inches='tight' 與雙格式輸出
    plt.savefig(f"{loc_name}_distribution.png", dpi=500, bbox_inches='tight')
    plt.savefig(f"{loc_name}_distribution.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print(f"[Image Saved] {loc_name}_distribution.png & .pdf")


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

def plot_time_drift_scatter(merged_df, loc_name):
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
    plt.colorbar(scatter, label=f'RSSI Drift (dBm) [2026/1/1 - {target_date}]')
    
    # 標註誤差大的點
    mask = (abs(merged_df['Diff_RTT']) > 1000) | (abs(merged_df['Diff_RSSI']) > 5)
    for idx, row in merged_df[mask].iterrows():
        plt.text(row['Pos_X'], row['Pos_Y']+0.15, row['Label'], fontsize=9, ha='center', color='darkred', fontweight='bold')

    plt.title(f'{loc_name} Time Drift Analysis\nSize = |RTT Diff|, Color = RSSI Diff', fontsize=14)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')
    
    plt.savefig(f"{loc_name}_Time_Drift_Scatter.png", dpi=300)
    plt.close()
    print(f"[Image Saved] {loc_name}_Time_Drift_Scatter.png")

# ==========================================
# 4. 主流程
# ==========================================

all_results = []

# 先讀取兩份完整資料
print("Loading datasets...")
df_base_full = extract_all_aps_data(FILE_BASE)
df_comp_full = extract_all_aps_data(FILE_COMP)

if df_base_full is not None and df_comp_full is not None:
    
    # 針對每一個 AP 進行對比分析
    for ap_name, bssid in TARGET_APS.items():
        print(f"\n{'='*30}\nComparing {ap_name} ({bssid})\n{'='*30}")
        
        # 篩選特定 AP
        df_base = df_base_full[df_base_full['BSSID'] == bssid].copy()
        df_comp = df_comp_full[df_comp_full['BSSID'] == bssid].copy()
        
        if df_base.empty or df_comp.empty:
            print(f"[Skip] {ap_name} data missing in one of the files")
            continue

        # 這裡沿用您原本的分析函式
        # 注意：原本函式裡的 'Single' 代表 Base 日期，'Full' 代表 Comp 日期
        # ======================
        #   論文用到了前兩個分析
        # ======================
        calculate_statistical_diff(df_base, df_comp, ap_name)
        plot_kde_comparison(df_base, df_comp, ap_name, source_date, target_date)
        plot_scatter(df_base, df_comp, ap_name)
        analyze_rp_drift_plot(df_base, df_comp, ap_name)
        
        # 空間漂移分析 (RP-wise)
        if 'Label' in df_base.columns and 'Label' in df_comp.columns:
            grp_b = df_base.groupby('Label')[['RSSI', 'Dist_mm']].mean()
            grp_c = df_comp.groupby('Label')[['RSSI', 'Dist_mm']].mean()
            
            merged = grp_b.join(grp_c, lsuffix='_Base', rsuffix='_Comp', how='inner')
            merged['Diff_RSSI'] = merged['RSSI_Comp'] - merged['RSSI_Base']
            merged['Diff_RTT'] = merged['Dist_mm_Comp'] - merged['Dist_mm_Base']
            merged = merged.reset_index()
            
            merged = map_coordinates(merged)
            merged['AP_ID'] = ap_name
            all_results.append(merged)
            
            # 呼叫您原本的空間散點圖
            # 這裡傳入的 loc_name 會變成 AP1, AP2...
            plot_time_drift_scatter(merged, ap_name)

# 4. 輸出總表
if all_results:
    final_df = pd.concat(all_results, ignore_index=True)
    cols = ['Position_ID', 'Label', 'Pos_X', 'Pos_Y', 'RSSI_Single', 'RSSI_Full', 'Diff_RSSI', 'Dist_mm_Single', 'Dist_mm_Full', 'Diff_RTT']
    final_df = final_df[[c for c in cols if c in final_df.columns]]
    final_df.to_csv('All_Pos_RP_Drift_Stats.csv', index=False, float_format='%.2f')
    print(f"\n[Success] Integrated stats saved to All_Pos_RP_Drift_Stats.csv")