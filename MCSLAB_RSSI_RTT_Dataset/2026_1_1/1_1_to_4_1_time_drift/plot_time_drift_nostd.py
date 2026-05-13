import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy.stats import wasserstein_distance, ks_2samp

# ==========================================
# 1. 參數與座標設定
# ==========================================

REMOVE_NEGATIVE_DIST = False  

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

FILE_BASE = '../all/All_Data_With_RSSI_Diff_withoutNA.csv'
FILE_COMP = '../../2026_4_1/All_Data_With_RSSI_Diff_withoutNA.csv'
source_date = '2026/1/1'
target_date = '2026/4/1'

TARGET_APS = {
    'AP1': '24:29:34:e2:4c:36',
    'AP2': '24:29:34:e1:ef:d4',
    'AP3': 'b0:e4:d5:88:16:86',
    'AP4': 'e4:5e:1b:a0:5e:85'
}

# ==========================================
# 2. 資料處理函式
# ==========================================

# 修正：加入 Diff_RSSI，確保它被正確轉型並保留
def clean_numeric_columns(df, base_cols=['RSSI', 'Dist_mm', 'Std_mm']):
    """
    動態抓取所有 Diff_RSSI 欄位並轉為數值型態
    """
    # 自動找出所有包含 Diff_RSSI 的欄位名稱
    diff_cols = [c for c in df.columns if 'Diff_RSSI' in c]
    print(diff_cols)
    all_target_cols = base_cols + diff_cols
    
    for col in all_target_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # 為了避免因為某些 Diff_RSSI 剛好是空值而把整列刪掉，這裡只針對 base_cols 做 dropna
    valid_cols_to_drop = [c for c in base_cols if c in df.columns]
    # print(df.dropna(subset=valid_cols_to_drop))
    return df.dropna(subset=valid_cols_to_drop)

def extract_all_aps_data(filepath):
    """
    從完整檔案中提取所有 AP 資料並轉為長表格，同時保留所有 AP 間的 RSSI 差值
    """
    if not os.path.exists(filepath):
        print(f"[Error] 找不到檔案: {filepath}")
        return None

    df_raw = pd.read_csv(filepath, low_memory=False)
    processed_frames = []
    
    # 找出原始資料中所有的 Diff_RSSI_X_Y 欄位
    diff_cols = [c for c in df_raw.columns if c.startswith('Diff_RSSI')]
    
    for i in range(1, 5):
        suffix = f'_{i}'
        if f'BSSID{suffix}' not in df_raw.columns: continue
            
        cols = {
            f'BSSID{suffix}': 'BSSID', 
            f'RSSI{suffix}': 'RSSI', 
            f'Dist_mm{suffix}': 'Dist_mm', 
            f'Std_mm{suffix}': 'Std_mm'
        }
        
        # 提取基礎欄位
        subset = df_raw[list(cols.keys())].rename(columns=cols)
        
        # 加入 Label
        if 'Label' in df_raw.columns:
            subset['Label'] = df_raw['Label']
            
        # 將所有的 Diff_RSSI_X_Y 欄位一併塞進這個 AP 的 DataFrame 中
        for dc in diff_cols:
            if dc in df_raw.columns:
                subset[dc] = df_raw[dc]
                
        processed_frames.append(subset)
    
    df_long = pd.concat(processed_frames, ignore_index=True)
    return clean_numeric_columns(df_long)

def map_coordinates(df):
    if 'Label' not in df.columns: return df
    
    x_coords, y_coords = [], []
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
def plot_all_diffs_grid(df_single, df_full, loc_name, source_date, target_date):
    """
    在同一張大圖 (2x3 網格) 中繪製所有 6 組 Diff_RSSI 的 KDE 分佈比較
    """
    # 這是你要看的 6 個差值特徵
    diff_features = [
        'Diff_RSSI_1_2', 'Diff_RSSI_1_3', 'Diff_RSSI_1_4', 
        'Diff_RSSI_2_3', 'Diff_RSSI_2_4', 'Diff_RSSI_3_4'
    ]
    
    # 建立 2 列 3 欄的畫布，尺寸設為 20x10 確保有足夠空間
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    # 加上大標題
    fig.suptitle(f'All Pairwise RSSI Differences @ {loc_name}', fontsize=20, fontweight='bold', y=1.02)
    
    # 將 2x3 的 axes 攤平為一維陣列，方便用迴圈依序畫圖
    axes = axes.flatten()
    
    for i, feature in enumerate(diff_features):
        ax = axes[i]
        
        # 檢查欄位是否存在於資料中
        if feature not in df_single.columns or feature not in df_full.columns:
            ax.set_title(f"{feature} (Data Missing)", color='red')
            ax.axis('off')
            continue

        # 抓取資料並排除 NaN
        d1 = df_single[feature].dropna()
        d2 = df_full[feature].dropna()

        # 繪製 KDE
        if len(d1) > 0: 
            sns.kdeplot(d1, ax=ax, label=source_date, fill=True, color='blue', alpha=0.3, linewidth=2.5, cut=0)
        if len(d2) > 0: 
            sns.kdeplot(d2, ax=ax, label=target_date, fill=True, color='orange', alpha=0.3, linewidth=2.5, cut=0)
        
        # 設定每個子圖的標題與樣式
        ax.set_title(feature, fontsize=14)
        ax.set_xlabel('RSSI Difference (dBm)', fontsize=12)
        ax.tick_params(axis='x', labelsize=10)
        
        # 隱藏 Y 軸與 Density 文字讓圖表乾淨
        ax.set_yticks([]) 
        ax.set_ylabel('') 
        
        # 加上網格與圖例
        ax.grid(True, alpha=0.4, linestyle='--')
        if len(d1) > 0 or len(d2) > 0:
            ax.legend(fontsize=10, loc='upper right')
            
    # 自動調整子圖間距
    plt.tight_layout()
    
    # 儲存圖片
    save_path = f"{loc_name}_All_Diff_Distributions.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Image Saved] {save_path}")


def calculate_statistical_diff(df_single, df_full, loc_name):
    # 修正：將 Diff_RSSI 加入統計觀察清單
    features = ['RSSI', 'Dist_mm', 'Std_mm', 'Diff_RSSI']
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

    if results:
        res_df = pd.DataFrame(results)
        print(res_df.to_string(index=False))
        res_df.to_csv(f"{loc_name}_stats.csv", index=False)
        return res_df
    return None

def plot_kde_comparison(df_single, df_full, loc_name, source_date, target_date, REMOVE_NEGATIVE_DIST=True):
    features = ['RSSI', 'Dist_mm']
    units = ['dBm', 'mm'] 
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    for i, feature in enumerate(features):
        ax = axes[i]
        if feature not in df_single.columns or feature not in df_full.columns: 
            # 萬一真的沒有該特徵，隱藏該子圖的邊框避免留白很醜
            ax.axis('off')
            continue

        d1 = df_single[feature].dropna()
        d2 = df_full[feature].dropna()

        if REMOVE_NEGATIVE_DIST and feature in ['Dist_mm']:
            d1 = d1[d1 > 0]
            d2 = d2[d2 > 0]

        if len(d1) > 0: 
            sns.kdeplot(d1, ax=ax, label=source_date, fill=True, color='blue', alpha=0.3, linewidth=2.5, cut=0)
        if len(d2) > 0: 
            sns.kdeplot(d2, ax=ax, label=target_date, fill=True, color='orange', alpha=0.3, linewidth=2.5, cut=0)
        
        ax.set_title(feature, fontsize=16)
        ax.set_xlabel(units[i], fontsize=14)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.set_yticks([]) 
        ax.set_ylabel('') 
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True, alpha=0.4, linestyle='--')
        
    plt.tight_layout()
    plt.savefig(f"{loc_name}_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Image Saved] {loc_name}_distribution.png")

def plot_scatter(df_single, df_full, loc_name):
    plt.figure(figsize=(10, 8))
    
    d1 = df_single[['RSSI', 'Dist_mm']].copy()
    d2 = df_full[['RSSI', 'Dist_mm']].copy()
    
    if REMOVE_NEGATIVE_DIST:
        d1 = d1[d1['Dist_mm'] > 0]
        d2 = d2[d2['Dist_mm'] > 0]
        
    plt.scatter(d2['RSSI'], d2['Dist_mm'], color='orange', label=target_date, alpha=0.3, s=20, marker='x')
    plt.scatter(d1['RSSI'], d1['Dist_mm'], color='blue', label=source_date, alpha=0.3, s=20, marker='o')

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
    if 'Label' not in df_single.columns or 'Label' not in df_full.columns:
        print(f"[Skip] RP Drift Plot skipped (Missing Label)")
        return

    grp_single = df_single.groupby('Label')[['RSSI', 'Dist_mm']].mean()
    grp_full = df_full.groupby('Label')[['RSSI', 'Dist_mm']].mean()
    
    merged = grp_single.join(grp_full, lsuffix='_S', rsuffix='_F', how='inner')
    merged['Diff_RSSI_Shift'] = merged['RSSI_F'] - merged['RSSI_S']
    merged['Diff_Dist_Shift'] = merged['Dist_mm_F'] - merged['Dist_mm_S']
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=merged, x='Diff_RSSI_Shift', y='Diff_Dist_Shift', s=100, alpha=0.7)
    
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    merged['Magnitude'] = np.sqrt(merged['Diff_RSSI_Shift']**2 + (merged['Diff_Dist_Shift']/1000)**2)
    if len(merged) > 0:
        top_diff = merged.nlargest(min(5, len(merged)), 'Magnitude')
        for idx, row in top_diff.iterrows():
            plt.text(row['Diff_RSSI_Shift'], row['Diff_Dist_Shift'], str(idx), fontsize=12, color='red', fontweight='bold')

    plt.title(f'RP-wise Drift Analysis @ {loc_name}', fontsize=14)
    plt.xlabel(f'RSSI Shift (dBm) [{target_date} - {source_date}]')
    plt.ylabel(f'RTT Shift (mm) [{target_date} - {source_date}]')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f"{loc_name}_RP_Drift.png", dpi=300)
    plt.close()
    print(f"[Image Saved] {loc_name}_RP_Drift.png")

def plot_time_drift_scatter(merged_df, loc_name):
    plt.figure(figsize=(10, 6))
    
    x = merged_df['Pos_X']
    y = merged_df['Pos_Y']
    c = merged_df['Shift_RSSI'] # 修正名稱對齊
    s = abs(merged_df['Shift_RTT']) / 5 + 30  
    
    scatter = plt.scatter(x, y, c=c, s=s, cmap='coolwarm', alpha=0.9, edgecolors='k', vmin=-8, vmax=8)
    plt.colorbar(scatter, label=f'RSSI Drift (dBm) [{source_date} -> {target_date}]')
    
    mask = (abs(merged_df['Shift_RTT']) > 1000) | (abs(merged_df['Shift_RSSI']) > 5)
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

print("Loading datasets...")
df_base_full = extract_all_aps_data(FILE_BASE)
print(df_base_full)
df_comp_full = extract_all_aps_data(FILE_COMP)
print(df_comp_full)

if df_base_full is not None and df_comp_full is not None:
    
    for ap_name, bssid in TARGET_APS.items():
        print(f"\n{'='*30}\nComparing {ap_name} ({bssid})\n{'='*30}")
        
        df_base = df_base_full[df_base_full['BSSID'] == bssid].copy()
        df_comp = df_comp_full[df_comp_full['BSSID'] == bssid].copy()
        
        if df_base.empty or df_comp.empty:
            print(f"[Skip] {ap_name} data missing in one of the files")
            continue

        # print(df_base)
        # print(df_comp)

        calculate_statistical_diff(df_base, df_comp, ap_name)
        plot_all_diffs_grid(df_base, df_comp, ap_name, source_date, target_date)
        plot_kde_comparison(df_base, df_comp, ap_name, source_date, target_date)
        plot_scatter(df_base, df_comp, ap_name)
        analyze_rp_drift_plot(df_base, df_comp, ap_name)
        
        # 空間漂移分析 (RP-wise)
        if 'Label' in df_base.columns and 'Label' in df_comp.columns:
            grp_b = df_base.groupby('Label')[['RSSI', 'Dist_mm']].mean()
            grp_c = df_comp.groupby('Label')[['RSSI', 'Dist_mm']].mean()
            
            merged = grp_b.join(grp_c, lsuffix='_Base', rsuffix='_Comp', how='inner')
            
            # 修正：為避免跟原本特徵本身的 Diff_RSSI 搞混，這裡的時間飄移改用 'Shift' 命名
            merged['Shift_RSSI'] = merged['RSSI_Comp'] - merged['RSSI_Base']
            merged['Shift_RTT'] = merged['Dist_mm_Comp'] - merged['Dist_mm_Base']
            merged = merged.reset_index()
            
            merged = map_coordinates(merged)
            merged['AP_ID'] = ap_name
            all_results.append(merged)
            
            plot_time_drift_scatter(merged, ap_name)

# 4. 輸出總表
if all_results:
    final_df = pd.concat(all_results, ignore_index=True)
    # 修正：對齊實際的 DataFrame 欄位名稱 (原本你的寫法因為名稱錯置會抓不到資料)
    cols = ['AP_ID', 'Label', 'Pos_X', 'Pos_Y', 'RSSI_Base', 'RSSI_Comp', 'Shift_RSSI', 'Dist_mm_Base', 'Dist_mm_Comp', 'Shift_RTT']
    final_df = final_df[[c for c in cols if c in final_df.columns]]
    final_df.to_csv('All_Pos_RP_Drift_Stats.csv', index=False, float_format='%.2f')
    print(f"\n[Success] Integrated stats saved to All_Pos_RP_Drift_Stats.csv")