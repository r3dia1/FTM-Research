import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. 設定檔案路徑與日期標籤
# ==========================================
# 基準日檔案 (較早的時間)
file_path_old = './all/All_Data_With_RSSI_Diff_withoutNA.csv'
date_old = "2026-01-01"

# 比較日檔案 (較晚的時間)
file_path_new = '../2026_2_4/All_Data_With_RSSI_Diff_withoutNA.csv'
date_new = "2026-02-04"

# 設定輸出資料夾
output_dir = "drift_analysis_by_date"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ==========================================
# 2. 讀取與清理資料
# ==========================================
try:
    df_old_raw = pd.read_csv(file_path_old)
    df_new_raw = pd.read_csv(file_path_new)
    print(f"成功讀取檔案：\n- 基準日: {date_old}\n- 比較日: {date_new}\n")
except FileNotFoundError:
    print("找不到檔案，請檢查路徑。")
    exit()

def prepare_data(df, label_name):
    """ 提取必要欄位並標記日期 """
    # 這裡假設我們要比對的是兩份檔案中的第 1 個 AP 欄位 (RSSI_1, Dist_mm_1)
    subset = df[['Label', 'RSSI_1', 'Dist_mm_1']].copy()
    subset['Date'] = label_name
    # 轉為數字並移除無效值 (過濾 RTT=0 或 RSSI=0)
    subset['RSSI_1'] = pd.to_numeric(subset['RSSI_1'], errors='coerce')
    subset['Dist_mm_1'] = pd.to_numeric(subset['Dist_mm_1'], errors='coerce')
    return subset.dropna(subset=['RSSI_1', 'Dist_mm_1'])

df_old = prepare_data(df_old_raw, date_old)
df_new = prepare_data(df_new_raw, date_new)

# 合併數據以便繪圖
df_combined = pd.concat([df_old, df_new], ignore_index=True)
df_combined = df_combined[df_combined['Dist_mm_1'] > 0] # 移除測距異常點

# ==========================================
# 3. 計算統計量 (相關係數)
# ==========================================
corr_old = df_old[df_old['Dist_mm_1'] > 0][['RSSI_1', 'Dist_mm_1']].corr().iloc[0, 1]
corr_new = df_new[df_new['Dist_mm_1'] > 0][['RSSI_1', 'Dist_mm_1']].corr().iloc[0, 1]

# ==========================================
# 4. 繪製全域相關性漂移圖
# ==========================================
plt.figure(figsize=(12, 8))

# 繪製散點 (Alpha 設低一點可以看到重疊密度)
sns.scatterplot(data=df_combined, x='RSSI_1', y='Dist_mm_1', hue='Date', 
                palette={'2026-01-01': 'steelblue', '2026-02-04': 'darkorange'},
                alpha=0.2, s=20)

# 加入趨勢線
sns.regplot(data=df_old, x='RSSI_1', y='Dist_mm_1', scatter=False, 
            color='steelblue', label=f'Trend {date_old} (r={corr_old:.2f})')
sns.regplot(data=df_new, x='RSSI_1', y='Dist_mm_1', scatter=False, 
            color='darkorange', label=f'Trend {date_new} (r={corr_new:.2f})')

plt.title(f'Global Time Drift: RSSI vs RTT ({date_old} vs {date_new})', fontsize=15)
plt.xlabel('RSSI (Signal Strength, dBm)', fontsize=12)
plt.ylabel('RTT Measured Distance (mm)', fontsize=12)
plt.legend(title="Testing Date")
plt.grid(True, linestyle='--', alpha=0.5)

# 儲存圖片
save_path = os.path.join(output_dir, "date_correlation_drift.png")
plt.savefig(save_path, dpi=300)
print(f"漂移分析圖表已儲存至: {save_path}")
plt.show()

# ==========================================
# 5. 輸出數值報告
# ==========================================
# 以 Label 為單位計算平均漂移
grp_old = df_old.groupby('Label')[['RSSI_1', 'Dist_mm_1']].mean()
grp_new = df_new.groupby('Label')[['RSSI_1', 'Dist_mm_1']].mean()
merged_stats = grp_old.join(grp_new, lsuffix='_old', rsuffix='_new', how='inner')

avg_rssi_drift = (merged_stats['RSSI_1_new'] - merged_stats['RSSI_1_old']).mean()
avg_dist_drift = (merged_stats['Dist_mm_1_new'] - merged_stats['Dist_mm_1_old']).mean()

print("\n" + "="*45)
print(f"數據漂移報告: {date_old} >>> {date_new}")
print("="*45)
print(f"[1] 物理相關性變化 (Correlation):")
print(f"   - {date_old}: {corr_old:.4f}")
print(f"   - {date_new}: {corr_new:.4f}")
print("-" * 45)
print(f"[2] 平均環境偏移 (Global Mean Shift):")
print(f"   - RSSI 平均漂移量: {avg_rssi_drift:+.2f} dBm")
print(f"   - RTT  平均漂移量: {avg_dist_drift:+.2f} mm")
print("   (正值代表新日期數值較大，負值代表縮小)")
print("="*45)