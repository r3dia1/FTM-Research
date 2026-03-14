# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# # 1. 設定檔案路徑
# file_path_1 = './all/Server_Wide_20260101_140347.csv'  # 包含 AP1,2,3,4 的大表
# file_path_2 = './Server_Wide_20260101_075453_location_AP1.csv'  # AP3 獨立測試的表

# # 設定圖表輸出的資料夾名稱
# output_dir = "analysis_charts"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # 2. 讀取資料
# try:
#     df_all = pd.read_csv(file_path_1)
#     df_ap3 = pd.read_csv(file_path_2)
#     print(f"成功讀取檔案:\n1. {file_path_1}\n2. {file_path_2}\n")
# except FileNotFoundError as e:
#     print(f"錯誤: 找不到檔案，請確認路徑。\n詳細訊息: {e}")
#     exit()

# # 3. 資料過濾與準備
# target_label = '1-1'

# # 處理 Dataset A (從大表取得 AP1 @ Pos1)
# df_a = df_all[df_all['Label'] == target_label][['RSSI_1', 'Dist_mm_1']].copy()
# df_a['Device'] = 'AP1 (Original)'

# # 處理 Dataset B (從獨立表取得 AP3 @ Pos1)
# df_b = df_ap3[df_ap3['Label'] == target_label][['RSSI_1', 'Dist_mm_1']].copy()
# df_b['Device'] = 'AP3 (Independent)'

# # 合併資料 (使用 ignore_index=True 避免索引衝突)
# df_combined = pd.concat([df_a, df_b], ignore_index=True)

# # 4. 計算統計數據 (平均值、標準差、飄移量)
# stats_a = df_a.describe().loc[['mean', 'std']]
# stats_b = df_b.describe().loc[['mean', 'std']]

# drift_rssi = stats_b.loc['mean', 'RSSI_1'] - stats_a.loc['mean', 'RSSI_1']
# drift_dist = stats_b.loc['mean', 'Dist_mm_1'] - stats_a.loc['mean', 'Dist_mm_1']

# # 計算相關係數
# corr_a = df_a['RSSI_1'].corr(df_a['Dist_mm_1'])
# corr_b = df_b['RSSI_1'].corr(df_b['Dist_mm_1'])

# # 5. 繪圖
# plt.figure(figsize=(18, 5))

# # 圖表 1: RSSI 分佈
# plt.subplot(1, 3, 1)
# sns.kdeplot(data=df_combined, x='RSSI_1', hue='Device', fill=True, common_norm=False)
# plt.title(f'RSSI Distribution (Label {target_label})')

# # 圖表 2: RTT (距離) 分佈
# plt.subplot(1, 3, 2)
# sns.kdeplot(data=df_combined, x='Dist_mm_1', hue='Device', fill=True, common_norm=False)
# plt.title(f'Distance Distribution (Label {target_label})')

# # 圖表 3: 關係圖
# plt.subplot(1, 3, 3)
# sns.regplot(data=df_a, x='RSSI_1', y='Dist_mm_1', label=f'AP1 (r={corr_a:.2f})', scatter_kws={'alpha':0.5})
# sns.regplot(data=df_b, x='RSSI_1', y='Dist_mm_1', label=f'AP3 (r={corr_b:.2f})', scatter_kws={'alpha':0.5})
# plt.title('RSSI vs Distance Relationship')
# plt.legend()

# plt.tight_layout()

# # 儲存圖片
# save_path = os.path.join(output_dir, f"drift_analysis_{target_label}.png")
# plt.savefig(save_path, dpi=300)
# print(f"圖表已儲存至: {save_path}\n")

# # 6. 輸出詳細統計報告 (包含平均值)
# print("=" * 40)
# print(f"詳細數據報告 (Label: {target_label})")
# print("=" * 40)

# print(f"[1] RSSI 分析 (訊號強度):")
# print(f"  - AP1 平均 (Mean): {stats_a.loc['mean', 'RSSI_1']:.2f} dBm (Std: {stats_a.loc['std', 'RSSI_1']:.2f})")
# print(f"  - AP3 平均 (Mean): {stats_b.loc['mean', 'RSSI_1']:.2f} dBm (Std: {stats_b.loc['std', 'RSSI_1']:.2f})")
# print(f"  > 飄移量 (Drift) : {drift_rssi:.2f} dBm")
# print("-" * 40)

# print(f"[2] Distance 分析 (測距):")
# print(f"  - AP1 平均 (Mean): {stats_a.loc['mean', 'Dist_mm_1']:.2f} mm (Std: {stats_a.loc['std', 'Dist_mm_1']:.2f})")
# print(f"  - AP3 平均 (Mean): {stats_b.loc['mean', 'Dist_mm_1']:.2f} mm (Std: {stats_b.loc['std', 'Dist_mm_1']:.2f})")
# print(f"  > 飄移量 (Drift) : {drift_dist:.2f} mm")
# print("-" * 40)

# print(f"[3] 關係分析 (Correlation):")
# print(f"  - AP1 相關係數 r : {corr_a:.4f}")
# print(f"  - AP3 相關係數 r : {corr_b:.4f}")
# print("=" * 40)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. 設定檔案路徑
file_path_1 = '../2026_1_2/Server_Wide_20260102_114230.csv'  # AP1 (從大表讀取)
file_path_2 = './Server_Wide_20260101_075453_location_AP1.csv'  # AP3 (獨立表)

# 設定輸出資料夾
output_dir = "analysis_charts_global"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. 讀取資料
try:
    df_all = pd.read_csv(file_path_1)
    df_ap3 = pd.read_csv(file_path_2)
    print("檔案讀取成功，開始全域分析...\n")
except FileNotFoundError:
    print("找不到檔案，請確認路徑。")
    exit()

# 3. 資料準備 (提取所有 Label 的數據)
# 這裡我們不篩選特定 Label，而是使用全部資料

# 準備 AP1 數據
df_a = df_all[['Label', 'RSSI_1', 'Dist_mm_1']].copy()
df_a['Device'] = 'AP1'

# 準備 AP3 數據
df_b = df_ap3[['Label', 'RSSI_1', 'Dist_mm_1']].copy()
df_b['Device'] = 'AP3'

# 合併數據
df_combined = pd.concat([df_a, df_b], ignore_index=True)

# 移除異常值 (例如距離為 0 或 RSSI 為 0 的無效數據，避免影響圖表)
df_combined = df_combined[(df_combined['Dist_mm_1'] > 0) & (df_combined['RSSI_1'] != 0)]

# 4. 計算全域相關係數 (Global Correlation)
corr_a = df_combined[df_combined['Device']=='AP1'][['RSSI_1', 'Dist_mm_1']].corr().iloc[0, 1]
corr_b = df_combined[df_combined['Device']=='AP3'][['RSSI_1', 'Dist_mm_1']].corr().iloc[0, 1]

# 5. 繪圖 - 呈現物理關係
plt.figure(figsize=(12, 8))

# 使用 Scatter Plot 繪製所有點
# alpha=0.1 讓點變透明，這樣才能看到點密集的地方 (熱區)
sns.scatterplot(data=df_combined, x='RSSI_1', y='Dist_mm_1', hue='Device', alpha=0.2, s=15)

# 加入趨勢線 (Regression Line)
sns.regplot(data=df_combined[df_combined['Device']=='AP1'], x='RSSI_1', y='Dist_mm_1', 
            scatter=False, color='blue', label=f'AP1 Trend (r={corr_a:.2f})')
sns.regplot(data=df_combined[df_combined['Device']=='AP3'], x='RSSI_1', y='Dist_mm_1', 
            scatter=False, color='orange', label=f'AP3 Trend (r={corr_b:.2f})')

plt.title('Global Correlation: RSSI vs RTT (All Labels)', fontsize=15)
plt.xlabel('RSSI (Signal Strength, dBm)', fontsize=12)
plt.ylabel('RTT Measured Distance (mm)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# 儲存圖片
save_path = os.path.join(output_dir, "global_correlation.png")
plt.savefig(save_path, dpi=300)
plt.show()

# 6. 進階分析：以 Label 為單位的平均值關係 (去除雜訊看本質)
# 計算每個 Label 的平均 RSSI 和平均距離
df_grouped = df_combined.groupby(['Device', 'Label'])[['RSSI_1', 'Dist_mm_1']].mean().reset_index()

# 計算 Group 後的相關係數 (這通常會非常高)
corr_group_a = df_grouped[df_grouped['Device']=='AP1'][['RSSI_1', 'Dist_mm_1']].corr().iloc[0, 1]
corr_group_b = df_grouped[df_grouped['Device']=='AP3'][['RSSI_1', 'Dist_mm_1']].corr().iloc[0, 1]

print("="*40)
print("全域數據分析報告 (Global Analysis)")
print("="*40)
print(f"[1] 原始數據相關性 (含雜訊):")
print(f"   - AP1 Correlation: {corr_a:.4f}")
print(f"   - AP3 Correlation: {corr_b:.4f}")
print(f"   (數值越接近 -1，代表物理關係越明顯)")
print("-" * 40)
print(f"[2] 去除雜訊後趨勢 (以 Label 取平均):")
print(f"   - AP1 Correlation: {corr_group_a:.4f}")
print(f"   - AP3 Correlation: {corr_group_b:.4f}")
print("   (這代表：當你真的移動到不同位置時，RSSI 和 RTT 的連動程度)")
print("="*40)