import pandas as pd
import matplotlib.pyplot as plt

# 1. 動態讀取 CSV 檔案
# 請將 'your_data.csv' 替換為你的實際 CSV 檔案路徑 (例如: 'results.csv' 或 'C:/data/results.csv')
csv_file_path = 'robustness_results.csv' 
df = pd.read_csv(csv_file_path)

COLORS = [
    '#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3',
    '#64B5CD', '#DA8BC3', '#8C8C8C', '#CCB974', '#937860'
]

# 2. 定義不同架構的樣式 (顏色與節點形狀)
# 你可以在這裡新增對應你 CSV 中 'Combo' 欄位的設定
# style_config = {
#     'CDAN': {'color': '#8172B3', 'marker': 'o', 'linestyle': '-'},  # 藍色, 圓形
#     'FusionCDAN': {'color': '#64B5CD', 'marker': 's', 'linestyle': '-'},    # 橘色, 方塊
#     'GeoSPA-Net': {'color': '#E31A1C', 'marker': '^', 'linestyle': '-'}
# }
style_config = {
    'DNN':        {'color': COLORS[0],        'marker': 'v', 'linestyle': '-'},  # 向下三角形
    'FusionDNN':  {'color': COLORS[1],        'marker': '^', 'linestyle': '-'},  # 向上三角形
    'DANN':       {'color': COLORS[2],        'marker': '<', 'linestyle': '-'},  # 向左三角形
    'FusionDANN': {'color': COLORS[3],        'marker': '>', 'linestyle': '-'},  # 向右三角形
    'CDAN':       {'color': COLORS[4],        'marker': 'o', 'linestyle': '-'},  # 圓形 (原本設定)
    'FusionCDAN': {'color': COLORS[5],        'marker': 's', 'linestyle': '-'},  # 方塊 (原本設定)
    'DAFI':       {'color': COLORS[6],        'marker': 'p', 'linestyle': '-'},  # 五角形
    
    # GeoSPA-Net 使用專屬設定顏色，標記改為菱形
    'GeoSPA-Net': {'color': '#E31A1C', 'marker': 'D', 'linestyle': '-'}   
}

# 3. 定義 X 軸階段
# 這裡假設每個架構都有 4 個測試時間點
x_labels = ['1/1', '2/4', '2/11', '3/5', '3/17']
x_positions = range(len(x_labels))

# 開始繪圖
plt.figure(figsize=(10, 6))

for combo_name, group_data in df.groupby('Combo', sort=False):
    
    # 取得樣式，若未來新增架構但未設定，則使用預設值
    style = style_config.get(combo_name, {'color': 'gray', 'marker': 'x', 'linestyle': '-'})
    
    # 提取 Source 與 Target 的 MDE 數值 (改成抓 MDE 欄位)
    src_mde = group_data['Avg_Src_MDE'].iloc[0]
    tgt_mde = group_data['Avg_Tgt_MDE'].iloc[0]
    
    # 提取所有的 Future (Test) MDE 數值
    fut_mdes = group_data['Avg_Fut_MDE'].tolist()
    
    # 將 Y 軸數值串接
    y_values = [src_mde, tgt_mde] + fut_mdes
    
    # 動態產生這條線專屬的 X 軸座標
    current_x_positions = range(len(y_values))
    
    # 繪製折線圖
    plt.plot(current_x_positions, y_values, 
             label=combo_name, 
             color=style['color'], 
             marker=style['marker'], 
             linestyle=style['linestyle'],
             linewidth=2.5, markersize=8)

# 5. 圖表外觀設定
# plt.title('Architecture MDE across Domains and Time', fontsize=16, fontweight='bold')
plt.xlabel('Domain & Temporal Progression', fontsize=14)
plt.ylabel('Mean Domain Error (MDE)', fontsize=14) # Y軸標籤改為 MDE

# 套用完整的 X 軸標籤
plt.xticks(x_positions, x_labels, fontsize=14)

# 注意：這裡拿掉了 plt.ylim(15, 105)，讓 Matplotlib 根據 MDE 的小數值 (約0~1.5) 自動調整 Y 軸縮放

# 加入網格線
plt.grid(True, linestyle='--', alpha=0.6)

# 設定圖例位置
plt.legend(loc='lower right', fontsize=16)

plt.tight_layout()
plt.savefig("localization_robustness.png", dpi=500, bbox_inches='tight')
plt.savefig("localization_robustness.pdf", format='pdf', bbox_inches='tight')