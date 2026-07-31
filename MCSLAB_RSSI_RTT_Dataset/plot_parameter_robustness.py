import matplotlib.pyplot as plt

# 實驗數據
alphas = ['0.01', '0.1', '0.5', '1.0', '2.0']
acc = [49.83, 50.54, 51.87, 50.78, 45.54]
mde = [0.4894, 0.4654, 0.4667, 0.4671, 0.5357]

# 建立畫布
fig, ax1 = plt.subplots(figsize=(7, 4.5))

# 繪製左 Y 軸 (Top-1 Accuracy)
color_acc = 'tab:blue'
ax1.set_xlabel(r'$\alpha_{dist}$', fontsize=12)
ax1.set_ylabel('Top-1 Accuracy (%)', color=color_acc, fontsize=12, fontweight='bold')
line1 = ax1.plot(alphas, acc, color=color_acc, marker='o', linestyle='-', 
                 linewidth=2.5, markersize=8, label='Accuracy')
ax1.tick_params(axis='y', labelcolor=color_acc)
ax1.set_ylim(44, 54) # 調整上下界讓折線看起來更清楚
ax1.grid(True, linestyle='--', alpha=0.5)

# 建立共用 X 軸的右 Y 軸
ax2 = ax1.twinx()  

# 繪製右 Y 軸 (MDE)
color_mde = 'tab:red'
ax2.set_ylabel('Mean Distance Error (m)', color=color_mde, fontsize=12, fontweight='bold')
line2 = ax2.plot(alphas, mde, color=color_mde, marker='^', linestyle='--', 
                 linewidth=2.5, markersize=8, label='MDE')
ax2.tick_params(axis='y', labelcolor=color_mde)
ax2.set_ylim(0.4, 0.6)

# 合併兩條線的圖例 (Legend)
lines = line1 + line2
labels = [l.get_label() for l in lines]
# 放在左下角或圖表內適當的空白處，避免遮擋數據
ax1.legend(lines, labels, loc='lower left', framealpha=0.9) 

# 設定標題 (可以根據需求開啟或關閉，有些論文要求圖標題寫在 caption 裡)
# plt.title(r'Sensitivity Analysis of $\alpha_{dist}$', fontsize=14, fontweight='bold')

plt.tight_layout()
# 儲存高解析度圖片 (300 dpi 適合論文列印)
plt.savefig('robustness_boundary.png', dpi=500, bbox_inches='tight', pad_inches=0.05)
plt.savefig('robustness_boundary.pdf', format='pdf', bbox_inches='tight', pad_inches=0.05)
plt.show()