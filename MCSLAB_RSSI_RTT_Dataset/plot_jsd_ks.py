import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 產生模擬的漂移數據 (你可以替換成真實 df 的 d1, d2)
# ==========================================
np.random.seed(42)
# 模擬 Source 數據 (例如 2026/1/1)
source_data = np.random.normal(loc=-61, scale=5, size=500)
# 模擬 Target 數據 (例如 2026/2/4，發生了偏移和微小變形)
target_data = np.random.normal(loc=-63, scale=6, size=500) 

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ==========================================
# 圖一：JSD 視覺化 (山峰圖 / KDE)
# 概念：測量兩座山峰「不重疊」的整體形狀差異
# ==========================================
ax1 = axes[0]
sns.kdeplot(source_data, ax=ax1, label='Source (2026/1/1)', fill=True, color='#1f77b4', alpha=0.4, linewidth=2.5)
sns.kdeplot(target_data, ax=ax1, label='Target (2026/2/4)', fill=True, color='#ff7f0e', alpha=0.4, linewidth=2.5)

ax1.set_title("JSD: Measuring Distribution Overlap (KDE)", fontsize=15, fontweight='bold')
ax1.set_xlabel("RSSI (dBm)", fontsize=13)
ax1.set_ylabel("Probability Density", fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.5)

# 加入文字方塊解釋 JSD
ax1.text(-50, 0.05, "JSD Evaluates:\nGlobal Shape Distortion\n& Non-Overlapping Area", 
         fontsize=12, color='darkred', 
         bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))


# ==========================================
# 圖二：KS Stat 視覺化 (S型爬坡圖 / CDF)
# 概念：測量兩條累積曲線在垂直方向上的「最大裂縫」
# ==========================================
ax2 = axes[1]
sns.ecdfplot(source_data, ax=ax2, label='Source (2026/1/1)', color='#1f77b4', linewidth=2.5)
sns.ecdfplot(target_data, ax=ax2, label='Target (2026/2/4)', color='#ff7f0e', linewidth=2.5)

# --- 計算 KS Stat 的最大垂直距離並畫出紅線 ---
x_concat = np.sort(np.concatenate([source_data, target_data]))
cdf_source = np.array([np.mean(source_data <= x) for x in x_concat])
cdf_target = np.array([np.mean(target_data <= x) for x in x_concat])
diffs = np.abs(cdf_source - cdf_target)

# 找出最大差距的索引與座標
max_idx = np.argmax(diffs)
max_x = x_concat[max_idx]
max_y1 = cdf_source[max_idx]
max_y2 = cdf_target[max_idx]

# 繪製代表 KS Stat 的紅色垂直線與端點
ax2.plot([max_x, max_x], [max_y1, max_y2], color='red', linestyle='--', linewidth=3, zorder=5)
ax2.scatter([max_x, max_x], [max_y1, max_y2], color='red', s=60, zorder=6)

ax2.set_title("KS Stat: Measuring Maximum Gap (CDF)", fontsize=15, fontweight='bold')
ax2.set_xlabel("RSSI (dBm)", fontsize=13)
ax2.set_ylabel("Cumulative Probability", fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.5)

# 標註 KS Stat 數值
ax2.text(max_x + 1, (max_y1 + max_y2)/2, f"KS Stat (Max Gap)\nD = {diffs[max_idx]:.3f}", 
         fontsize=12, color='red', fontweight='bold', verticalalignment='center',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

plt.tight_layout()
plt.savefig("Distribution_Metrics_Explanation.png", dpi=300, bbox_inches='tight')
print("Image saved as Distribution_Metrics_Explanation.png")