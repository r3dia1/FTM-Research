import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

# 1. 直接讀取 CSV 檔案
# 請確保 experiment_summary_2_4.csv 與此 Python 程式碼在同一個資料夾
file_path = 'experiment_summary_2_4.csv'
df = pd.read_csv(file_path)

# 從資料表中提取出需要的欄位
models = df['Combo'].tolist()
acc_data = df['Avg_Tgt_Acc'].tolist()
mde_data = df['Avg_Tgt_MDE'].tolist()

# 2. 設定圖片樣式與顏色 (完美對齊你提供的漸層藍色系)
colors = ['#d1e0f0', '#8cb4d6', '#4a76a8', '#1a325a']
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

# 由於資料只有一組 Target Domain，我們將其設定為一個 Group
x = np.arange(1)  
width = 0.15

# --- 3. 繪製左圖：Avg_Tgt_Acc ---
ax1 = axes[0]
multiplier = 0
for i, model in enumerate(models):
    offset = width * multiplier
    ax1.bar(x + offset, acc_data[i], width, label=model, color=colors[i], edgecolor='white', linewidth=0.5)
    multiplier += 1

ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_xticks(x + width * 1.5)
ax1.set_xticklabels(['Target Domain'], fontsize=12)
ax1.set_ylim(0, max(acc_data) * 1.2) 
ax1.legend(loc='upper center', ncol=2, fontsize=10, handlelength=1, framealpha=0.9)
ax1.text(0.5, -0.15, '(a) Average Target Accuracy', transform=ax1.transAxes, ha='center', fontsize=13)

# --- 4. 繪製右圖：Avg_Tgt_MDE ---
ax2 = axes[1]
multiplier = 0
for i, model in enumerate(models):
    offset = width * multiplier
    ax2.bar(x + offset, mde_data[i], width, label=model, color=colors[i], edgecolor='white', linewidth=0.5)
    multiplier += 1

ax2.set_ylabel('Mean Direction Error (MDE)', fontsize=12)
ax2.set_xticks(x + width * 1.5)
ax2.set_xticklabels(['Target Domain'], fontsize=12)
ax2.set_ylim(0, max(mde_data) * 1.2)
ax2.legend(loc='upper center', ncol=2, fontsize=10, handlelength=1, framealpha=0.9)
ax2.text(0.5, -0.15, '(b) Average Target MDE', transform=ax2.transAxes, ha='center', fontsize=13)

# 5. 整體排版與顯示
plt.tight_layout()
plt.subplots_adjust(bottom=0.2) 

# --- 新增：將圖片儲存下來 ---
# dpi=300 可以確保圖片在高解析度螢幕或列印時依然清晰
# bbox_inches='tight' 可以避免圖片邊緣被意外裁切
plt.savefig('ablation_study_results.png', dpi=500, bbox_inches='tight')
print("圖片已成功儲存為 'ablation_study_results.png'")

plt.show()