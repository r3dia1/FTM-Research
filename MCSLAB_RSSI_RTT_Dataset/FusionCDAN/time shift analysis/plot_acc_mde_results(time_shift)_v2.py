import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_metrics_v7_line(csv_file_path, output_dir='output_charts(折線圖)'):
    """
    V7 更新: 將長條圖改為折線圖
    """
    print(f"\n>>> 正在處理檔案: {csv_file_path}")
    base_filename = os.path.splitext(os.path.basename(csv_file_path))[0]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        df = pd.read_csv(csv_file_path)
        df.columns = df.columns.str.strip()
        
        # 取得第一行數據
        row_data = df.iloc[0]
    except Exception as e:
        print(f"    [錯誤]: 讀取失敗 - {e}")
        return

    def draw_combined_plot(metric_type):
        target_cols = [col for col in df.columns if metric_type.lower() in col.lower()]
        if not target_cols:
            return

        values = [row_data[col] for col in target_cols]
        print(values)
        labels = [col.lower().replace(metric_type.lower(), '').strip().title() for col in target_cols]
        
        x_pos = np.arange(len(labels))
        plt.figure(figsize=(8, 6))
        
        # === 核心修改處：改用 plt.plot 畫折線圖 ===
        # marker='o' 加上圓點, linestyle='-' 加上實線
        if metric_type == 'mde':
            plt.plot(x_pos, values, marker='o', markersize=8, linestyle='-', linewidth=2, color="#3B8E2E")
        else:
            plt.plot(x_pos, values, marker='o', markersize=8, linestyle='-', linewidth=2, color='#4c72b0')

        # 圖表設定
        plt.title(f'{metric_type.upper()} Performance Comparison (GACDAN)', fontsize=14, fontweight='bold')
        plt.ylabel(f'{metric_type.upper()}', fontsize=12)
        plt.xticks(x_pos, labels, fontsize=10) 
        
        # 加上 XY 軸網格讓折線圖更容易對齊數值
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        plt.grid(axis='x', linestyle='--', alpha=0.2) 

        # 在資料點上方標示數值 (移除 bbox 白底)
        if metric_type == 'mde':
            for i, val in enumerate(values):
                plt.annotate(
                    f'{val:.4f}',                  # 顯示的文字
                    xy=(x_pos[i], val),            # 資料點座標
                    xytext=(0, 6),                 # 往上偏移 8 個單位
                    textcoords='offset points',    # 偏移單位的基準為 pixel points
                    ha='center', va='bottom', 
                    fontsize=10, fontweight='bold'
                )
        else:
            for i, val in enumerate(values):
                plt.annotate(
                    f'{val:.2f}',                  # 顯示的文字
                    xy=(x_pos[i], val),            # 資料點座標
                    xytext=(0, 6),                 # 往上偏移 8 個單位
                    textcoords='offset points',    # 偏移單位的基準為 pixel points
                    ha='center', va='bottom', 
                    fontsize=10, fontweight='bold'
                )

        plt.tight_layout()

        # 儲存
        save_name = f"{base_filename}_{metric_type.upper()}_Line_Comparison.png"
        plt.savefig(os.path.join(output_dir, save_name), dpi=300)
        plt.close()
        print(f"    -> 已產生: {save_name}")

    draw_combined_plot('acc')
    draw_combined_plot('mde')

# --- 執行處 ---
if __name__ == "__main__":
    csv_files = ["./time_shift_by_date_2_4 copy 2.csv"]

    for csv_file in csv_files:
        if os.path.exists(csv_file):
            plot_metrics_v7_line(csv_file)
        else:
            print(f"找不到檔案: {csv_file}")