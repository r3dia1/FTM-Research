import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_metrics_v7(csv_file_path, output_dir='output_charts_mcapTest'):
    """
    V7 更新:
    1. 針對新格式：Source/Target + Fusion/RTT + ACC/MDE。
    2. 自動將所有 ACC 欄位繪製於同一張圖，MDE 亦然。
    3. 自動美化標籤名稱。
    """
    print(f"\n>>> 正在處理檔案: {csv_file_path}")
    base_filename = os.path.splitext(os.path.basename(csv_file_path))[0]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # 讀取資料 (處理無 Header 或 逗號分隔的情況)
        df = pd.read_csv(csv_file_path)
        print(df)
        df.columns = df.columns.str.strip()
        
        # 取得第一行數據 (假設新資料只有一行)
        row_data = df.iloc[0]
        print(row_data)
    except Exception as e:
        print(f"    [錯誤]: 讀取失敗 - {e}")
        return

    def draw_combined_plot(metric_type):
        # 篩選包含該指標的欄位 (例如 'acc' 或 'mde')
        target_cols = [col for col in df.columns if metric_type.lower() in col.lower()]
        if not target_cols:
            return
        print(target_cols)

        values = [row_data[col] for col in target_cols]
        print(values)
        
        # 格式化標籤：把 "source fusion acc" 變成 "Source Fusion"
        labels = [col.lower().replace(metric_type.lower(), '').strip().title() for col in target_cols]
        
        x_pos = np.arange(len(labels))
        plt.figure(figsize=(6, 4))
        
        # 配色方案
        colors = ['#4c72b0', '#55a868', '#dd8452', '#c44e52', '#8172b3']
        
        bars = plt.bar(x_pos, values, width=0.35, color=colors[:len(values)])

        # --- 【修改重點 1】：動態調整 Y 軸範圍放大差距 ---
        min_val = min(values)
        max_val = max(values)
        diff = max_val - min_val
        
        # 如果有差距，上下多留 50% 的空間；如果數值完全一樣，就留 5% 的預設空間
        padding = diff * 0.5 if diff > 0 else (max_val * 0.05 if max_val != 0 else 0.1)
        
        # 設定 Y 軸上下限 (確保下限不會變成負的)
        plt.ylim(max(0, min_val - padding), max_val + padding * 1.2) 
        # --------------------------------------------------

        # 圖表設定
        plt.title(f'{metric_type.upper()} Performance Comparison (DNN)', fontsize=14, fontweight='bold')
        plt.ylabel('Value', fontsize=12)
        plt.xticks(x_pos, labels, fontsize=10) 
        plt.grid(axis='y', linestyle='--', alpha=0.4)

        # --- 【修改重點 2】：控制小數點顯示位數 ---
        for bar in bars:
            height = bar.get_height()
            # 使用 :.4g 或 :.3f 來確保小數點後的細微差異能被印出來 (這裡預設顯示到小數後第3位)
            plt.text(bar.get_x() + bar.get_width()/2, height, 
                     f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        # --------------------------------------------------

        plt.tight_layout()

        # 儲存
        save_name = f"{base_filename}_{metric_type.upper()}_Comparison.png"
        plt.savefig(os.path.join(output_dir, save_name), dpi=300)
        plt.close()
        print(f"    -> 已產生: {save_name}")

    # 執行繪圖
    draw_combined_plot('acc')
    draw_combined_plot('mde')

# --- 執行處 ---
if __name__ == "__main__":
    # 請確保此路徑下的 CSV 內容如你所述
    csv_files = ["./mcap_used_test.csv"]
    # csv_files = ["./time_shift_by_date_2_4 copy.csv"]

    for csv_file in csv_files:
        if os.path.exists(csv_file):
            plot_metrics_v7(csv_file)
        else:
            print(f"找不到檔案: {csv_file}")