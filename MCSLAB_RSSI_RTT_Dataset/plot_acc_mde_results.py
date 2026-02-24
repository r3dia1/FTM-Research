import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_with_legend(csv_file_path, output_dir='output_charts_legend(GACDAN)'):
    """
    V6 更新:
    1. 隱藏 X 軸過長的文字。
    2. 使用 Legend (圖例) 在右側顯示架構名稱。
    3. 針對每一個 mcAP amount 分別產出圖片 (共8張)。
    4. 數值完全不四捨五入。
    """
    print(f"\n>>> 正在處理檔案: {csv_file_path}")
    
    base_filename = os.path.splitext(os.path.basename(csv_file_path))[0]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # 讀取與清理資料
        df = pd.read_csv(csv_file_path, sep=None, engine='python')
        df.columns = df.columns.str.strip()

        # 強制轉值
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 設定 X 軸 (mcAP amounts)
        index_col = None
        for col in df.columns:
            if 'mcap' in col.lower() or 'amount' in col.lower():
                index_col = col
                break
        
        if index_col:
            df.set_index(index_col, inplace=True)
        else:
            df.set_index(df.columns[0], inplace=True)

        if df.shape[0] == 0:
            print("    [錯誤]: 資料為空。")
            return

    except Exception as e:
        print(f"    [讀取失敗]: {e}")
        return

    # --- 定義單張圖繪製函數 ---
    def draw_single_plot(data_series, metric_type, mcap_val):
        """
        data_series: 某一列的數據
        metric_type: 'Accuracy' 或 'MDE'
        mcap_val: 當前的 mcAP 數值
        """
        # 準備架構名稱 (用於 Legend)
        arch_names = []
        for name in data_series.index:
            clean_name = name.lower()
            if 'acc' in clean_name:
                clean_name = clean_name.replace('acc', '').strip()
            elif 'mde' in clean_name:
                clean_name = clean_name.replace('mde', '').strip()
            arch_names.append(clean_name.upper()) # 轉大寫

        values = data_series.values
        x_pos = np.arange(len(values)) # X 軸的位置 [0, 1, 2, 3...]

        # 設定畫布大小 (寬度稍微加寬以容納 Legend)
        plt.figure(figsize=(9, 6))
        
        # 定義顏色 (可依需求新增更多顏色)
        colors = ['#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b3', '#937860']
        # 確保顏色數量足夠
        if len(values) > len(colors):
            colors = colors * 2
            
        # 繪製長條圖
        bars = plt.bar(x_pos, values, width=0.6, color=colors[:len(values)])

        # 設定標題與軸標籤
        plt.title(f'{metric_type} Comparison (mcAP = {mcap_val})\n[{base_filename}]', fontsize=14, fontweight='bold')
        plt.ylabel(metric_type, fontsize=12)
        
        # 【關鍵修改】移除 X 軸下方的文字標籤 (因為太長)
        plt.xticks([]) 
        plt.xlabel('Architectures', fontsize=12)

        # 【關鍵修改】加入 Legend (圖例)
        # bbox_to_anchor=(1.02, 1): 將圖例放在圖表右側外
        plt.legend(bars, arch_names, title="Architecture Models", 
                   bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

        plt.grid(axis='y', linestyle='--', alpha=0.5)

        # 在柱狀圖上方標示原始數值
        for bar, original_val in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height, 
                     str(original_val), # 不四捨五入
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout() # 自動調整佈局避免 Legend 被切掉

        # 儲存圖片
        save_name = f"{base_filename}_mcAP_{mcap_val}_{metric_type}.png"
        save_path = os.path.join(output_dir, save_name)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"    -> 已產生: {save_name}")

    # --- 主迴圈 ---
    for mcap_val in df.index:
        row_data = df.loc[mcap_val]

        # 1. 篩選 ACC 並繪圖
        acc_data = row_data[row_data.index.str.contains('acc', case=False)]
        if not acc_data.empty:
            draw_single_plot(acc_data, 'Accuracy', mcap_val)

        # 2. 篩選 MDE 並繪圖
        mde_data = row_data[row_data.index.str.contains('mde', case=False)]
        if not mde_data.empty:
            draw_single_plot(mde_data, 'MDE', mcap_val)

# --- 主程式 ---

if __name__ == "__main__":
    # 設定您的檔案路徑
    # csv_files = ["Avg_Acc_MDE_record(source_1_1)/avg_result_1_14_with_dsbn.csv"] 
    # csv_files = ["Avg_Acc_MDE_record(source_1_1)/avg_result_1_23_with_dsbn.csv"]
    # csv_files = ["Avg_Acc_MDE_record(source_1_1)/avg_result_1_28_with_dsbn.csv"]
    # csv_files = ["Avg_Acc_MDE_record(source_1_1)/avg_result_2_4_with_dsbn.csv"]
    # csv_files = ["Avg_Acc_MDE_record(source_1_1)/noRSSIDiff baseline/avg_result_1_14.csv"] 
    # csv_files = ["Avg_Acc_MDE_record(source_1_1)/noRSSIDiff baseline/avg_result_1_23.csv"]
    # csv_files = ["Avg_Acc_MDE_record(source_1_1)/noRSSIDiff baseline/avg_result_1_28.csv"]
    # csv_files = ["Avg_Acc_MDE_record(source_1_1)/noRSSIDiff baseline/avg_result_2_4.csv"]
    # csv_files = ["Avg_Acc_MDE_record(source_1_1)/GACDAN/avg_result_1_14.csv"] 
    # csv_files = ["Avg_Acc_MDE_record(source_1_1)/GACDAN/avg_result_1_23.csv"]
    # csv_files = ["Avg_Acc_MDE_record(source_1_1)/GACDAN/avg_result_1_28.csv"]
    csv_files = ["Avg_Acc_MDE_record(source_1_1)/GACDAN/avg_result_2_4.csv"]

    print(f"當前工作目錄: {os.getcwd()}")
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            plot_with_legend(csv_file)
        else:
            print(f"錯誤: 找不到檔案: {csv_file}")