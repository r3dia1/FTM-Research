import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# --- 1. 加入 IEEE 論文通用字型與排版設定 (加入備用字體防呆) ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'Bitstream Vera Serif']
plt.rcParams['font.size'] = 12

def plot_ieee_style_color(csv_file_path, output_dir='output_charts_IEEE_Color'):
    """
    IEEE 論文風格 (彩色版) 更新:
    1. 移除斜線網格等紋理，畫面更乾淨。
    2. 使用高質感的學術彩色色系 (Seaborn Deep 質感)。
    3. 保留黑框與字體設定，維持論文專業度。
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
        
        # 準備架構名稱
        arch_names = []
        for name in data_series.index:
            clean_name = name.lower()
            if 'acc' in clean_name:
                clean_name = clean_name.replace('acc', '').strip()
            elif 'mde' in clean_name:
                clean_name = clean_name.replace('mde', '').strip()
            arch_names.append(clean_name.upper()) 

        values = data_series.values
        x_pos = np.arange(len(values))

        # 設定畫布大小
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # --- 2. 高質感學術彩色色盤 (類似 Seaborn Deep) ---
        colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', 
                  '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD']
        # 如果模型數量超過顏色數量，就讓顏色循環使用
        if len(values) > len(colors):
            colors = colors * (len(values) // len(colors) + 1)
            
        # --- 3. 繪製彩色長條圖 (保留 edgecolor='black' 增加俐落感) ---
        bars = ax.bar(x_pos, values, width=0.6, 
                      color=colors[:len(values)], edgecolor='black', linewidth=1.2)

        # 設定軸標籤
        ax.set_ylabel(metric_type, fontsize=14)
        ax.set_xlabel('Architectures', fontsize=14)

        # X 軸標籤傾斜 45 度
        ax.set_xticks(x_pos)
        ax.set_xticklabels(arch_names, rotation=45, ha='right', fontsize=11)

        # 設定 Y 軸網格線 (放在長條圖後方)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

        # 加上圖表外圍的實心黑框 (Spines)
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.2)

        # 在柱狀圖上方標示原始數值
        for bar, original_val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + (max(values)*0.015), 
                     str(original_val), 
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

        # 為了容納上方的數字，稍微拉高 Y 軸的上限
        ax.set_ylim(0, max(values) * 1.15)

        # 自動調整佈局
        plt.tight_layout()

        # 儲存圖片
        save_name_png = f"{base_filename}_mcAP_{mcap_val}_{metric_type}.png"
        save_name_pdf = f"{base_filename}_mcAP_{mcap_val}_{metric_type}.pdf"
        
        save_path_png = os.path.join(output_dir, save_name_png)
        save_path_pdf = os.path.join(output_dir, save_name_pdf)
        
        plt.savefig(save_path_png, dpi=500, bbox_inches='tight')
        plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')
        
        plt.close()
        print(f"    -> 已產生: {save_name_png} / .pdf")

    # --- 主迴圈 ---
    for mcap_val in df.index:
        row_data = df.loc[mcap_val]

        # 1. 篩選 ACC 並繪圖
        acc_data = row_data[row_data.index.str.contains('acc', case=False)]
        if not acc_data.empty:
            draw_single_plot(acc_data, 'Accuracy (%)', mcap_val)

        # 2. 篩選 MDE 並繪圖
        mde_data = row_data[row_data.index.str.contains('mde', case=False)]
        if not mde_data.empty:
            draw_single_plot(mde_data, 'MDE', mcap_val)

# --- 主程式 ---
if __name__ == "__main__":
    csv_files = ["Avg_Acc_MDE_record(source_1_1)/GACDAN/source target 1:0.5/avg_result_2_4.csv"]

    print(f"當前工作目錄: {os.getcwd()}")
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            plot_ieee_style_color(csv_file)
        else:
            print(f"錯誤: 找不到檔案: {csv_file}")