import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# --- 1. 加入 IEEE 論文通用字型與排版設定 ---
# --- 1. 加入 IEEE 論文通用字型與排版設定 ---
plt.rcParams['font.family'] = 'serif'
# 給予多個備選字體，找不到前面的就會往後找
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'Bitstream Vera Serif']
plt.rcParams['font.size'] = 12
def plot_ieee_style(csv_file_path, output_dir='output_charts_IEEE_Style'):
    """
    IEEE 論文風格更新:
    1. 全局使用 Times New Roman 字體。
    2. 移除圖表上方標題 (交由論文排版的 Caption 處理)。
    3. 長條圖加上黑框，並填入紋理 (Hatch) 以利黑白印刷。
    4. 捨棄右側 Legend，將架構名稱傾斜 45 度顯示在 X 軸下方。
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
            # 遇到過長的名字可以手動加入換行，或依賴下方的 45 度傾斜
            arch_names.append(clean_name.upper()) 

        values = data_series.values
        x_pos = np.arange(len(values))

        # 設定畫布大小 (8x5 比例適合論文單欄或雙欄縮放)
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # --- 2. 論文風格長條圖：淺灰底色 + 黑邊框 ---
        bars = ax.bar(x_pos, values, width=0.6, color='#E8E8E8', edgecolor='black', linewidth=1.2)

        # --- 3. 論文風格紋理 (Hatches) ---
        # 準備足夠的紋理樣式，*2 代表讓紋理密一點
        hatches = ['//', '\\\\', 'xx', '--', '++', '||', 'oo', 'OO', '..', '**']
        for i, bar in enumerate(bars):
            # 循環套用紋理
            bar.set_hatch(hatches[i % len(hatches)])

        # 設定軸標籤
        ax.set_ylabel(metric_type, fontsize=14)
        ax.set_xlabel('Architectures', fontsize=14)

        # --- 4. X 軸標籤傾斜 45 度，取代 Legend ---
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
            # 稍微將文字往上提一點點，避免與長條框線重疊
            ax.text(bar.get_x() + bar.get_width()/2, height + (max(values)*0.015), 
                     str(original_val), 
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

        # 為了容納上方的數字，稍微拉高 Y 軸的上限
        ax.set_ylim(0, max(values) * 1.15)

        # 自動調整佈局，避免 X 軸傾斜的文字被切掉
        plt.tight_layout()

        # 儲存圖片 (高解析度 PNG 與 論文最愛的 PDF 向量圖)
        save_name_png = f"{base_filename}_mcAP_{mcap_val}_{metric_type}.png"
        save_name_pdf = f"{base_filename}_mcAP_{mcap_val}_{metric_type}.pdf"
        
        save_path_png = os.path.join(output_dir, save_name_png)
        save_path_pdf = os.path.join(output_dir, save_name_pdf)
        
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight') # 順便存一份 PDF 給 LaTeX 用
        
        plt.close()
        print(f"    -> 已產生: {save_name_png} / .pdf")

    # --- 主迴圈 ---
    for mcap_val in df.index:
        row_data = df.loc[mcap_val]

        # 1. 篩選 ACC 並繪圖
        acc_data = row_data[row_data.index.str.contains('acc', case=False)]
        if not acc_data.empty:
            draw_single_plot(acc_data, 'Accuracy (%)', mcap_val) # 論文習慣加上單位

        # 2. 篩選 MDE 並繪圖
        mde_data = row_data[row_data.index.str.contains('mde', case=False)]
        if not mde_data.empty:
            draw_single_plot(mde_data, 'MDE', mcap_val)

# --- 主程式 ---
if __name__ == "__main__":
    # 測試用路徑 (請依您的實際環境修改)
    csv_files = ["Avg_Acc_MDE_record(source_1_1)/GACDAN/avg_result_2_4.csv"]

    print(f"當前工作目錄: {os.getcwd()}")
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            plot_ieee_style(csv_file)
        else:
            print(f"錯誤: 找不到檔案: {csv_file}")