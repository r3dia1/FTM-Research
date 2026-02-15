import pandas as pd
import os

def clean_and_report(file_path, output_path=None):
    """
    讀取 CSV，刪除任何包含缺失值 (NaN) 的列，並依據 Label 輸出統計報告。
    """
    # 檢查檔案是否存在
    if not os.path.exists(file_path):
        print(f"錯誤: 找不到檔案 {file_path}")
        return

    # 1. 讀取資料
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"讀取 CSV 失敗: {e}")
        return

    print(f"--- 開始處理檔案: {file_path} ---")
    print(f"原始資料總筆數: {len(df)}")

    # 2. 統計每個 Label 的原始數量
    # value_counts() 會計算每個 Label 出現幾次
    original_counts = df['Label'].value_counts().sort_index()

    # 3. 執行刪除動作 (一旦有 NaN 就丟掉整列)
    # how='any' 代表只要該列有一個欄位是空值，就刪除
    df_clean = df.dropna(how='any')
    
    # 4. 統計每個 Label 的剩餘數量
    remaining_counts = df_clean['Label'].value_counts().sort_index()

    # 5. 整合統計報表
    # 將兩個 Series 合併成一個 DataFrame 以便計算
    stats_df = pd.DataFrame({
        'Original': original_counts,
        'Remaining': remaining_counts
    })

    # 處理可能某個 Label 全部被刪光的情況 (NaN 會變成 0)
    stats_df['Remaining'] = stats_df['Remaining'].fillna(0).astype(int)

    # 計算被丟掉的數量
    stats_df['Dropped'] = stats_df['Original'] - stats_df['Remaining']

    # 重新排列欄位順序，方便閱讀
    stats_df = stats_df[['Original', 'Dropped', 'Remaining']]
    stats_df.index.name = 'Label'

    # 6. 輸出統計結果到畫面
    print("\n=== 資料清洗統計報告 (依 Label 分組) ===")
    # 設置顯示選項，確保印出所有列，不會被省略
    pd.set_option('display.max_rows', None) 
    print(stats_df)
    print("========================================")
    print(f"\n總共移除列數: {len(df) - len(df_clean)}")
    print(f"清洗後資料總筆數: {len(df_clean)}")

    # (選用) 如果有指定輸出路徑，將清洗後的資料存檔
    if output_path:
        df_clean.to_csv(output_path, index=False)
        print(f"\n清洗後的資料已儲存至: {output_path}")

# ==========================================
# 使用範例
# ==========================================

# 請將這裡換成你的 CSV 檔案路徑
input_csv = 'All_Data_With_RSSI_Diff.csv' 
output_csv = 'All_Data_With_RSSI_Diff_withoutNA.csv' # 處理好的檔案要存去哪 (選用)



if __name__ == '__main__':
    # 執行主程式
    # 如果你是直接傳入檔名，確保檔案在同一目錄下
    clean_and_report(input_csv, output_csv)