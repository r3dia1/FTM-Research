import pandas as pd
import os

def replace_ap_data(main_file_path, replacement_file_path, target_ap_index, output_file_path):
    """
    main_file_path: 原始包含 4個 AP 的 CSV 路徑
    replacement_file_path: 要用來替換的單一 AP CSV 路徑
    target_ap_index: 整數 1, 2, 3 或 4 (你想替換掉哪一組 AP)
    output_file_path: 輸出的檔案路徑
    """
    
    # 1. 讀取資料
    try:
        df_main = pd.read_csv(main_file_path)
        df_rep = pd.read_csv(replacement_file_path)
    except FileNotFoundError as e:
        print(f"錯誤: 找不到檔案 - {e}")
        return

    # 定義要替換的特徵欄位名稱 (不含後綴)
    # 根據你的範例，這些是主要的測量數據
    # feature_cols = ['SSID', 'BSSID', 'RSSI', 'Dist_mm', 'Std_mm', 'Succ', 'Att', 'Rate']
    feature_cols = ['SSID', 'BSSID', 'Dist_mm', 'Std_mm', 'Succ', 'Att', 'Rate']
    # feature_cols = ['SSID', 'BSSID', 'RSSI', 'Succ', 'Att', 'Rate']

    # 2. 準備欄位對應
    # 替換檔通常只有一組數據，根據你的範例是 _1 結尾
    source_suffix = "_1" 
    target_suffix = f"_{target_ap_index}"

    # 檢查替換檔是否真的有這些欄位
    source_cols = [f"{col}{source_suffix}" for col in feature_cols]
    target_cols = [f"{col}{target_suffix}" for col in feature_cols]

    # 檢查欄位是否存在
    if not all(col in df_rep.columns for col in source_cols):
        print(f"錯誤: 替換檔案中缺少必要的欄位 (預期包含 {source_cols})")
        return
    if not all(col in df_main.columns for col in target_cols):
        print(f"錯誤: 主檔案中缺少目標 AP{target_ap_index} 的欄位")
        return

    print(f"正在處理: 將 AP{target_ap_index} 的資料替換為新資料...")
    print(f"對齊規則: 依照 Label 分組，取兩者最小數量 (Minimum Count)")

    processed_dfs = []
    
    # 3. 取得所有出現過的 Label (聯集)
    unique_labels = set(df_main['Label'].unique()).union(set(df_rep['Label'].unique()))

    for label in unique_labels:
        # 篩選特定 Label 的資料
        sub_main = df_main[df_main['Label'] == label].copy()
        sub_rep = df_rep[df_rep['Label'] == label].copy()

        # 如果某個檔案完全沒有這個 Label，則跳過 (因為無法配對，且要求取最小數量，0 就是最小)
        if sub_main.empty or sub_rep.empty:
            continue

        # 4. 計算最小數量並切分 (Truncate)
        min_len = min(len(sub_main), len(sub_rep))
        
        # 只保留前 min_len 筆資料
        # reset_index 確保之後賦值時不會因為原始 index 不對齊而出錯
        sub_main = sub_main.iloc[:min_len].reset_index(drop=True)
        sub_rep = sub_rep.iloc[:min_len].reset_index(drop=True)

        # 5. 執行替換
        # 將 sub_rep 的 source columns 數值 塞入 sub_main 的 target columns
        # 使用 .values 確保只複製數值，忽略欄位名稱不匹配的問題
        sub_main[target_cols] = sub_rep[source_cols].values

        processed_dfs.append(sub_main)

    # 6. 合併並輸出
    if processed_dfs:
        final_df = pd.concat(processed_dfs, ignore_index=True)
        
        # 依照 Timestamp 排序 (可選，通常為了整齊)
        final_df = final_df.sort_values(by='Timestamp')
        
        final_df.to_csv(output_file_path, index=False)
        print(f"成功! 檔案已儲存至: {output_file_path}")
        print(f"總筆數: {len(final_df)}")
    else:
        print("警告: 處理後沒有資料 (可能是 Label 完全無法對應)")


# 假設你的檔案名稱如下 (請修改成你實際的檔名)
main_csv = './all/Server_Wide_20260101_140347.csv'      # 這是那份有很多 AP 的檔案
replace_csv = './Server_Wide_20260101_075453_location_AP1.csv'   # 這是你要拿來替換的檔案
# replace_csv = './Server_Wide_20260101_124356_location_AP2.csv'
# replace_csv = './Server_Wide_20260101_065323_location_AP3.csv'
# replace_csv = './Server_Wide_20260101_113813_location_AP4.csv'

# 設定你想替換掉主檔案中的哪一個 AP (1, 2, 3, 或 4)
target_ap = 1

output_csv = f'merged_output_ap{target_ap}_replaced.csv'

# 執行函式
replace_ap_data(main_csv, replace_csv, target_ap, output_csv)