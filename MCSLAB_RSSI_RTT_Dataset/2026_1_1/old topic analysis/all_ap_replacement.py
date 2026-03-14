import pandas as pd
import os

def replace_multiple_aps(main_file_path, replacement_map, output_file_path):
    """
    同時替換多個 AP 資料的函式。
    
    Args:
        main_file_path: 原始包含 4個 AP 的 CSV 路徑
        replacement_map: 一個字典，格式為 { AP編號: 檔案路徑 }
                         例如: {1: 'path/to/ap1.csv', 2: 'path/to/ap2.csv', ...}
        output_file_path: 輸出的檔案路徑
    """
    
    # 定義特徵欄位 (不含後綴)
    feature_cols = ['SSID', 'BSSID', 'Dist_mm', 'Std_mm', 'Succ', 'Att', 'Rate']
    # 來源檔案(替換檔)通常預設後綴是 _1
    source_suffix = "_1" 

    # 1. 讀取主檔案
    print(f"正在讀取主檔案: {main_file_path}")
    try:
        df_main = pd.read_csv(main_file_path)
    except FileNotFoundError:
        print(f"錯誤: 找不到主檔案")
        return

    # 2. 讀取所有替換檔案並存入字典
    # 結構: rep_dfs = { 1: dataframe_ap1, 2: dataframe_ap2, ... }
    rep_dfs = {}
    
    for ap_idx, file_path in replacement_map.items():
        try:
            print(f"正在讀取 AP{ap_idx} 替換檔: {file_path}")
            df_temp = pd.read_csv(file_path)
            
            # 簡單檢查必要欄位是否存在
            source_cols = [f"{col}{source_suffix}" for col in feature_cols]
            if not all(col in df_temp.columns for col in source_cols):
                print(f"錯誤: AP{ap_idx} 替換檔缺少必要欄位，跳過此 AP。")
                continue
                
            rep_dfs[ap_idx] = df_temp
        except FileNotFoundError:
            print(f"錯誤: 找不到 AP{ap_idx} 的檔案: {file_path}")
            return

    if not rep_dfs:
        print("沒有成功讀取任何替換檔案，程式結束。")
        return

    print("-" * 30)
    print("開始處理資料對齊與替換...")
    print("對齊規則: 取主檔與所有替換檔在該 Label 下的「全域最小筆數」")

    processed_dfs = []

    # 3. 取得所有檔案 Label 的聯集 (確保不會漏掉任何出現過的 Label)
    all_labels = set(df_main['Label'].unique())
    for df in rep_dfs.values():
        all_labels = all_labels.union(set(df['Label'].unique()))

    # 4. 針對每個 Label 進行處理
    for label in all_labels:
        # 4-1. 提取主檔該 Label 的資料
        sub_main = df_main[df_main['Label'] == label].copy()
        
        # 如果主檔沒這個 Label，這組資料就廢了 (因為無法合併進主結構)，跳過
        if sub_main.empty:
            continue

        # 4-2. 提取所有替換檔該 Label 的資料
        sub_reps = {} # 用來暫存該 Label 的替換資料
        current_lengths = [len(sub_main)] # 用來算最小長度
        
        skip_label = False
        for ap_idx, df_rep in rep_dfs.items():
            sub_r = df_rep[df_rep['Label'] == label].copy()
            
            # 如果某個要替換的 AP 檔案裡完全沒有這個 Label
            # 代表這組數據無法湊齊，必須跳過這個 Label (或視為 0 筆)
            if sub_r.empty:
                skip_label = True
                break
            
            sub_reps[ap_idx] = sub_r
            current_lengths.append(len(sub_r))
        
        if skip_label:
            # print(f"Label {label} 資料不全，跳過。") 
            continue

        # 4-3. 計算全域最小長度 (Global Minimum Length)
        # 為了讓同一 Row 的 4 個 AP 資料對應，必須切成大家都有的長度
        min_len = min(current_lengths)
        
        if min_len == 0:
            continue

        # 4-4. 切分主檔 (Truncate)
        sub_main = sub_main.iloc[:min_len].reset_index(drop=True)

        # 4-5. 執行替換迴圈
        for ap_idx, sub_r in sub_reps.items():
            # 切分替換檔
            sub_r = sub_r.iloc[:min_len].reset_index(drop=True)
            
            # 定義欄位名稱
            source_cols = [f"{col}{source_suffix}" for col in feature_cols] # 來源通常是 _1
            target_cols = [f"{col}_{ap_idx}" for col in feature_cols]       # 目標是 _1, _2, _3, _4
            
            # 賦值替換
            sub_main[target_cols] = sub_r[source_cols].values

        processed_dfs.append(sub_main)

    # 5. 合併與輸出
    if processed_dfs:
        final_df = pd.concat(processed_dfs, ignore_index=True)
        
        # 依照 Timestamp 排序
        if 'Timestamp' in final_df.columns:
            final_df = final_df.sort_values(by='Timestamp')
            
        final_df.to_csv(output_file_path, index=False)
        print(f"成功! 檔案已儲存至: {output_file_path}")
        print(f"總筆數: {len(final_df)}")
    else:
        print("警告: 處理後沒有資料 (可能是 Label 完全無法對應)")

# ==========================================
# 使用範例
# ==========================================

# 1. 主檔案
main_csv = '../2025_1_3/mcslab_2025_1_3.csv'

# 2. 設定要替換的 AP 對應表 { AP編號 : 檔案路徑 }
# 你可以只放 1 個，也可以放 4 個，程式會自動處理
ap_replacement_map = {
    1: './Server_Wide_20260101_075453_location_AP1.csv',
    2: './Server_Wide_20260101_124356_location_AP2.csv',
    3: './Server_Wide_20260101_065323_location_AP3.csv',
    4: './Server_Wide_20260101_113813_location_AP4.csv'
}

output_csv = 'merged_output_ALL_APs_replaced.csv'

# 執行
replace_multiple_aps(main_csv, ap_replacement_map, output_csv)