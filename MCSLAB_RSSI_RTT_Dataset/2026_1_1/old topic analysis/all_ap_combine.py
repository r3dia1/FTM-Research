import pandas as pd

# 1. 設定檔案路徑 (請依照你希望的 AP1, AP2, AP3, AP4 順序排列)
file_paths = [
    './Server_Wide_20260101_075453_location_AP1.csv',  # 這是這份將保留 Timestamp 和 Duration 的主檔
    './Server_Wide_20260101_124356_location_AP2.csv',
    './Server_Wide_20260101_065323_location_AP3.csv',
    './Server_Wide_20260101_113813_location_AP4.csv'
]

# 儲存讀取後的 DataFrames
dfs = []

# 2. 讀取並預處理每一個檔案
for i, path in enumerate(file_paths):
    # 讀取 CSV
    df = pd.read_csv(path)
    
    # --- 關鍵步驟：建立對齊用的 ID ---
    # 我們假設資料是依照時間排序的。
    # 這行程式碼會針對每個 Label，將資料標上 0, 1, 2... 的流水號
    # 這樣才能把 File1 的 "Label A 第1筆" 與 File2 的 "Label A 第1筆" 對在一起
    df['seq_id'] = df.groupby('Label').cumcount()
    
    # AP 編號 (1, 2, 3, 4)
    ap_num = i + 1
    
    if ap_num == 1:
        # 第一個檔案：保留所有欄位，不需要改名 (因為原始就是 _1)
        # 只要把 seq_id 留著做合併用
        dfs.append(df)
    else:
        # 第二個以後的檔案：
        # 1. 刪除 Timestamp, Duration_ms (因為要用第一個檔案的)
        # 2. 修改欄位名稱: 將 _1 結尾改成 _2, _3, _4
        
        # 先丟棄不需要的共用欄位，保留 Label 和 seq_id 作為 Key
        cols_to_drop = ['Timestamp', 'Duration_ms']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        
        # 重新命名欄位 (例如: RSSI_1 -> RSSI_2)
        rename_map = {}
        for col in df.columns:
            if col.endswith('_1'):
                # 把 _1 替換成 _2, _3...
                new_col = col.replace('_1', f'_{ap_num}')
                rename_map[col] = new_col
        
        df = df.rename(columns=rename_map)
        dfs.append(df)

# 3. 執行合併 (Merge)
# 以第一個 DataFrame 為基礎
df_final = dfs[0]

for i in range(1, 4):
    # 依照 'Label' 和 'seq_id' 進行合併
    # how='inner': 取交集。如果某個 Label 在某個檔案少了幾筆，會以最少的那個檔案為主，確保每一列都有完整的 4 個 AP 資料。
    df_final = pd.merge(df_final, dfs[i], on=['Label', 'seq_id'], how='inner')

# 4. 清理與排序
# 移除剛剛產生的輔助欄位 'seq_id'
df_final = df_final.drop(columns=['seq_id'])

# 依照 Timestamp 排序 (可選)
df_final = df_final.sort_values(by=['Timestamp'])

# 檢查結果格式
print(f"合併完成，資料維度: {df_final.shape}")
print("欄位列表:", df_final.columns.tolist())

# 5. 輸出成 CSV
output_filename = 'merged_wifi_dataset.csv'
df_final.to_csv(output_filename, index=False)
print(f"檔案已儲存至: {output_filename}")

# 顯示前幾筆資料預覽
print(df_final.head())