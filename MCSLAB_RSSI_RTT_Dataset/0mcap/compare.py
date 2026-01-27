import pandas as pd

# 1. 讀取 CSV 檔案
df1 = pd.read_csv('debug_fusion_input_data.csv')
df2 = pd.read_csv('../2026_1_1/all/Server_Wide_20260101_140347.csv')

# 2. 設定要檢查的指定欄位
target_cols = ['RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4']  # 請替換成你的欄位名稱

# 3. 提取指定欄位並進行比對
# equals() 會考慮資料順序、數值與型態
are_columns_identical = df1[target_cols].equals(df2[target_cols])

if are_columns_identical:
    print("指定欄位完全相同！")
else:
    print("指定欄位有差異。")
    
    # 進階：找出差異的地方
    # 這裡使用 compare() 來列出不同的列 (Pandas 1.1.0+)
    diff = df1[target_cols].compare(df2[target_cols])
    print("\n差異處如下：")
    print(diff)