import pandas as pd

# 設定檔案名稱
file_name = 'Debug_Generated_RTT.csv'

try:
    # 1. 讀取 CSV 檔案
    df = pd.read_csv(file_name)

    # 顯示原始資料的前幾筆（選用，方便除錯）
    print("原始資料前 5 筆：")
    print(df.head())

    # 2. 依據 'Label' 欄位從小到大排序
    # ascending=True 代表升冪（小到大），若要大到小可改為 False
    df_sorted = df.sort_values(by='Label', ascending=True)

    # 3. 寫回檔案
    # index=False 代表不要將 pandas 的索引欄位寫入檔案
    df_sorted.to_csv(file_name, index=False)

    print(f"\n成功！資料已根據 Label 排序並寫回 {file_name}")
    print("排序後資料前 5 筆：")
    print(df_sorted.head())

except FileNotFoundError:
    print(f"錯誤：找不到檔案 {file_name}，請確認檔案是否在程式執行的目錄下。")
except Exception as e:
    print(f"發生未預期的錯誤：{e}")