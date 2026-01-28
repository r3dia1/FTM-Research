import pandas as pd
import itertools

# ORIGIN_PATH = '/home/mcslab/yutung/MCSLAB_RSSI_RTT_Dataset/2026_1_1/all/Server_Wide_20260101_140347.csv'
# TARGET_PATH = '/home/mcslab/yutung/MCSLAB_RSSI_RTT_Dataset/2026_1_1/all/All_Data_With_RSSI_Diff.csv'
ORIGIN_PATH = '/home/mcslab/yutung/MCSLAB_RSSI_RTT_Dataset/2026_1_28/Server_Wide_20260128_082804.csv'
TARGET_PATH = '/home/mcslab/yutung/MCSLAB_RSSI_RTT_Dataset/2026_1_28/All_Data_With_RSSI_Diff.csv'

def process_differential_rssi(input_file, output_file):
    # 1. 讀取 CSV
    df = pd.read_csv(input_file)

    # 2. 自動偵測 RSSI 欄位 (假設欄位名稱格式為 RSSI_x)
    rssi_cols = [col for col in df.columns if col.startswith('RSSI_')]
    
    if not rssi_cols:
        print("未偵測到 RSSI 欄位，請確認 CSV 標頭名稱。")
        return

    print(f"偵測到的 RSSI 欄位: {rssi_cols}")

    # 3. 計算兩兩差分 (Differential RSSI)
    # 使用 itertools.combinations 取出所有不重複的配對
    pairs = list(itertools.combinations(rssi_cols, 2))

    for col_a, col_b in pairs:
        # 從欄位名稱取出編號 (例如 RSSI_1 -> 1)
        idx_a = col_a.split('_')[-1]
        idx_b = col_b.split('_')[-1]
        
        # 定義新欄位名稱，例如 Diff_RSSI_1_2
        new_col_name = f'Diff_RSSI_{idx_a}_{idx_b}'
        
        # 計算差值 (A - B)
        # 注意：若有缺失值 (NaN)，Pandas 會自動處理為 NaN
        df[new_col_name] = df[col_a] - df[col_b]

    # 4. 寫入新的 CSV
    df.to_csv(output_file, index=False)
    print(f"處理完成！新檔案已儲存為: {output_file}")
    print(f"新增了 {len(pairs)} 個差分特徵欄位。")

# 使用範例 (請替換成您的實際檔名)
process_differential_rssi(ORIGIN_PATH, TARGET_PATH)