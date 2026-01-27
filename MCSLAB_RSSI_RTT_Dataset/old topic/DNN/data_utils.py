import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def process_sliding_window(df, window_size=5):
    """
    1. 對 RSSI, Dist_mm 計算 Mean 和 IQR (四分位距)。
    2. 對已經存在的 Std_mm 計算 Mean (代表這段時間的平均標準差)。
    """
    # 定義兩組要處理的欄位
    # Group A: 需要計算 Mean 和 IQR 的原始訊號
    metrics_raw = ['RSSI', 'Dist_mm']
    # Group B: 已經是統計值的欄位，只需要做平均 (Smoothing)
    metrics_existing_std = ['Std_mm']
    
    processed_rows = []
    grouped = df.groupby('Label')
    
    print(f"正在進行滑動視窗處理 (Window Size={window_size})...")
    
    for label, group in grouped:
        # 準備欄位名
        cols_raw = []
        cols_std = []
        for i in range(1, 5): # AP1 ~ AP4
            for m in metrics_raw:
                cols_raw.append(f"{m}_{i}")
            for m in metrics_existing_std:
                cols_std.append(f"{m}_{i}")
        
        # 確保欄位存在，避免報錯 (有些資料集可能沒有 Std_mm)
        cols_raw = [c for c in cols_raw if c in group.columns]
        cols_std = [c for c in cols_std if c in group.columns]

        # === 處理 Raw Signal (RSSI, Dist) -> 產出 Mean, IQR ===
        if cols_raw:
            val_raw = group[cols_raw]
            roll_raw = val_raw.rolling(window=window_size, min_periods=window_size)
            
            feat_raw_mean = roll_raw.mean()
            # 重新命名 (加上 _mean)
            feat_raw_mean.columns = [f"{c}_mean" for c in cols_raw]
            
            # 計算 IQR
            feat_raw_iqr = roll_raw.apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25), raw=True)
            feat_raw_iqr.columns = [f"{c}_iqr" for c in cols_raw]
        
        # === 處理 Existing Std (Std_mm) -> 產出 Mean ===
        # 這裡的邏輯是：這段時間內，硬體回報的 Std 平均是多少
        if cols_std:
            val_std = group[cols_std]
            roll_std = val_std.rolling(window=window_size, min_periods=window_size)
            
            feat_std_avg = roll_std.mean()
            # 這裡我們不加 _mean 後綴，或者可以保留原名，或是加上 _avg 以示區別
            # 為了讓 feature config 容易對應，這裡建議保留原名或是加上 _mean
            # 您的原始欄位叫 Std_mm_1，這裡輸出變成 Std_mm_1_mean (代表視窗內的平均 Std)
            feat_std_avg.columns = [f"{c}_mean" for c in cols_std]

        # === 合併所有特徵 ===
        concat_list = []
        if cols_raw:
            concat_list.extend([feat_raw_mean, feat_raw_iqr])
        if cols_std:
            concat_list.append(feat_std_avg)
            
        if not concat_list:
            continue
            
        features = pd.concat(concat_list, axis=1)
        features['Label'] = label
        
        # 去除前 N-1 筆 NaN
        features = features.dropna()
        processed_rows.append(features)
        
    if not processed_rows:
        return pd.DataFrame()
        
    final_df = pd.concat(processed_rows, ignore_index=True)
    return final_df

# def get_data_loaders(csv_path, feature_config, batch_size=32, window_size=10, random_state=42):
#     try:
#         df = pd.read_csv(csv_path)
#     except FileNotFoundError:
#         print(f"錯誤: 找不到檔案 {csv_path}")
#         return None

#     # 1. 轉數值與補值
#     # 這裡要包含所有可能的欄位
#     possible_metrics = ['RSSI', 'Dist_mm', 'Std_mm']
#     raw_cols = [f'{m}_{i}' for i in range(1, 5) for m in possible_metrics if f'{m}_{i}' in df.columns]

#     for col in raw_cols:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
    
#     # 補值
#     df[raw_cols] = df.groupby('Label')[raw_cols].transform(lambda x: x.fillna(x.mean())).fillna(0)

#     # 2. 執行滑動視窗
#     df_processed = process_sliding_window(df, window_size=window_size)
    
#     if df_processed is None or df_processed.empty:
#         print("錯誤：處理後無資料")
#         return None

#     # 3. 動態組裝欄位
#     # 邏輯：
#     # 如果 config 有 'RSSI' -> 抓 'RSSI_x_mean', 'RSSI_x_iqr'
#     # 如果 config 有 'Dist_mm' -> 抓 'Dist_mm_x_mean', 'Dist_mm_x_iqr'
#     # 如果 config 有 'Std_mm' -> 抓 'Std_mm_x_mean' (我們在上面把它命名為 _mean 了)
    
#     selected_cols = []
#     for i in range(1, 5): # AP1 ~ AP4
#         for f_type in feature_config:
#             col_base = f"{f_type}_{i}"
            
#             # 判斷是哪種類型
#             if f_type in ['RSSI', 'Dist_mm']:
#                 # 這些會有 mean 和 iqr
#                 col_mean = f"{col_base}_mean"
#                 col_iqr = f"{col_base}_iqr"
#                 if col_mean in df_processed.columns: selected_cols.append(col_mean)
#                 if col_iqr in df_processed.columns: selected_cols.append(col_iqr)
            
#             elif f_type == 'Std_mm':
#                 # 這個只有 mean
#                 col_mean = f"{col_base}_mean"
#                 if col_mean in df_processed.columns: selected_cols.append(col_mean)

#     print(f"使用特徵 ({len(selected_cols)}維): {selected_cols}")

#     # 檢查
#     missing = [c for c in selected_cols if c not in df_processed.columns]
#     if missing:
#         print(f"警告：找不到以下欄位 {missing}")
#         return None

#     X = df_processed[selected_cols].values
#     y_raw = df_processed['Label'].values

#     # Label Encode -> Split -> Scale -> Tensor (維持不變)
#     label_encoder = LabelEncoder()
#     y = label_encoder.fit_transform(y_raw)
    
#     X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=random_state, stratify=y)
#     val_size = 0.15 / 0.85
#     X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp)

#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_val_scaled = scaler.transform(X_val)
#     X_test_scaled = scaler.transform(X_test)

#     train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train)), batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val_scaled), torch.LongTensor(y_val)), batch_size=batch_size, shuffle=False)
#     test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test_scaled), torch.LongTensor(y_test)), batch_size=batch_size, shuffle=False)

#     meta_data = {
#         "input_dim": X.shape[1],
#         "num_classes": len(label_encoder.classes_),
#         "label_encoder": label_encoder,
#         "scaler": scaler
#     }
    
#     return train_loader, val_loader, test_loader, meta_data

def get_data_loaders(csv_path, feature_config, batch_size=32, window_size=10, random_state=42):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"錯誤: 找不到檔案 {csv_path}")
        return None

    print(f"--- 資料量檢查 ---")
    print(f"1. 原始 CSV 資料筆數: {len(df)}")

    # 1. 轉數值與補值
    # 這裡要包含所有可能的欄位
    possible_metrics = ['RSSI', 'Dist_mm', 'Std_mm']
    raw_cols = [f'{m}_{i}' for i in range(1, 5) for m in possible_metrics if f'{m}_{i}' in df.columns]

    for col in raw_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 補值
    df[raw_cols] = df.groupby('Label')[raw_cols].transform(lambda x: x.fillna(x.mean())).fillna(0)

    # 2. 執行滑動視窗
    df_processed = process_sliding_window(df, window_size=window_size)
    
    if df_processed is None or df_processed.empty:
        print("錯誤：處理後無資料")
        return None

    print(f"2. 滑動視窗處理後筆數: {len(df_processed)}")
    print(f"   (預期減少量 ≈ 總RP數 * (window_size - 1))")

    # 3. 動態組裝欄位
    # 邏輯：
    # 如果 config 有 'RSSI' -> 抓 'RSSI_x_mean', 'RSSI_x_iqr'
    # 如果 config 有 'Dist_mm' -> 抓 'Dist_mm_x_mean', 'Dist_mm_x_iqr'
    # 如果 config 有 'Std_mm' -> 抓 'Std_mm_x_mean' (我們在上面把它命名為 _mean 了)
    
    selected_cols = []
    for i in range(1, 5): # AP1 ~ AP4
        for f_type in feature_config:
            col_base = f"{f_type}_{i}"

            col_mean = f"{col_base}_mean"
            if col_mean in df_processed.columns: selected_cols.append(col_mean)
            
            # 判斷是哪種類型
            # if f_type == 'Dist_mm':
            #     # 這些會有 mean 和 iqr
            #     col_mean = f"{col_base}_mean"
            #     col_iqr = f"{col_base}_iqr"
            #     if col_mean in df_processed.columns: selected_cols.append(col_mean)
            #     if col_iqr in df_processed.columns: selected_cols.append(col_iqr)
            
            # elif f_type in ['RSSI', 'Std_mm']:
            #     # 這個只有 mean
            #     col_mean = f"{col_base}_mean"
            #     if col_mean in df_processed.columns: selected_cols.append(col_mean)

    print(f"使用特徵 ({len(selected_cols)}維): {selected_cols}")

    # 檢查
    missing = [c for c in selected_cols if c not in df_processed.columns]
    if missing:
        print(f"警告：找不到以下欄位 {missing}")
        return None
    
    X = df_processed[selected_cols].values
    y_raw = df_processed['Label'].values
    
    # Check class distribution
    unique, counts = np.unique(y_raw, return_counts=True)
    print(f"3. 每個 RP 的樣本數 (Min/Max): {min(counts)} / {max(counts)}")
    if min(counts) < 5:
        print("警告：部分 RP 的資料量過少，可能導致切分後 Train/Test 不存在")

    # Label Encode
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    
    print("正在執行時間序列切分 (防止 Sliding Window 資料洩漏)...")
    
    X_train_list, y_train_list = [], []
    X_val_list,   y_val_list   = [], []
    X_test_list,  y_test_list  = [], []

    # 針對每個 Label (RP) 獨立進行按時間順序切分
    unique_labels = np.unique(y)
    
    # 設定切分比例 (70% Train, 15% Val, 15% Test)
    train_ratio = 0.70
    val_ratio = 0.15
    # test_ratio = 0.15 (剩下的)

    for label in unique_labels:
        # 找出該 Label 的所有 indices (假設資料原本就是按時間排序的)
        indices = np.where(y == label)[0]
        
        n_samples = len(indices)
        if n_samples < 3: # 防呆，資料太少不切分，全丟 Train
             X_train_list.append(X[indices])
             y_train_list.append(y[indices])
             continue

        n_train = int(n_samples * train_ratio)
        n_val   = int(n_samples * val_ratio)
        # 剩下的給 Test
        
        # 按順序切 (關鍵：這裡絕對不能 Shuffle)
        train_idx = indices[:n_train]
        val_idx   = indices[n_train : n_train + n_val]
        test_idx  = indices[n_train + n_val :]
        
        X_train_list.append(X[train_idx])
        y_train_list.append(y[train_idx])
        
        X_val_list.append(X[val_idx])
        y_val_list.append(y[val_idx])
        
        X_test_list.append(X[test_idx])
        y_test_list.append(y[test_idx])

    # 合併所有 RP 的資料
    X_train = np.concatenate(X_train_list)
    y_train = np.concatenate(y_train_list)
    
    X_val = np.concatenate(X_val_list) if X_val_list else np.array([])
    y_val = np.concatenate(y_val_list) if y_val_list else np.array([])
    
    X_test = np.concatenate(X_test_list) if X_test_list else np.array([])
    y_test = np.concatenate(y_test_list) if y_test_list else np.array([])

    print(f"4. 最終切分結果 (Time-based):")
    print(f"   Train: {len(X_train)} 筆")
    print(f"   Val  : {len(X_val)} 筆")
    print(f"   Test : {len(X_test)} 筆")
    print(f"------------------")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val_scaled), torch.LongTensor(y_val)), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test_scaled), torch.LongTensor(y_test)), batch_size=batch_size, shuffle=False)

    meta_data = {
        "input_dim": X.shape[1],
        "num_classes": len(label_encoder.classes_),
        "label_encoder": label_encoder,
        "scaler": scaler
    }
    
    return train_loader, val_loader, test_loader, meta_data