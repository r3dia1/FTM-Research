import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def get_data_loaders(csv_path, feature_types, batch_size=16, random_state=42):
    """
    Args:
        feature_types (list): 例如 ['RSSI', 'Dist_mm', 'Std_mm'] 或只選 ['RSSI']
        random_state (int): 控制切分種子
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"錯誤: 找不到檔案 {csv_path}")
        return None

    # --- 1. 動態特徵選擇 ---
    # 假設 CSV 欄位格式固定為 "{Type}_{AP_Index}" (例如 RSSI_1, Dist_mm_1)
    # 根據您的 CSV 範例：RSSI_1, Dist_mm_1, Std_mm_1
    selected_cols = []
    for i in range(1, 5): # AP1 ~ AP4
        for f_type in feature_types:
            col_name = f"{f_type}_{i}"
            selected_cols.append(col_name)
    
    # 檢查欄位是否存在
    missing_cols = [c for c in selected_cols if c not in df.columns]
    if missing_cols:
        print(f"警告: CSV 中缺少以下欄位: {missing_cols}")
        return None

    print(f"使用特徵: {selected_cols}")
    
    # 轉數值
    for col in selected_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- 2. 缺值處理 (依 Label 分組補平均) ---
    df[selected_cols] = df.groupby('Label')[selected_cols].transform(lambda x: x.fillna(x.mean()))
    df[selected_cols] = df[selected_cols].fillna(0) # 防呆

    X = df[selected_cols].values
    y_raw = df['Label'].values

    # --- 3. Label Encoding ---
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    
    # --- 4. 資料切分 (Train: 70%, Val: 15%, Test: 15%) ---
    # 第一刀：切出 Test (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=random_state, stratify=y
    )
    # 第二刀：剩下的資料切出 Val (0.15 / 0.85 ≈ 0.176)
    val_size = 0.15 / (1 - 0.15)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )

    # --- 5. 標準化 (嚴謹模式：只 Fit Train) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # --- 6. 轉 Tensor ---
    # 封裝成 Dataset
    train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.LongTensor(y_test))

    # 封裝成 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    meta_data = {
        "input_dim": X.shape[1],
        "num_classes": len(label_encoder.classes_),
        "label_encoder": label_encoder,
        "scaler": scaler # 回傳 scaler 以便將來有新資料時使用
    }
    
    return train_loader, val_loader, test_loader, meta_data