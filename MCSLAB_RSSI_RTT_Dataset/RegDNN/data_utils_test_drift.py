import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import SGDRegressor

def train_augment_regressor(df, pretrained_model=None, pretrained_scaler=None):
    """
    RegDNN 核心邏輯 (修正版): 
    - 參考 Code2 邏輯，動態尋找所有 (RSSI, Dist_mm) 配對進行訓練。
    - 針對所有存在的 RSSI 欄位生成 Dist_Pred。
    - 若無傳入 pretrained，則訓練新的 Regressor (Training Mode)。
    - 若有傳入 pretrained，則直接使用舊參數預測 (Inference Mode)。
    """
    
    # 找出所有 RSSI 開頭的欄位，例如 ['RSSI_1', 'RSSI_2', ...]
    rssi_cols = [c for c in df.columns if c.startswith('RSSI_')]
    
    # 如果找不到任何 RSSI 欄位，直接回傳
    if not rssi_cols:
        return df, pretrained_model, pretrained_scaler

    model_reg = pretrained_model
    scaler_reg = pretrained_scaler

    # === Training Mode (若沒有傳入模型，則進行訓練) ===
    if model_reg is None or scaler_reg is None:
        # print("--- [RegDNN] 啟動動態回歸訓練 ---")
        train_dfs = []
        
        # 1. 動態蒐集所有可用的 (RSSI, Dist) 配對
        for rssi_col in rssi_cols:
            # 解析 ID, e.g. 'RSSI_1' -> '1'
            ap_id = rssi_col.replace('RSSI_', '')
            dist_col = f'Dist_mm_{ap_id}'
            
            if dist_col in df.columns:
                # Code1 的資料尚未補值，所以必須 dropna 確保訓練資料乾淨
                temp_df = df[[rssi_col, dist_col]].dropna().rename(columns={rssi_col: 'Rssi', dist_col: 'Distance'})
                train_dfs.append(temp_df)
        
        # 如果完全沒有配對資料
        if not train_dfs:
            return df, None, None

        train_data_reg = pd.concat(train_dfs, ignore_index=True)
        
        # 檢查樣本數
        if len(train_data_reg) < 10:
            return df, None, None

        # 2. 訓練模型
        X_train_reg = train_data_reg[['Rssi']].values
        y_train_reg = train_data_reg['Distance'].values

        scaler_reg = StandardScaler()
        X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)

        model_reg = SGDRegressor(loss='huber', penalty='l2', max_iter=5000, tol=1e-4, random_state=42)
        model_reg.fit(X_train_reg_scaled, y_train_reg)

    # === Inference / Augmentation (對所有 RSSI 生成預測) ===
    # 防呆：確保 scaler_reg 存在才做轉換
    if scaler_reg is not None:
        for rssi_col in rssi_cols:
            ap_id = rssi_col.replace('RSSI_', '')
            pred_col = f'Dist_Pred_{ap_id}'
            
            # 初始化預測欄位
            df[pred_col] = 0.0
            
            # Code1 的邏輯：只對非 NaN 的 RSSI 進行預測
            mask = df[rssi_col].notna()
            if mask.any():
                rssi_vals = df.loc[mask, [rssi_col]].values
                rssi_scaled = scaler_reg.transform(rssi_vals)
                df.loc[mask, pred_col] = model_reg.predict(rssi_scaled)

    return df, model_reg, scaler_reg

# === 輔助函式：特徵選擇與補值 ===
def process_features(df):
    feature_cols = [
        # 'Dist_mm_3', 'Dist_mm_2', 'Dist_mm_1', 'Dist_mm_4',
        # 'Std_mm_3', 'Std_mm_2', 'Std_mm_1', 'Std_mm_4',
        'RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4',
        # 'Dist_Pred_2', 'Dist_Pred_3' # 確保包含擴充欄位
    ]
    
    for col in feature_cols:
        if col not in df.columns: 
            df[col] = 0.0
        else: 
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 補值策略：若有 Label 則用 Group Mean，否則填 0
    if 'Label' in df.columns:
        df[feature_cols] = df.groupby('Label')[feature_cols].transform(lambda x: x.fillna(x.mean()))
    
    df[feature_cols] = df[feature_cols].fillna(0)
    return df[feature_cols].values

# === 主函式 1: 訓練用 DataLoader (包含 Split) ===
def get_data_loaders(csv_path, batch_size=32, random_state=42, loop_id=0, samples_per_label=None):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"錯誤: 找不到檔案 {csv_path}")
        return None

    # 1. Sampling (模擬少資料情境)
    if samples_per_label is not None:
        print(f"[Data Sampling] 每個 RP 取 {samples_per_label} 筆")
        df = df.groupby('Label').apply(
            lambda x: x.sample(n=min(len(x), samples_per_label), random_state=random_state)
        ).reset_index(drop=True)

    # 2. RegDNN 擴充 (Training Mode -> 產出 model_reg, scaler_reg)
    df, model_reg, scaler_reg = train_augment_regressor(df)

    # 3. 處理特徵與 Label
    X = process_features(df)
    y_raw = df['Label'].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # 4. Split (Train / Val / Test)
    # 這裡保留你原本的邏輯：切出 Test(15%) -> 再切 Val(15%)
    try:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=random_state, stratify=y
        )
        val_size = 0.15 / 0.85 
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
        )
    except ValueError:
        print("警告: 樣本過少無法 Stratify，改為隨機切分")
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=random_state)
        val_size = 0.15 / 0.85
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, random_state=random_state)

    print(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # 5. 標準化 (關鍵步驟)
    scaler_dnn = StandardScaler()
    X_train_scaled = scaler_dnn.fit_transform(X_train) # 只有 Train 能 Fit
    X_val_scaled = scaler_dnn.transform(X_val)         # Val 用 Train 的尺
    X_test_scaled = scaler_dnn.transform(X_test)       # Test 用 Train 的尺
    
    # 6. 包裝 Loader
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val_scaled), torch.LongTensor(y_val)), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test_scaled), torch.LongTensor(y_test)), batch_size=batch_size, shuffle=False)

    # 7. 打包所有工具 (Meta)
    meta = {
        "input_dim": X.shape[1],
        "num_classes": len(label_encoder.classes_),
        "label_encoder": label_encoder,
        "scaler_dnn": scaler_dnn,       # 這是 DNN 用的主要 Scaler
        "model_reg": model_reg,         # 這是特徵工程用的回歸模型
        "scaler_reg": scaler_reg        # 這是回歸模型用的 Scaler
    }
    
    return train_loader, val_loader, test_loader, meta

# === 主函式 2: 新資料測試用 Loader (不訓練，只轉換) ===
def get_new_data_loader(csv_path, meta, batch_size=32):
    """
    讀取新資料，並「嚴格」使用 meta 中的參數進行轉換
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"錯誤: 找不到檔案 {csv_path}")
        return None

    # 1. 使用舊的 Regressor 擴充特徵 (Inference Mode)
    df, _, _ = train_augment_regressor(
        df, 
        pretrained_model=meta['model_reg'], 
        pretrained_scaler=meta['scaler_reg']
    )

    # 2. 處理特徵
    X = process_features(df)
    y_raw = df['Label'].values

    # 3. Label Encoding (使用舊 Encoder)
    encoder = meta['label_encoder']
    
    # 過濾未知 Label (避免報錯)
    known_labels = set(encoder.classes_)
    mask = [l in known_labels for l in y_raw]
    
    if not all(mask):
        unknown_count = len(y_raw) - sum(mask)
        print(f"警告: 過濾掉 {unknown_count} 筆訓練集中沒出現過的 Label")
        X = X[mask]
        y_raw = y_raw[mask]

    if len(X) == 0:
        return None

    y = encoder.transform(y_raw)

    # 4. 標準化 (使用舊 Scaler -> Transform Only)
    scaler_dnn = meta['scaler_dnn']
    X_scaled = scaler_dnn.transform(X)

    # 5. 回傳 Loader
    loader = DataLoader(TensorDataset(torch.FloatTensor(X_scaled), torch.LongTensor(y)), batch_size=batch_size, shuffle=False)
    
    return loader