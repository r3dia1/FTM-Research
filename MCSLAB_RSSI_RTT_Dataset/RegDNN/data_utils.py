import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import SGDRegressor
import re

def train_augment_regressor(df, feature_cols):
    """
    RegDNN 核心邏輯:
    此時傳入的 df 已經完成補值 (Imputed)，所以不需要擔心 NaN 問題。
    1. 使用資料集中「所有」可用的 (RSSI, Dist_mm) 配對來訓練 Regressor。
    2. 根據 feature_cols 的要求，只針對需要的 Dist_Pred_X 進行預測。
    """
    print("--- [RegDNN] 啟動線性回歸訓練與指定特徵擴充 ---")

    # === 1. 準備 Regressor 訓練資料 (利用所有可用 AP) ===
    # 找出所有同時擁有 RSSI 和 Dist_mm 的 AP ID
    rssi_cols = [c for c in df.columns if c.startswith('RSSI_')]
    potential_ids = [c.replace('RSSI_', '') for c in rssi_cols]
    # print(rssi_cols)
    # print(potential_ids)
    
    train_dfs = []
    for ap_id in potential_ids:
        dist_col = f'Dist_mm_{ap_id}'
        rssi_col = f'RSSI_{ap_id}'
        
        # 雖然已經補值，但檢查一下欄位是否存在總是安全的
        if dist_col in feature_cols:
            # 這裡理論上已經沒有 NaN，但為了程式健壯性保留 dropna，或者可以直接取值
            print(dist_col)
            temp_df = df[[rssi_col, dist_col]].rename(columns={rssi_col: 'Rssi', dist_col: 'Distance'})
            train_dfs.append(temp_df)

        # if dist_col in df.columns:
        #     # 這裡理論上已經沒有 NaN，但為了程式健壯性保留 dropna，或者可以直接取值
        #     temp_df = df[[rssi_col, dist_col]].rename(columns={rssi_col: 'Rssi', dist_col: 'Distance'})
        #     train_dfs.append(temp_df)
    
    if not train_dfs:
        print("警告: 找不到任何成對的 (RSSI, Dist_mm) 資料，跳過訓練。")
        return df, None

    train_data_reg = pd.concat(train_dfs, ignore_index=True)
    
    # 檢查樣本數
    if len(train_data_reg) < 10:
        print("警告: 回歸訓練樣本太少，跳過。")
        return df, None

    # 訓練模型
    X_train_reg = train_data_reg[['Rssi']].values
    y_train_reg = train_data_reg['Distance'].values
    
    scaler_reg = StandardScaler()
    X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)

    model_reg = SGDRegressor(
        loss='huber', penalty='l2', alpha=0.0001, 
        learning_rate='optimal', eta0=0.1, max_iter=5000, tol=1e-4, random_state=42
    )
    model_reg.fit(X_train_reg_scaled, y_train_reg)
    print("Regressor 訓練完成。")

    # === 2. 根據 feature_cols 執行預測 ===
    # 解析 feature_cols，找出需要產生預測的欄位 (Dist_Pred_X)
    pred_targets = [c for c in feature_cols if c.startswith('Dist_Pred_')]
    
    count_predicted = 0
    for pred_col in pred_targets:
        # 取得 ID, 例如 "Dist_Pred_2" -> "2"
        ap_id = pred_col.replace('Dist_Pred_', '')
        rssi_col = f'RSSI_{ap_id}'
        
        # 檢查是否有對應的 RSSI 欄位可供預測
        if rssi_col not in df.columns:
            print(f"警告: 需要 {pred_col} 但找不到對應的 {rssi_col}，填 0。")
            df[pred_col] = 0.0
            continue

        # 預測邏輯 (此時 RSSI 已經補完值，是完整的)
        df[pred_col] = 0.0 
        
        # 直接拿整欄預測，不需要 mask 判斷是否為 NaN (因為已經補過值了)
        rssi_val = df[[rssi_col]].values
        rssi_val_scaled = scaler_reg.transform(rssi_val)
        df[pred_col] = model_reg.predict(rssi_val_scaled)
        count_predicted += 1
            
    print(f"已針對 feature_cols 要求，補充了 {count_predicted} 個 AP 的預測距離。")

    return df, model_reg

def get_data_loaders(csv_path, batch_size=32, random_state=42, loop_id=0, samples_per_label=None):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"錯誤: 找不到檔案 {csv_path}")
        return None

    # === 1. [Data Sampling] ===
    if samples_per_label is not None:
        print(f"\n[Data Sampling] 限制每個 RP 使用 {samples_per_label} 筆資料...")
        min_count = df['Label'].value_counts().min()
        if min_count < samples_per_label:
            print(f"警告: 部分 Label 資料不足 {samples_per_label}，將取最大可用量。")

        df = df.groupby('Label').apply(
            lambda x: x.sample(n=min(len(x), samples_per_label), random_state=random_state)
        ).reset_index(drop=True)

    # === 2. [Imputation (優先執行)] ===
    # 在預測之前，先針對資料集中「所有數值欄位」進行補值
    # 確保 RSSI, Dist_mm, Std_mm 等都有值
    print("正在執行 Label Group Mean Imputation (全面補值)...")
    
    # 篩選數值型欄位進行補值 (排除 Label 字串等)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # 使用 GroupBy Label 的平均值填補
    df[numeric_cols] = df.groupby('Label')[numeric_cols].transform(lambda x: x.fillna(x.mean()))
    
    # 如果該 Label 整組都是 NaN，則用 0 填補 (Fallback)
    df[numeric_cols] = df[numeric_cols].fillna(0)


    # === 3. [Feature Selection (手動指定)] ===
    feature_cols = [
        'Dist_mm_1', 
        'Dist_mm_2',
        # 'Dist_mm_3', 
        'Dist_mm_4',
        'Std_mm_1',  
        'Std_mm_2',
        # 'Std_mm_3', 
        'Std_mm_4',
        'RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4',
        # 'RSSI_2', 'RSSI_3', 'RSSI_4',
        # 'Dist_Pred_1', 'Dist_Pred_2', 'Dist_Pred_4',
        # 'Dist_Pred_3'
    ]
    
    # === 4. [RegDNN 擴充] ===
    # 此時傳入的 df 已經很乾淨，Regressor 可以學到更多，預測也更完整
    df, _ = train_augment_regressor(df, feature_cols)

    # 確保特徵欄位存在 (防呆)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 準備 X, y
    X = df[feature_cols].values
    # print(X)
    y_raw = df['Label'].values

    # === 5. [Label Encode & Split] ===
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    print(f"正在執行 Random Split (seed={random_state})...")
    try:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=random_state, stratify=y
        )
        val_size = 0.15 / 0.85
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
        )
    except ValueError:
        print("Split Error (樣本數過少): 切換為無 Stratify 分割...")
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=random_state
        )
        val_size = 0.15 / 0.85
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state
        )

    print(f"切分結果: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # === 6. [Scale & Loaders] ===
    scaler_dnn = StandardScaler()
    X_train_scaled = scaler_dnn.fit_transform(X_train)
    X_val_scaled = scaler_dnn.transform(X_val)
    X_test_scaled = scaler_dnn.transform(X_test)
    
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val_scaled), torch.LongTensor(y_val)), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test_scaled), torch.LongTensor(y_test)), batch_size=batch_size, shuffle=False)

    meta = {
        "input_dim": X.shape[1],
        "num_classes": len(label_encoder.classes_),
        "label_encoder": label_encoder,
        "feature_names": feature_cols
    }
    
    return train_loader, val_loader, test_loader, meta







# import pandas as pd
# import numpy as np
# import torch
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.linear_model import SGDRegressor
# import joblib

# def train_augment_regressor(df):
#     """
#     RegDNN 核心邏輯: 訓練 Regressor 並擴充欄位 (Dist_Pred_2, Dist_Pred_3)
#     """
#     print("--- [RegDNN] 啟動線性回歸訓練與特徵擴充 ---")
    
#     # 1. 準備訓練資料
#     cols_check = ['RSSI_1', 'Dist_mm_1', 'RSSI_4', 'Dist_mm_4']
#     if not all(col in df.columns for col in cols_check):
#         print(f"警告: 找不到 Regressor 需要的欄位 {cols_check}，跳過擴充。")
#         return df, None

#     ap1_data = df[['RSSI_1', 'Dist_mm_1']].dropna().rename(columns={'RSSI_1': 'Rssi', 'Dist_mm_1': 'Distance'})
#     ap4_data = df[['RSSI_4', 'Dist_mm_4']].dropna().rename(columns={'RSSI_4': 'Rssi', 'Dist_mm_4': 'Distance'})
    
#     train_data_reg = pd.concat([ap1_data, ap4_data], ignore_index=True)
    
#     # 檢查是否有足夠資料進行訓練
#     if len(train_data_reg) < 10:
#         print("警告: 可用於回歸訓練的樣本數過少，跳過訓練。")
#         return df, None

#     X_train_reg = train_data_reg[['Rssi']].values
#     y_train_reg = train_data_reg['Distance'].values

#     # 2. 標準化 (只對 RSSI)
#     scaler_reg = StandardScaler()
#     X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)

#     # 3. 訓練 Regressor
#     model_reg = SGDRegressor(
#         loss='huber', penalty='l2', alpha=0.0001, 
#         learning_rate='optimal', eta0=0.1, max_iter=5000, tol=1e-4, random_state=42
#     )
#     model_reg.fit(X_train_reg_scaled, y_train_reg)
    
#     # 4. 擴充預測
#     df['Dist_Pred_2'] = 0.0
#     df['Dist_Pred_3'] = 0.0
    
#     if 'RSSI_2' in df.columns:
#         mask = df['RSSI_2'].notna()
#         if mask.any():
#             rssi_scaled = scaler_reg.transform(df.loc[mask, ['RSSI_2']].values)
#             df.loc[mask, 'Dist_Pred_2'] = model_reg.predict(rssi_scaled)
            
#     if 'RSSI_3' in df.columns:
#         mask = df['RSSI_3'].notna()
#         if mask.any():
#             rssi_scaled = scaler_reg.transform(df.loc[mask, ['RSSI_3']].values)
#             df.loc[mask, 'Dist_Pred_3'] = model_reg.predict(rssi_scaled)

#     return df, model_reg

# def get_data_loaders(csv_path, batch_size=32, random_state=42, loop_id=0, samples_per_label=None):
#     """
#     修改說明: 新增 samples_per_label 參數來動態控制每個 RP 的資料量
#     """
#     try:
#         df = pd.read_csv(csv_path)
#     except FileNotFoundError:
#         print(f"錯誤: 找不到檔案 {csv_path}")
#         return None

#     # === [新功能] 動態控制資料量 (Sampling) ===
#     # 邏輯: 在進入 Regressor 訓練前就先篩選資料，確保模擬的是「資料量少」的情境
#     if samples_per_label is not None:
#         print(f"\n[Data Sampling] 限制每個 RP 使用 {samples_per_label} 筆資料...")
        
#         # 檢查資料是否足夠
#         min_count = df['Label'].value_counts().min()
#         if min_count < samples_per_label:
#             print(f"警告: 部分 Label 資料不足 {samples_per_label} 筆 (最少僅 {min_count})，將取最大可用量。")

#         # 分層抽樣
#         # 使用 lambda 處理不同 Label 數量可能不一致的問題，避免報錯
#         df = df.groupby('Label').apply(
#             lambda x: x.sample(n=min(len(x), samples_per_label), random_state=random_state)
#         ).reset_index(drop=True)
            
#         print(f"Sampling 完成，目前總資料筆數: {len(df)}")
#     # ==========================================

#     # 1. RegDNN 擴充 (現在這一步使用的是 Sampling 後的資料)
#     df, _ = train_augment_regressor(df)

#     # 2. 選擇特徵
#     feature_cols = [
#         # 'Dist_mm_3', 
#         'Dist_mm_2','Dist_mm_1', 'Dist_mm_4'
#         # 'Std_mm_3',  
#         'Std_mm_2','Std_mm_1',  'Std_mm_4',
#         'RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4',
#         # 'Dist_Pred_1', 'Dist_Pred_2', 'Dist_Pred_4'
#         # 'Dist_Pred_3'
#     ]
    
#     # 確保所有特徵欄位存在
#     for col in feature_cols:
#         if col not in df.columns: 
#             df[col] = 0.0
#         else: 
#             df[col] = pd.to_numeric(df[col], errors='coerce')
    
#     # 補值
#     df[feature_cols] = df.groupby('Label')[feature_cols].transform(lambda x: x.fillna(x.mean()))
#     df[feature_cols] = df[feature_cols].fillna(0)

#     X = df[feature_cols].values
#     y_raw = df['Label'].values

#     # 3. Label Encode
#     label_encoder = LabelEncoder()
#     y = label_encoder.fit_transform(y_raw)

#     # 4. Random Split
#     print(f"正在執行 Random Split (seed={random_state})...")
    
#     # 確保有足夠樣本進行 Split (若 sampling 數量太極端，例如每類只取 1 筆，stratify 會報錯)
#     try:
#         # 第一刀：切出 Test (15%)
#         X_temp, X_test, y_temp, y_test = train_test_split(
#             X, y, test_size=0.15, random_state=random_state, stratify=y
#         )
        
#         # 第二刀：從剩下的 85% 中切出 Val
#         val_size = 0.15 / 0.85
#         X_train, X_val, y_train, y_val = train_test_split(
#             X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
#         )
#     except ValueError as e:
#         print(f"Split Error (可能因樣本數過少導致): {e}")
#         print("切換為不使用 Stratify 的分割方式...")
#         X_temp, X_test, y_temp, y_test = train_test_split(
#             X, y, test_size=0.15, random_state=random_state
#         )
#         val_size = 0.15 / 0.85
#         X_train, X_val, y_train, y_val = train_test_split(
#             X_temp, y_temp, test_size=val_size, random_state=random_state
#         )

#     print(f"切分結果: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

#     # 5. 標準化
#     scaler_dnn = StandardScaler()
#     X_train_scaled = scaler_dnn.fit_transform(X_train)
#     X_val_scaled = scaler_dnn.transform(X_val)
#     X_test_scaled = scaler_dnn.transform(X_test)
    
#     # 6. DataLoader
#     train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train)), batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val_scaled), torch.LongTensor(y_val)), batch_size=batch_size, shuffle=False)
#     test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test_scaled), torch.LongTensor(y_test)), batch_size=batch_size, shuffle=False)

#     meta = {
#         "input_dim": X.shape[1],
#         "num_classes": len(label_encoder.classes_),
#         "label_encoder": label_encoder
#     }
    
#     return train_loader, val_loader, test_loader, meta