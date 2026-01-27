import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

def apply_calibration(df, config):
    """
    根據 config 對 dataframe 進行 RSSI 與 Dist_mm 的校正。
    """
    print("--- [Calibration] 正在執行資料校正 ---")
    df_cal = df.copy()
    
    # 遍歷每一個 AP 設定 (Pos1 ~ Pos4)
    for pos_name, params in config.items():
        bssid = params['target_bssid']
        rssi_off = params['rssi_offset']
        dist_off = params['dist_offset']
        
        # 掃描 _1 到 _4 的欄位，因為 AP 可能出現在任意位置
        for i in range(1, 5):
            suffix = f'_{i}'
            bssid_col = f'BSSID{suffix}'
            rssi_col = f'RSSI{suffix}'
            dist_col = f'Dist_mm{suffix}'
            
            if bssid_col not in df_cal.columns: continue
            
            # 找出符合該 AP BSSID 的 Rows
            mask = df_cal[bssid_col] == bssid
            
            if mask.any():
                # 1. 校正 RSSI
                if rssi_col in df_cal.columns:
                    df_cal.loc[mask, rssi_col] += rssi_off
                
                # 2. 校正 Dist_mm
                if dist_col in df_cal.columns:
                    df_cal.loc[mask, dist_col] += dist_off
                    
                    # 3. 負值處理 (Clipping): 距離不可能小於 0
                    neg_mask = (df_cal[dist_col] < 0) & mask
                    if neg_mask.any():
                        df_cal.loc[neg_mask, dist_col] = 0

    return df_cal

# === 1. 定義 CVAE 模型 (維持不變) ===
class CVAE(nn.Module):
    def __init__(self, input_dim, cond_dim, latent_dim=8):
        super(CVAE, self).__init__()
        # --- Encoder ---
        self.encoder_fc1 = nn.Linear(input_dim + cond_dim, 256)
        self.encoder_fc2 = nn.Linear(256, 128)
        self.z_mean = nn.Linear(128, latent_dim)
        self.z_log_var = nn.Linear(128, latent_dim)
        # --- Decoder ---
        self.decoder_fc1 = nn.Linear(latent_dim + cond_dim, 128)
        self.decoder_fc2 = nn.Linear(128, 256)
        self.decoder_out = nn.Linear(256, input_dim) 
        self.relu = nn.ReLU()

    def encode(self, x, c):
        inputs = torch.cat([x, c], dim=1)
        h = self.relu(self.encoder_fc1(inputs))
        h = self.relu(self.encoder_fc2(h))
        return self.z_mean(h), self.z_log_var(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        inputs = torch.cat([z, c], dim=1)
        h = self.relu(self.decoder_fc1(inputs))
        h = self.relu(self.decoder_fc2(h))
        return self.decoder_out(h)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar

def loss_function_cvae(recon_x, x, mu, logvar, kl_weight=0.1):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD * kl_weight, MSE, KLD

# === 2. [新功能] 單一 CVAE 訓練核心 ===
def train_single_cvae(train_file_path, target_name, epochs=100, batch_size=64, device='cpu'):
    """
    讀取指定的訓練檔案，訓練一個專屬的 CVAE 模型。
    假設訓練檔案中有 'RSSI' 和 'Dist_mm' 欄位。
    """
    print(f"\n--- [CVAE Training] Target: {target_name} | Source: {train_file_path} ---")
    
    if not os.path.exists(train_file_path):
        print(f"Error: 訓練檔案不存在 {train_file_path}，跳過此模型。")
        return None, None, None

    # 讀取訓練資料
    df_train = pd.read_csv(train_file_path)
    
    # 欄位檢查與重新命名 (這裡假設訓練檔是標準格式，若有不同需在此調整)
    # 我們需要標準化欄位名為 'Rssi' 和 'Distance' 以便訓練
    req_cols = ['RSSI', 'Dist_mm'] 
    
    # 相容性處理：有些檔案可能叫 RSSI_1, Dist_mm_1，這裡做簡單模糊搜尋
    if 'RSSI' not in df_train.columns:
        # 嘗試找 RSSI_1 或類似的
        candidates = [c for c in df_train.columns if 'RSSI' in c]
        if candidates: df_train.rename(columns={candidates[0]: 'RSSI'}, inplace=True)
    
    if 'Dist_mm' not in df_train.columns:
        candidates = [c for c in df_train.columns if 'Dist_mm' in c]
        if candidates: df_train.rename(columns={candidates[0]: 'Dist_mm'}, inplace=True)

    if 'RSSI' not in df_train.columns or 'Dist_mm' not in df_train.columns:
        print(f"Error: 訓練檔案欄位不足 (需 RSSI, Dist_mm)，跳過。")
        return None, None, None

    # 準備資料
    data = df_train[['RSSI', 'Dist_mm']].dropna()
    cond_train = data[['RSSI']].values
    target_train = data[['Dist_mm']].values

    # 獨立的 Scaler (這很重要，因為每個檔案的分佈可能不同)
    scaler_cond = StandardScaler()
    scaler_target = StandardScaler()
    
    cond_scaled = scaler_cond.fit_transform(cond_train)
    target_scaled = scaler_target.fit_transform(target_train)

    # 轉 Tensor
    cond_tensor = torch.FloatTensor(cond_scaled).to(device)
    target_tensor = torch.FloatTensor(target_scaled).to(device)
    
    dataset = TensorDataset(target_tensor, cond_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    model = CVAE(input_dim=1, cond_dim=1, latent_dim=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 訓練迴圈
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for x, c in loader:
            optimizer.zero_grad()
            recon, mu, logvar = model(x, c)
            loss, _, _ = loss_function_cvae(recon, x, mu, logvar, kl_weight=0.1)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch+1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(loader.dataset):.4f}")

    return model, scaler_cond, scaler_target

# === 3. [修改] CVAE 管理與生成管線 ===
def manager_cvae_pipeline(df_main, cvae_config):
    """
    df_main: 主要的 Full Set 資料集 (待補全)
    cvae_config: 設定檔，格式如下
        {
            'Dist_Pred_2': {'train_file': './teacher_pos2.csv', 'rssi_col': 'RSSI_2'},
            'Dist_Pred_3': {'train_file': './teacher_pos3.csv', 'rssi_col': 'RSSI_3'},
            ...
        }
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化預測欄位 (如果不存在)
    for target_col in cvae_config.keys():
        if target_col not in df_main.columns:
            df_main[target_col] = 0.0

    # 針對設定檔中的每一個目標進行 loop
    for target_col, config in cvae_config.items():
        train_file = config['train_file']
        rssi_col_in_main = config['rssi_col']
        
        # 1. 訓練專屬模型
        model, scaler_c, scaler_t = train_single_cvae(train_file, target_col, device=device)
        
        if model is None:
            continue # 訓練失敗或檔案沒找到
            
        # 2. 推論 (Generation)
        print(f"  -> Generating {target_col} using RSSI source: {rssi_col_in_main}...")
        model.eval()
        
        if rssi_col_in_main not in df_main.columns:
            print(f"  Warning: 主資料集缺少 {rssi_col_in_main}，無法生成 {target_col}")
            continue

        # 找出需要生成的 Rows (有 RSSI 但沒距離，或者全部重算，這裡假設全部重算)
        mask = df_main[rssi_col_in_main].notna()
        if not mask.any():
            continue

        rssi_values = df_main.loc[mask, [rssi_col_in_main]].values
        
        # 使用該模型的 Scaler 轉換
        rssi_scaled = scaler_c.transform(rssi_values)
        c_tensor = torch.FloatTensor(rssi_scaled).to(device)
        
        # Latent Space Sampling
        z = torch.randn(c_tensor.size(0), 8).to(device) # latent_dim=8
        
        with torch.no_grad():
            pred_scaled = model.decode(z, c_tensor)
            pred_real = scaler_t.inverse_transform(pred_scaled.cpu().numpy())
        
        # 填回資料表
        df_main.loc[mask, target_col] = pred_real.flatten()
        
        # 簡單驗證
        print(f"  -> {target_col} Generated. Mean: {np.mean(pred_real):.2f}, Std: {np.std(pred_real):.2f}")

    return df_main

# === 4. [修改] 資料載入與整合 ===
def get_data_loaders(main_csv_path, selected_features, cvae_config, batch_size=32, random_state=42, samples_per_label=None):
    """
    參數:
    - selected_features: list, 例如 ['Dist_mm_1', 'RSSI_1', 'Dist_Pred_2', 'RSSI_2']
    - cvae_config: dict, CVAE 的訓練設定
    """
    try:
        df = pd.read_csv(main_csv_path)
    except FileNotFoundError:
        print(f"錯誤: 找不到檔案 {main_csv_path}")
        return None
    
    """
    新增參數:
    - samples_per_label (int): 每個 RP (Label) 要隨機抽取的筆數。若為 None 則使用全部資料。
    """
    try:
        df = pd.read_csv(main_csv_path)
    except FileNotFoundError:
        print(f"錯誤: 找不到檔案 {main_csv_path}")
        return None

    # === [新功能] 動態控制資料量 ===
    if samples_per_label is not None:
        print(f"\n[Data Sampling] 限制每個 RP 使用 {samples_per_label} 筆資料...")
        
        # 檢查是否有 Label 資料過少
        min_count = df['Label'].value_counts().min()
        if min_count < samples_per_label:
            print(f"警告: 部分 Label 資料不足 {samples_per_label} 筆 (最少僅 {min_count})，將取最大可用量。")

        # 使用 groupby + sample 進行分層隨機抽樣
        # random_state 確保每次實驗內部的抽樣結果是可重現的 (或隨 run_id 變化)
        try:
            df = df.groupby('Label').sample(n=samples_per_label, random_state=random_state, replace=False)
        except ValueError:
            # 如果某個 Label 樣本數不夠，sample 會報錯，這裡改用 lambda 處理
            df = df.groupby('Label').apply(lambda x: x.sample(n=min(len(x), samples_per_label), random_state=random_state)).reset_index(drop=True)
            
        print(f"Sampling 完成，目前總資料筆數: {len(df)}")

    # 1. 執行 CVAE 生成流程
    # 這會將預測的距離填入 df 的對應欄位 (如 Dist_Pred_2)
    df = manager_cvae_pipeline(df, cvae_config)

    # 2. 動態特徵選擇
    print(f"\n--- Feature Selection ---")
    print(f"Selected Features: {selected_features}")
    
    # 檢查欄位是否存在，不存在補 0
    for col in selected_features:
        if col not in df.columns: 
            print(f"Warning: {col} not in dataset, filling with 0.")
            df[col] = 0.0
        else: 
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 補值策略 (針對 Label 分組補平均，若無則補 0)
    if 'Label' in df.columns:
        df[selected_features] = df.groupby('Label')[selected_features].transform(lambda x: x.fillna(x.mean()))
    df[selected_features] = df[selected_features].fillna(0)

    X = df[selected_features].values
    y_raw = df['Label'].values

    # 3. Label Encode
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # 4. Random Split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=random_state, stratify=y
    )
    val_size = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )

    # 5. 標準化 (針對輸入特徵)
    scaler_dnn = StandardScaler()
    X_train_scaled = scaler_dnn.fit_transform(X_train)
    X_val_scaled = scaler_dnn.transform(X_val)
    X_test_scaled = scaler_dnn.transform(X_test)
    
    # 6. DataLoader
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val_scaled), torch.LongTensor(y_val)), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test_scaled), torch.LongTensor(y_test)), batch_size=batch_size, shuffle=False)

    meta = {
        "input_dim": X.shape[1],
        "num_classes": len(label_encoder.classes_),
        "label_encoder": label_encoder,
        "feature_names": selected_features # 保存特徵名稱以供後續分析
    }
    
    return train_loader, val_loader, test_loader, meta