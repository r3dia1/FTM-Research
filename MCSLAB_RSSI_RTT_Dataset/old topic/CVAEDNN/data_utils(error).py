import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import json
import os

# === 1. CVAE 模型 (架構微調) ===
class CVAE(nn.Module):
    def __init__(self, input_dim, cond_dim, latent_dim=8):
        super(CVAE, self).__init__()
        # Cond_dim 現在會是 2 (x, y)，Input_dim 是 1 (RTT)
        self.encoder_fc1 = nn.Linear(input_dim + cond_dim, 256)
        self.encoder_fc2 = nn.Linear(256, 128)
        self.z_mean = nn.Linear(128, latent_dim)
        self.z_log_var = nn.Linear(128, latent_dim)
        
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

# === 2. 訓練單一 CVAE (Condition on Location) ===
def train_single_cvae(train_file_path, target_name, coords_map, epochs=200, batch_size=64, device='cpu'):
    """
    Input: Label (converted to x,y)
    Output: Teacher RTT (Dist_mm)
    """
    print(f"\n--- [CVAE Training] Target: {target_name} (Cond: Location) | Source: {train_file_path} ---")
    
    if not os.path.exists(train_file_path):
        print(f"Error: 訓練檔案不存在 {train_file_path}")
        return None, None, None

    df_train = pd.read_csv(train_file_path)
    
    # 確保有 Label 和 Dist_mm
    # Teacher 檔案通常有 'Label' 欄位，如果沒有，請在 main.py 確認檔案格式
    if 'Label' not in df_train.columns:
        print("Error: 訓練檔缺少 'Label' 欄位，無法進行位置條件訓練。")
        return None, None, None

    # 處理 Dist_mm 欄位名稱 (相容性)
    if 'Dist_mm' not in df_train.columns:
        candidates = [c for c in df_train.columns if 'Dist_mm' in c]
        if candidates: df_train.rename(columns={candidates[0]: 'Dist_mm'}, inplace=True)
    
    if 'Dist_mm' not in df_train.columns:
        print("Error: 訓練檔缺少 'Dist_mm' 欄位。")
        return None, None, None

    # 準備資料
    # Condition: (x, y) from Label
    # Target: Dist_mm
    
    # 轉換 Label 為座標 (x, y)
    def get_coords(label):
        return coords_map.get(str(label), (0.0, 0.0))
    
    coords_data = np.array([get_coords(l) for l in df_train['Label'].values])
    dist_data = df_train[['Dist_mm']].values

    # 移除 NaN
    mask = ~np.isnan(dist_data).flatten()
    cond_train = coords_data[mask]
    target_train = dist_data[mask]

    # Scaler
    scaler_cond = StandardScaler() # Scale (x, y)
    scaler_target = StandardScaler() # Scale Dist
    
    cond_scaled = scaler_cond.fit_transform(cond_train)
    target_scaled = scaler_target.fit_transform(target_train)

    # Tensor
    cond_tensor = torch.FloatTensor(cond_scaled).to(device)
    target_tensor = torch.FloatTensor(target_scaled).to(device) # x
    
    dataset = TensorDataset(target_tensor, cond_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型: Input=1 (Dist), Cond=2 (X,Y)
    model = CVAE(input_dim=1, cond_dim=2, latent_dim=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
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

# === 3. CVAE 生成管線 (使用 Label 生成) ===
def manager_cvae_pipeline(df_main, cvae_config, coords_map):
    """
    df_main: 包含 Student Data (必須有 'Label' 欄位用來做 Training 階段的生成)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if 'Label' not in df_main.columns:
        print("Error: 主資料集缺少 'Label'，無法進行 CVAE 生成 (需座標條件)。")
        return df_main

    # 預先轉換 df_main 的 Label 為座標 (快取)
    print("  -> Preparing coordinates from Labels...")
    main_coords = np.array([coords_map.get(str(l), (0.0, 0.0)) for l in df_main['Label'].values])

    # 初始化欄位
    for target_col in cvae_config.keys():
        if target_col not in df_main.columns:
            df_main[target_col] = 0.0

    for target_col, config in cvae_config.items():
        train_file = config['train_file']
        # 注意：這裡不再需要 rssi_col，因為我們是用 Location 生成
        
        # 1. 訓練模型 (Cond: Label -> Target: Teacher Dist)
        model, scaler_c, scaler_t = train_single_cvae(train_file, target_col, coords_map, device=device)
        
        if model is None: continue
            
        # 2. 生成 (Inference)
        print(f"  -> Generating {target_col} using Label Coordinates...")
        model.eval()
        
        # 使用 main data 的座標作為 Condition
        c_scaled = scaler_c.transform(main_coords)
        c_tensor = torch.FloatTensor(c_scaled).to(device)
        
        # Sampling z
        z = torch.randn(c_tensor.size(0), 8).to(device)
        
        with torch.no_grad():
            pred_scaled = model.decode(z, c_tensor)
            pred_real = scaler_t.inverse_transform(pred_scaled.cpu().numpy())
        
        df_main[target_col] = pred_real.flatten()
        print(f"  -> {target_col} Generated. Mean: {np.mean(pred_real):.2f}")

    return df_main

# === 4. Data Loaders (整合介面) ===
def get_data_loaders(main_csv_path, selected_features, cvae_config, coords_map, batch_size=32, random_state=42, samples_per_label=None):
    
    try:
        df = pd.read_csv(main_csv_path)
    except FileNotFoundError:
        print(f"錯誤: 找不到檔案 {main_csv_path}")
        return None

    # Data Sampling
    if samples_per_label is not None:
        print(f"\n[Data Sampling] 限制每個 RP 使用 {samples_per_label} 筆資料...")
        try:
            df = df.groupby('Label').sample(n=samples_per_label, random_state=random_state, replace=False)
        except ValueError:
            df = df.groupby('Label').apply(lambda x: x.sample(n=min(len(x), samples_per_label), random_state=random_state)).reset_index(drop=True)

    # === CVAE Pipeline (Updated) ===
    # 傳入 coords_map
    df = manager_cvae_pipeline(df, cvae_config, coords_map)

    # 特徵處理
    print(f"\n--- Feature Selection ---")
    for col in selected_features:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    X = df[selected_features].values
    y_raw = df['Label'].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # Split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=random_state, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=random_state, stratify=y_temp)

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
        "feature_names": selected_features
    }
    
    return train_loader, val_loader, test_loader, meta