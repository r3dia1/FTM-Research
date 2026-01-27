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

# === 1. 定義 CVAE 模型 (PyTorch 版) ===
class CVAE(nn.Module):
    def __init__(self, input_dim, cond_dim, latent_dim=8):
        super(CVAE, self).__init__()
        
        # --- Encoder ---
        # 輸入: Distance (Target) + RSSI (Condition)
        self.encoder_fc1 = nn.Linear(input_dim + cond_dim, 256)
        self.encoder_fc2 = nn.Linear(256, 128)
        self.z_mean = nn.Linear(128, latent_dim)
        self.z_log_var = nn.Linear(128, latent_dim)
        
        # --- Decoder ---
        # 輸入: Latent z + RSSI (Condition)
        self.decoder_fc1 = nn.Linear(latent_dim + cond_dim, 128)
        self.decoder_fc2 = nn.Linear(128, 256)
        # 輸出: Distance (回歸任務通常不加 Sigmoid，除非資料歸一化到 0-1)
        self.decoder_out = nn.Linear(256, input_dim) 
        
        self.relu = nn.ReLU()

    def encode(self, x, c):
        # x: Distance, c: RSSI
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
    # Reconstruction Loss (MSE): 衡量距離預測準不準
    # reduction='sum' 表示加總整個 batch 的誤差，稍後算平均時要注意
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    
    # KL Divergence: 衡量 Latent z 是否符合常態分佈
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total Loss
    total_loss = MSE + KLD * kl_weight
    
    return total_loss, MSE, KLD

def visualize_cvae_results(df, loop_id=0):
    """
    畫出兩張圖：
    1. 真實資料分佈 (AP1 & AP4): 用來當作 Ground Truth 的參考基準。
    2. CVAE 生成結果 (AP2 & AP3): 檢查生成出來的距離與 RSSI 的關係是否符合物理定律。
    """
    plt.figure(figsize=(18, 6))

    # --- 子圖 1: AP1 & AP4 的真實資料 (訓練來源) ---
    plt.subplot(1, 2, 1)
    
    # 提取 AP1/AP4 真實資料
    real_data_1 = df[['RSSI_1', 'Dist_mm_1']].dropna()
    real_data_1.columns = ['RSSI', 'Distance']
    real_data_1['Source'] = 'AP1 (Real)'
    
    real_data_4 = df[['RSSI_4', 'Dist_mm_4']].dropna()
    real_data_4.columns = ['RSSI', 'Distance']
    real_data_4['Source'] = 'AP4 (Real)'
    
    real_combined = pd.concat([real_data_1, real_data_4], ignore_index=True)
    
    # 繪圖
    sns.scatterplot(data=real_combined, x='RSSI', y='Distance', hue='Source', alpha=0.5, style='Source')
    plt.title('Ground Truth Distribution (AP1 & AP4)')
    plt.xlabel('RSSI (dBm)')
    plt.ylabel('Distance (mm)')
    plt.grid(True, linestyle='--', alpha=0.6)
    # 反轉 X 軸，因為 RSSI 越大(越右邊)代表距離越近，符合直覺
    # plt.gca().invert_xaxis() 

    # --- 子圖 2: AP2 & AP3 的 CVAE 生成資料 ---
    plt.subplot(1, 2, 2)
    
    # 提取生成的資料
    # 注意：這裡看的是 Dist_Pred_2 和 Dist_Pred_3
    pred_data_2 = df[['RSSI_2', 'Dist_Pred_2', 'Label']].dropna()
    pred_data_2.columns = ['RSSI', 'Distance', 'Label']
    pred_data_2['Source'] = 'AP2 (Generated)'
    
    pred_data_3 = df[['RSSI_3', 'Dist_Pred_3', 'Label']].dropna()
    pred_data_3.columns = ['RSSI', 'Distance', 'Label']
    pred_data_3['Source'] = 'AP3 (Generated)'
    
    pred_combined = pd.concat([pred_data_2, pred_data_3], ignore_index=True)

    if not pred_combined.empty:
        # 這裡我們用 'Label' 當作 hue，看看不同 RP 的生成狀況
        # 由於 Label 太多，建議只取前 10 個 Label 來畫，不然圖會太亂
        top_labels = pred_combined['Label'].unique()[:10]
        subset = pred_combined[pred_combined['Label'].isin(top_labels)]
        
        sns.scatterplot(data=subset, x='RSSI', y='Distance', hue='Label', alpha=0.7, palette='tab10')
        plt.title('CVAE Generated Distribution (AP2 & AP3) - Top 10 RPs')
        plt.xlabel('RSSI (dBm)')
        plt.ylabel('Predicted Distance (mm)')
        plt.grid(True, linestyle='--', alpha=0.6)
    else:
        plt.title("No Generated Data for AP2/AP3")

    plt.tight_layout()
    filename = f'cvae_generation_check_loop_{loop_id}.png'
    plt.savefig(filename)
    print(f"CVAE 生成分佈圖已儲存為: {filename}")
    # plt.show() # 如果是 Jupyter Notebook 可以打開這行

def save_cvae_generation_stats(df, loop_id=0):
    """
    計算每個 RP (Label) 在 AP2 和 AP3 的：
    1. 平均 RSSI (Condition)
    2. 平均 預測距離 (Generated Distance Mean)
    3. 預測距離的標準差 (Generated Distance Std)
    並儲存為 JSON。
    """
    stats_report = {}
    
    # 定義我們要分析的目標：(AP名稱, RSSI欄位, 預測距離欄位)
    targets = [
        ('AP2', 'RSSI_2', 'Dist_Pred_2'), 
        ('AP3', 'RSSI_3', 'Dist_Pred_3')
    ]

    for ap_name, rssi_col, dist_col in targets:
        # 檢查欄位是否存在
        if rssi_col not in df.columns or dist_col not in df.columns:
            continue
            
        # 只取有預測值的資料 (非 0 且非 NaN)
        # 假設未預測填為 0.0，我們這裡過濾掉
        valid_df = df[(df[dist_col].notna()) & (df[dist_col] != 0)]
        
        if valid_df.empty:
            continue

        # 準備存放該 AP 的統計資料
        ap_stats = {}
        
        # 依照 Label 分組計算
        grouped = valid_df.groupby('Label')
        
        for label, group in grouped:
            # 計算統計量
            rssi_mean = group[rssi_col].mean()
            dist_mean = group[dist_col].mean()
            dist_std = group[dist_col].std()
            
            # 處理 Std 為 NaN 的情況 (如果該 Label 只有 1 筆資料)
            if pd.isna(dist_std):
                dist_std = 0.0
            
            # 寫入字典 (轉成 float 確保 JSON 可序列化)
            ap_stats[str(label)] = {
                "avg_rssi": float(rssi_mean),
                "avg_dist_pred": float(dist_mean),
                "std_dist_pred": float(dist_std),
                "count": int(len(group)) # 順便記一下樣本數
            }
            
        stats_report[ap_name] = ap_stats

    # 儲存 JSON
    filename = f"CVAE_Gen_Stats_Run_{loop_id}.json"
    with open(filename, 'w') as f:
        json.dump(stats_report, f, indent=4)
        
    print(f"CVAE 生成統計報表已儲存: {filename}")

# === 2. 修改訓練函式，加入紀錄與繪圖 ===
def train_augment_cvae(df, loop_id=0):
    print("--- [Reg-CVAE] 啟動 CVAE 生成式回歸訓練 ---")
    
    # ... (資料準備與標準化部分維持不變) ...
    cols_check = ['RSSI_1', 'Dist_mm_1', 'RSSI_4', 'Dist_mm_4']
    if not all(col in df.columns for col in cols_check):
        print(f"警告: 欄位缺失 {cols_check}，跳過 CVAE。")
        return df, None

    ap1_data = df[['RSSI_1', 'Dist_mm_1']].dropna().rename(columns={'RSSI_1': 'Rssi', 'Dist_mm_1': 'Distance'})
    ap4_data = df[['RSSI_4', 'Dist_mm_4']].dropna().rename(columns={'RSSI_4': 'Rssi', 'Dist_mm_4': 'Distance'})
    train_data = pd.concat([ap1_data, ap4_data], ignore_index=True)

    cond_train = train_data[['Rssi']].values
    target_train = train_data[['Distance']].values

    scaler_cond = StandardScaler()
    scaler_target = StandardScaler()
    cond_train_scaled = scaler_cond.fit_transform(cond_train)
    target_train_scaled = scaler_target.fit_transform(target_train)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cond_tensor = torch.FloatTensor(cond_train_scaled).to(device)
    target_tensor = torch.FloatTensor(target_train_scaled).to(device)
    
    # 初始化 CVAE
    input_dim = 1
    cond_dim = 1
    latent_dim = 8
    model_cvae = CVAE(input_dim, cond_dim, latent_dim).to(device)
    optimizer = optim.Adam(model_cvae.parameters(), lr=0.001)

    # --- 訓練參數與紀錄列表 ---
    epochs = 100 # 建議增加 Epoch 觀察收斂
    batch_size = 64
    kl_weight = 0.1 # 這是關鍵參數
    
    loss_history = {'total': [], 'mse': [], 'kld': []} # 用來存紀錄
    
    dataset = TensorDataset(target_tensor, cond_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model_cvae.train()
    print("開始訓練 CVAE...")
    
    for epoch in range(epochs):
        epoch_total = 0
        epoch_mse = 0
        epoch_kld = 0
        num_samples = 0
        
        for x_batch, c_batch in loader:
            optimizer.zero_grad()
            recon_x, mu, logvar = model_cvae(x_batch, c_batch)
            
            # 計算 Loss
            loss, mse, kld = loss_function_cvae(recon_x, x_batch, mu, logvar, kl_weight)
            
            loss.backward()
            optimizer.step()
            
            # 累加誤差 (注意 loss 是 sum reduction)
            epoch_total += loss.item()
            epoch_mse += mse.item()
            epoch_kld += kld.item()
            num_samples += x_batch.size(0)
        
        # 計算平均 Loss
        avg_total = epoch_total / num_samples
        avg_mse = epoch_mse / num_samples
        avg_kld = epoch_kld / num_samples
        
        loss_history['total'].append(avg_total)
        loss_history['mse'].append(avg_mse)
        loss_history['kld'].append(avg_kld)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Total: {avg_total:.4f} | MSE: {avg_mse:.4f} | KLD: {avg_kld:.4f}")

    print("CVAE Trained.")

    # === 3. 繪製 Loss 曲線 ===
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history['total'], label='Total Loss', color='blue', linewidth=2)
    plt.plot(loss_history['mse'], label='Reconstruction (MSE)', color='orange', linestyle='--')
    plt.plot(loss_history['kld'], label='KL Divergence', color='green', linestyle=':')
    plt.title('CVAE Training Loss Convergence')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 儲存圖片以便查看 (或者在 Notebook 中直接 plt.show())
    plt.savefig('cvae_loss_convergence.png')
    print("Loss 曲線已儲存為 'cvae_loss_convergence.png'")
    # plt.show() # 如果是 Jupyter Notebook 可取消註解

    # ... (後續預測部分維持不變) ...
    model_cvae.eval()
    df['Dist_Pred_2'] = 0.0
    df['Dist_Pred_3'] = 0.0
    
    with torch.no_grad():
        if 'RSSI_2' in df.columns:
            mask = df['RSSI_2'].notna()
            if mask.any():
                rssi_in = df.loc[mask, ['RSSI_2']].values
                rssi_scaled = scaler_cond.transform(rssi_in)
                c_tensor = torch.FloatTensor(rssi_scaled).to(device)
                z = torch.randn(c_tensor.size(0), latent_dim).to(device)
                pred_scaled = model_cvae.decode(z, c_tensor)
                pred_real = scaler_target.inverse_transform(pred_scaled.cpu().numpy())
                df.loc[mask, 'Dist_Pred_2'] = pred_real

        if 'RSSI_3' in df.columns:
            mask = df['RSSI_3'].notna()
            if mask.any():
                rssi_in = df.loc[mask, ['RSSI_3']].values
                rssi_scaled = scaler_cond.transform(rssi_in)
                c_tensor = torch.FloatTensor(rssi_scaled).to(device)
                z = torch.randn(c_tensor.size(0), latent_dim).to(device)
                pred_scaled = model_cvae.decode(z, c_tensor)
                pred_real = scaler_target.inverse_transform(pred_scaled.cpu().numpy())
                df.loc[mask, 'Dist_Pred_3'] = pred_real

    visualize_cvae_results(df, loop_id)
    save_cvae_generation_stats(df, loop_id)

    return df

# === 下面的 get_data_loaders 維持原樣，只需呼叫 train_augment_cvae ===
def get_data_loaders(csv_path, batch_size=32, random_state=42, loop_id=0):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"錯誤: 找不到檔案 {csv_path}")
        return None

    df = train_augment_cvae(df, loop_id=loop_id)

    # 2. 選擇特徵 (與 RegDNN 相同)
    feature_cols = [
        'Dist_mm_1', 
        # 'Dist_mm_2','Dist_mm_3', 'Dist_mm_4'
        'Std_mm_1',  
        # 'Std_mm_2','Std_mm_3',  'Std_mm_4',
        'RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4',
        # 'Dist_pred_1', 
        'Dist_Pred_2', 'Dist_Pred_3', 'Dist_Pred_4'
    ]
    
    for col in feature_cols:
        if col not in df.columns: df[col] = 0.0
        else: df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 補值
    df[feature_cols] = df.groupby('Label')[feature_cols].transform(lambda x: x.fillna(x.mean()))
    df[feature_cols] = df[feature_cols].fillna(0)

    X = df[feature_cols].values
    y_raw = df['Label'].values

    # 3. Label Encode
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # 4. Random Split
    print(f"正在執行 Random Split (seed={random_state})...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=random_state, stratify=y
    )
    val_size = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )

    # 5. 標準化
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
        "label_encoder": label_encoder
    }
    
    return train_loader, val_loader, test_loader, meta