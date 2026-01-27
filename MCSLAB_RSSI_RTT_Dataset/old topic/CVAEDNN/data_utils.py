import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import os

CVAE_TRAIN_EPOCH = 1500

# === CVAE 模型架構 (結構不變，參數在生成時改變) ===
class CVAE(nn.Module):
    def __init__(self, input_dim, cond_dim, latent_dim=16): # 稍微加大 latent_dim 以容納 4D 關係
        super(CVAE, self).__init__()
        # Encoder
        self.encoder_fc1 = nn.Linear(input_dim + cond_dim, 256)
        self.encoder_fc2 = nn.Linear(256, 128)
        self.z_mean = nn.Linear(128, latent_dim)
        self.z_log_var = nn.Linear(128, latent_dim)
        # Decoder
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
    # Sum over features (4 dimensions), then average or sum over batch
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD * kl_weight, MSE, KLD

# === [核心新功能] MIMO 資料配對 ===
def prepare_mimo_data(df_student, cvae_config):
    """
    功能: 準備 4-to-4 的訓練資料
    Condition: [RSSI_1, RSSI_2, RSSI_3, RSSI_4]
    Target:    [Dist_1, Dist_2, Dist_3, Dist_4]
    """
    print("  -> Preparing MIMO Dataset (Merging 4 Teacher files)...")
    
    # 1. 解析 Config，預先載入所有 Teacher Data 到記憶體
    # 結構: teacher_data_map[Label] = { 'Dist_Pred_1': [values...], 'Dist_Pred_2': [...], ... }
    
    teacher_lookups = {} # Key: Target_Col_Name, Value: DataFrame (Label -> Dist values)
    rssi_cols = []       # 順序很重要，要跟 Target 對齊
    target_cols = []     # e.g., ['Dist_Pred_1', 'Dist_Pred_2'...]
    
    for target_col, config in cvae_config.items():
        t_path = config['train_file']
        rssi_col = config['rssi_col']
        
        rssi_cols.append(rssi_col)
        target_cols.append(target_col)
        
        if not os.path.exists(t_path):
            print(f"Error: Teacher file missing {t_path}")
            return None, None
            
        df_t = pd.read_csv(t_path)
        
        # 找 Dist 欄位
        dist_c = [c for c in df_t.columns if 'Dist_mm' in c]
        if not dist_c: return None, None
        dist_col_name = dist_c[0]
        
        # 清洗 NaN
        df_t = df_t.dropna(subset=[dist_col_name])
        
        # 轉成 Dictionary 以加速查找: dict[Label] = numpy_array_of_distances
        # groupby apply list is slow, so we iterate manually or use efficient grouping
        grouped = df_t.groupby('Label')[dist_col_name].apply(np.array).to_dict()
        teacher_lookups[target_col] = grouped

    # 2. 遍歷 Student 資料，構建 (RSSI Vector, Dist Vector)
    # 確保 Student 的 RSSI 欄位都存在
    if not all(col in df_student.columns for col in rssi_cols):
        print("Error: Student data missing required RSSI columns.")
        return None, None

    # 過濾出 Student 中有完整 RSSI 的 rows (雖然外面已經 dropna，但再確認一次)
    valid_student = df_student.dropna(subset=rssi_cols)
    
    cond_list = []
    target_list = []
    
    # 為了加速，我們以 Label 為單位處理
    # 找出所有 Teacher 都有資料的 Label (交集)
    valid_labels = set(valid_student['Label'].unique())
    for t_data in teacher_lookups.values():
        valid_labels = valid_labels.intersection(set(t_data.keys()))
        
    print(f"  -> Found {len(valid_labels)} labels common to Student and ALL Teachers.")
    
    for label in valid_labels:
        # 取出該 Label 的 Student RSSI 矩陣 (N_student, 4)
        s_subset = valid_student[valid_student['Label'] == label]
        if s_subset.empty: continue
        
        # (N_samples, 4)
        s_rssi_matrix = s_subset[rssi_cols].values 
        n_samples = len(s_rssi_matrix)
        
        # 建構對應的 Teacher Dist 矩陣 (N_samples, 4)
        # 對於每一個 AP (Dimension)，從 Teacher lookup 中隨機抽樣 n_samples 個值
        
        t_matrix_cols = []
        for t_col in target_cols:
            t_vals = teacher_lookups[t_col][label] # 這是該 Label 下該 AP 的所有測距值
            if len(t_vals) == 0: 
                # 理論上前面 intersection 擋過了，但以防萬一
                t_sampled = np.zeros(n_samples) 
            else:
                t_sampled = np.random.choice(t_vals, size=n_samples, replace=True)
            t_matrix_cols.append(t_sampled)
            
        # Stack 成 (N_samples, 4)
        # t_matrix_cols is list of arrays. stack -> (4, N) -> transpose -> (N, 4)
        t_dist_matrix = np.vstack(t_matrix_cols).T
        
        cond_list.append(s_rssi_matrix)
        target_list.append(t_dist_matrix)
        
    if not cond_list:
        return None, None

    # 合併所有 Label 的資料
    final_cond = np.vstack(cond_list)   # (Total_Samples, 4)
    final_target = np.vstack(target_list) # (Total_Samples, 4)
    
    return final_cond, final_target

# === MIMO 訓練函數 ===
def train_mimo_cvae(df_student, cvae_config, run_id, device='gpu'):
    
    # 1. 準備資料
    cond_data, target_data = prepare_mimo_data(df_student, cvae_config)
    
    if cond_data is None:
        print("Error: MIMO data preparation failed.")
        return None
        
    print(f"--- Training MIMO CVAE (Samples: {len(cond_data)}, Dim: 4->4) ---")

    # 2. Scaling
    scaler_cond = StandardScaler()
    scaler_target = StandardScaler()
    
    cond_scaled = scaler_cond.fit_transform(cond_data)
    target_scaled = scaler_target.fit_transform(target_data)
    
    # 3. DataLoader
    dataset = TensorDataset(
        torch.FloatTensor(target_scaled).to(device), # x
        torch.FloatTensor(cond_scaled).to(device)    # c
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 4. Model Setup (Input=4, Cond=4)
    num_dims = cond_data.shape[1] # Should be 4
    model = CVAE(input_dim=num_dims, cond_dim=num_dims, latent_dim=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    history = {'total_loss': [], 'mse': []}
    
    # 5. Training Loop
    model.train()
    for epoch in range(CVAE_TRAIN_EPOCH):
        epoch_loss = 0
        epoch_mse = 0
        
        for x, c in loader:
            optimizer.zero_grad()
            recon, mu, logvar = model(x, c)
            loss, mse, kld = loss_function_cvae(recon, x, mu, logvar, kl_weight=0.1)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_mse += mse.item()
            
        avg_loss = epoch_loss / len(loader.dataset)
        avg_mse = epoch_mse / len(loader.dataset)
        history['total_loss'].append(avg_loss)
        history['mse'].append(avg_mse)
        
        if (epoch+1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{CVAE_TRAIN_EPOCH} | Loss: {avg_loss:.4f} (MSE: {avg_mse:.4f})")
            
    # 6. Save Plot
    plt.figure()
    plt.plot(history['total_loss'], label='Total Loss')
    plt.plot(history['mse'], label='MSE', linestyle='--')
    plt.title(f'MIMO CVAE Training (Run {run_id})')
    plt.savefig(f'MIMO_CVAE_Loss_Run_{run_id}.png')
    plt.close()
    
    return {
        'model': model,
        'scaler_c': scaler_cond,
        'scaler_t': scaler_target,
        'config': cvae_config # 記住 config 以便知道 column 順序
    }

# === MIMO 推論函數 ===
def augment_with_mimo_cvae(df, mimo_tools, device='cpu'):
    if mimo_tools is None: return df
    
    df_out = df.copy()
    model = mimo_tools['model']
    scaler_c = mimo_tools['scaler_c']
    scaler_t = mimo_tools['scaler_t']
    config = mimo_tools['config']
    
    # 1. 準備輸入矩陣 (必須依照 Config 的順序提取 RSSI)
    rssi_cols = []
    target_cols = []
    for t_col, cfg in config.items():
        rssi_cols.append(cfg['rssi_col'])
        target_cols.append(t_col)
        
    # 確認欄位存在
    if not all(c in df_out.columns for c in rssi_cols):
        return df_out
        
    rssi_data = df_out[rssi_cols].fillna(-100).values # (N, 4)
    
    try:
        rssi_scaled = scaler_c.transform(rssi_data)
    except:
        return df_out
        
    c_tensor = torch.FloatTensor(rssi_scaled).to(device)
    z = torch.randn(c_tensor.size(0), 16).to(device) # Latent dim = 16
    
    model.eval()
    with torch.no_grad():
        pred_scaled = model.decode(z, c_tensor)
        pred_real = scaler_t.inverse_transform(pred_scaled.cpu().numpy()) # (N, 4)
        
    # 2. 寫回 DataFrame
    # pred_real 的第 k 個 column 對應 target_cols 的第 k 個名稱
    for i, t_col in enumerate(target_cols):
        df_out[t_col] = pred_real[:, i]
        
    return df_out

# === 在 data_utils.py 新增此函數 ===
def analyze_cvae_quality(df_test, cvae_config, run_id=0):
    """
    功能: 畫出 True Distance (Teacher) vs Predicted Distance (CVAE Student) 的散佈圖
    目的: 檢查 CVAE 在 Test Set 上是否有學到東西，還是只在瞎猜平均值
    """
    print(f"  -> [Diagnosis] Visualizing CVAE generation quality for Run {run_id}...")
    
    # 設定圖表大小 (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, (target_col, config) in enumerate(cvae_config.items()):
        ax = axes[idx]
        
        # 1. 取得該 AP 的真實距離 (Ground Truth)
        # 我們必須重新讀取 Teacher 檔案，計算每個 Label 的「平均真實距離」作為對照基準
        if not os.path.exists(config['train_file']):
            continue
            
        df_teacher = pd.read_csv(config['train_file'])
        dist_col_name = [c for c in df_teacher.columns if 'Dist_mm' in c][0]
        
        # 計算 Teacher 每個 Label 的平均距離
        teacher_truth_map = df_teacher.groupby('Label')[dist_col_name].mean().to_dict()
        
        # 2. 將真實距離 Mapping 到 df_test
        # 注意: 這裡是為了畫圖比較，所以我們將同一 Label 的所有 Student Sample 都對應到該 Label 的 Teacher 平均距離
        truth_values = df_test['Label'].map(teacher_truth_map)
        pred_values = df_test[target_col] # 這是 CVAE 生成的
        
        # 過濾掉對應不到 Label 的資料 (以防萬一)
        valid_mask = ~truth_values.isna()
        truth_values = truth_values[valid_mask]
        pred_values = pred_values[valid_mask]
        
        if len(truth_values) == 0:
            continue

        # 3. 畫散佈圖
        ax.scatter(truth_values, pred_values, alpha=0.3, s=10, c='blue', label='Test Samples')
        
        # 畫一條 y=x 的理想線 (紅線)
        min_val = min(truth_values.min(), pred_values.min())
        max_val = max(truth_values.max(), pred_values.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal (y=x)')
        
        # 計算相關係數 (Correlation)
        corr = np.corrcoef(truth_values, pred_values)[0, 1]
        
        ax.set_title(f"{target_col} (Corr: {corr:.3f})")
        ax.set_xlabel("True Teacher Dist (Mean per Label)")
        ax.set_ylabel("CVAE Generated Dist")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle(f"CVAE Generation Diagnosis (Test Set) - Run {run_id}", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"Diagnosis_CVAE_Run_{run_id}.png")
    plt.close()
    print(f"  -> Saved diagnosis plot to 'Diagnosis_CVAE_Run_{run_id}.png'")

# === Data Loading Pipeline ===
def get_data_loaders(main_csv_path, selected_features, cvae_config, batch_size=32, random_state=42, samples_per_label=None, run_id=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load & Clean Student Data
    try:
        df = pd.read_csv(main_csv_path)
    except FileNotFoundError:
        return None

    df = df.dropna()
    
    if samples_per_label is not None:
        try:
            df = df.groupby('Label').sample(n=samples_per_label, random_state=random_state, replace=False)
        except ValueError:
            df = df.groupby('Label').apply(lambda x: x.sample(n=min(len(x), samples_per_label), random_state=random_state)).reset_index(drop=True)

    # 2. Split FIRST (先切分)
    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(df['Label'].values)
    
    # Split
    df_temp, df_test, y_temp, y_test = train_test_split(
        df, y_all, test_size=0.15, random_state=random_state, stratify=y_all
    )
    df_train, df_val, y_train, y_val = train_test_split(
        df_temp, y_temp, test_size=0.15/0.85, random_state=random_state, stratify=y_temp
    )

    # 3. Train MIMO CVAE ONLY on Training Data (只用訓練集訓練 CVAE)
    # 注意：這裡只傳入 df_train
    mimo_tools = train_mimo_cvae(df_train, cvae_config, run_id, device=device)

    # 4. MIMO Inference (對所有集合進行推論)
    # Train set: 使用自己訓練出來的模型生成特徵 (In-sample generation)
    # Val/Test set: 使用 Train set 訓練出來的模型生成特徵 (Out-of-sample generation, 符合現實)
    df_train = augment_with_mimo_cvae(df_train, mimo_tools, device)
    df_val = augment_with_mimo_cvae(df_val, mimo_tools, device)
    df_test = augment_with_mimo_cvae(df_test, mimo_tools, device)

    # === [新增] 除錯用：儲存生成結果 (只在第一次執行時存檔) ===
    if run_id == 0:
        print("  -> [Debug] Saving generated RTT data to 'Debug_Generated_RTT.csv'...")
        # 將 Train/Val/Test 合併以便觀察整體分佈
        df_debug = pd.concat([df_train, df_val, df_test], axis=0)
        
        # 準備要輸出的欄位：Label + RSSI columns + Generated RTT columns
        rssi_cols = [cfg['rssi_col'] for cfg in cvae_config.values()]
        rtt_cols = list(cvae_config.keys()) # e.g., ['Dist_Pred_1', 'Dist_Pred_2'...]
        
        # 確保欄位存在 (避免報錯)
        cols_to_save = ['Label'] + [c for c in rssi_cols if c in df_debug.columns] + [c for c in rtt_cols if c in df_debug.columns]
        
        # 存檔
        df_debug[cols_to_save].to_csv('Debug_Generated_RTT.csv', index=False)
        print("  -> Saved!")
    # ========================================================

    # 5. Feature Selection
    def process_features(sub_df):
        for col in selected_features:
            if col not in sub_df.columns: sub_df[col] = 0.0
        return sub_df[selected_features].values

    X_train = process_features(df_train)
    X_val   = process_features(df_val)
    X_test  = process_features(df_test)

    if run_id == 0:
        analyze_cvae_quality(df_test, cvae_config, run_id=0)

    scaler_dnn = StandardScaler()
    X_train_scaled = scaler_dnn.fit_transform(X_train)
    X_val_scaled   = scaler_dnn.transform(X_val)
    X_test_scaled  = scaler_dnn.transform(X_test)
    
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train)), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.FloatTensor(X_val_scaled),   torch.LongTensor(y_val)),   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(TensorDataset(torch.FloatTensor(X_test_scaled),  torch.LongTensor(y_test)),  batch_size=batch_size, shuffle=False)

    meta = {
        "input_dim": len(selected_features),
        "num_classes": len(label_encoder.classes_),
        "label_encoder": label_encoder,
        "feature_names": selected_features
    }
    
    return train_loader, val_loader, test_loader, meta