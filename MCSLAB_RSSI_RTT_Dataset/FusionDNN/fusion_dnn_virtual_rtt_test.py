import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os
import random
import argparse
import csv

# ==========================================
# 0. 參數解析與設置
# ==========================================
parser = argparse.ArgumentParser(description='Baseline DNN with PCN Ablation')
parser.add_argument('--rtt_indices', type=str, default="1 2 3 4", help='Space separated indices')
parser.add_argument('--base_path', type=str, default='..', help='Base path for data')
parser.add_argument('--mode', type=str, required=True, choices=['fusion', 'rtt', 'rssi'], 
                    help='Input mode: fusion (RSSI+RTT), rtt (RTT only), rssi (RSSI only)')
args = parser.parse_args()

# 解析 Columns (對應 4 根天線的所有 RTT 與 RSSI)
rtt_indices = args.rtt_indices.strip().split()
mc_idx = [int(i) - 1 for i in rtt_indices]  # 轉換為 0-indexed，例如 '1 2' -> [0, 1]

ALL_RTT_COLS = ['Dist_mm_1', 'Dist_mm_2', 'Dist_mm_3', 'Dist_mm_4']
RSSI_COLS = ['RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4'] 
# 請確認您的資料集 RSSI_Diff 欄位名稱是否如下
DIFF_COLS = ['Diff_RSSI_1_2', 'Diff_RSSI_1_3', 'Diff_RSSI_1_4', 'Diff_RSSI_2_3', 'Diff_RSSI_2_4', 'Diff_RSSI_3_4']
RSSI_10_COLS = RSSI_COLS + DIFF_COLS

# 因為透過 PCN 補全了缺失的 RTT，所以進入 DNN 的維度固定為 4
RTT_DIM = 4  
RSSI_DIM = 4

if args.mode == 'fusion':
    COMBO_NAME = f"FusionDNN_PCN_RTT_{'_'.join(rtt_indices)}"
elif args.mode == 'rtt':
    COMBO_NAME = f"FusionDNN_PCN_Only_RTT_{'_'.join(rtt_indices)}"
elif args.mode == 'rssi':
    COMBO_NAME = "FusionDNN_PCN_Only_RSSI_Fixed"

print(f"==========================================")
print(f"Experiment: {COMBO_NAME}")
print(f"Mode: {args.mode} | mc_idx: {mc_idx} | Fusion Input RTT Dim: {RTT_DIM}")
print(f"==========================================")

RESULT_DIR = "results"
CDF_DIR = os.path.join(RESULT_DIR, "cdf_data")
os.makedirs(CDF_DIR, exist_ok=True)

# ==========================================
# 1. 網路架構定義 (PCN + Baseline DNN)
# ==========================================
class PathLossCalibrationNetwork(nn.Module):
    def __init__(self, in_dim=10, out_dim=4):
        super(PathLossCalibrationNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.ReLU(True),
            nn.Linear(16, out_dim)
        )

    def forward(self, diff_rssi):
        return self.net(diff_rssi)

class BaselineDNN(nn.Module):
    def __init__(self, rtt_dim=4, rssi_dim=4, num_classes=5, hidden_dim=64):
        super(BaselineDNN, self).__init__()
        self.feature_extractor_rtt = nn.Sequential(
            nn.Linear(rtt_dim, 32), 
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True)
        )
        self.feature_extractor_rssi = nn.Sequential(
            nn.Linear(rssi_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True)
        )
        self.class_classifier = nn.Sequential(
            nn.Linear(hidden_dim*2, 64), 
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x_rtt, x_rssi):
        rtt_features = self.feature_extractor_rtt(x_rtt)
        rssi_features = self.feature_extractor_rssi(x_rssi)
        features = torch.cat((rtt_features, rssi_features), dim=1) 
        class_output = self.class_classifier(features)
        return class_output

# ==========================================
# 2. 資料處理
# ==========================================
rssi_10_scaler = MinMaxScaler(feature_range=(-1, 1)) # 用於 PCN 輸入 (10維)
rssi_4_scaler = MinMaxScaler(feature_range=(-1, 1))  # 用於 DNN 輸入 (4維)
rtt_scaler = MinMaxScaler(feature_range=(-1, 1))
label_encoder = LabelEncoder()
is_scaler_fitted = False

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_wifi_data(csv_path, is_source=True, samples_per_label=None):
    global is_scaler_fitted
    df = pd.read_csv(csv_path)
    
    for col in RSSI_10_COLS:
        df[col] = df[col].replace(-100, np.nan)
    for col in ALL_RTT_COLS:
        df[col] = df[col].replace([0, -1], np.nan)

    def fill_with_mean(x):
        return x.fillna(x.mean())
    
    cols_to_fix = RSSI_10_COLS + ALL_RTT_COLS
    if is_source:
        df[cols_to_fix] = df.groupby('Label')[cols_to_fix].transform(fill_with_mean)
        
    df[RSSI_10_COLS] = df[RSSI_10_COLS].fillna(-100)
    df[ALL_RTT_COLS] = df[ALL_RTT_COLS].fillna(-1)

    if samples_per_label is not None:
        df = df.groupby('Label').apply(
            lambda x: x.sample(n=samples_per_label, replace=True) if len(x) < samples_per_label else x.sample(n=samples_per_label, replace=False)
        ).reset_index(drop=True)

    rssi_10_data = df[RSSI_10_COLS].values.astype(np.float32)
    rssi_4_data = df[RSSI_COLS].values.astype(np.float32)
    rtt_data = df[ALL_RTT_COLS].values.astype(np.float32)
    raw_labels = df['Label'].values

    if is_source:
        rssi_10_data = rssi_10_scaler.fit_transform(rssi_10_data)
        rssi_4_data = rssi_4_scaler.fit_transform(rssi_4_data)
        rtt_data = rtt_scaler.fit_transform(rtt_data)
        labels = label_encoder.fit_transform(raw_labels)
        is_scaler_fitted = True
    else:
        rssi_10_data = rssi_10_scaler.transform(rssi_10_data)
        rssi_4_data = rssi_4_scaler.transform(rssi_4_data)
        rtt_data = rtt_scaler.transform(rtt_data)
        try: labels = label_encoder.transform(raw_labels)
        except: labels = np.zeros(len(df))
        
    return (torch.tensor(rssi_10_data), torch.tensor(rssi_4_data), 
            torch.tensor(rtt_data), torch.tensor(labels, dtype=torch.long))

# 座標映射函數 (保留原邏輯)
LABEL_TO_COORDS = {
    "1-1": (0, 0), "1-2": (0.6, 0), "1-3": (1.2, 0), "1-4": (1.8, 0), "1-5": (2.4, 0), "1-6": (3.0, 0),"1-7": (3.6, 0), "1-8": (4.2, 0), "1-9": (4.8, 0), "1-10": (5.4, 0), "1-11": (6.0, 0),
    "2-1": (0, 0.6), "2-11": (6.0, 0.6),
    "3-1": (0, 1.2), "3-11": (6.0, 1.2),
    "4-1": (0, 1.8), "4-11": (6.0, 1.8),
    "5-1": (0, 2.4), "5-11": (6.0, 2.4),
    "6-1": (0, 3.0), "6-2": (0.6, 3.0), "6-3": (1.2, 3.0), "6-4": (1.8, 3.0), "6-5": (2.4, 3.0),"6-6": (3.0, 3.0), "6-7": (3.6, 3.0), "6-8": (4.2, 3.0), "6-9": (4.8, 3.0), "6-10": (5.4, 3.0), "6-11": (6.0, 3.0),
    "7-1": (0, 3.6), "7-11": (6.0, 3.6),
    "8-1": (0, 4.2), "8-11": (6.0, 4.2),
    "9-1": (0, 4.8), "9-11": (6.0, 4.8),
    "10-1": (0, 5.4), "10-11": (6.0, 5.4),
    "11-1": (0, 6.0), "11-2": (0.6, 6.0), "11-3": (1.2, 6.0), "11-4": (1.8, 6.0), "11-5": (2.4, 6.0),"11-6": (3.0, 6.0), "11-7": (3.6, 6.0), "11-8": (4.2, 6.0), "11-9": (4.8, 6.0), "11-10": (5.4, 6.0), "11-11": (6.0, 6.0)
}
def create_coord_tensor(dataset_classes, device):
    coords_list = []
    for cls_name in dataset_classes:
        if cls_name in LABEL_TO_COORDS: coords_list.append(LABEL_TO_COORDS[cls_name])
        else: coords_list.append((0, 0))
    return torch.tensor(coords_list, dtype=torch.float32).to(device)

def evaluate(model, data_loader, coord_tensor, device, return_all_errors=False):
    model.eval()
    correct = 0; total = 0; total_dist_error = 0.0; all_dists = []
    with torch.no_grad():
        for x_rtt_b, x_rssi_b, labels_b in data_loader:
            x_rtt_b, x_rssi_b, labels_b = x_rtt_b.to(device), x_rssi_b.to(device), labels_b.to(device)
            class_out = model(x_rtt_b, x_rssi_b)
            
            preds = torch.argmax(class_out, dim=1)
            correct += (preds == labels_b).sum().item()
            total += labels_b.size(0)
            
            dist = torch.norm(coord_tensor[preds] - coord_tensor[labels_b], p=2, dim=1)
            total_dist_error += dist.sum().item()
            if return_all_errors:
                all_dists.extend(dist.cpu().numpy())

    if total == 0: return 0, 0, []
    return 100.*correct/total, total_dist_error/total, np.array(all_dists)

# ==========================================
# 3. PCN 訓練邏輯
# ==========================================
def train_pcn(data_loader, mc_idx, device, epochs=50, domain_name="Source"):
    pcn = PathLossCalibrationNetwork(in_dim=10, out_dim=4).to(device)
    if len(mc_idx) == 0: return pcn

    optimizer = optim.Adam(pcn.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=0.05) 
    
    pcn.train()
    print(f">> Training PCN on {domain_name} Domain...")
    for epoch in range(epochs):
        total_loss = 0
        for rssi_10, rtt_gt, _ in data_loader:
            rssi_10, rtt_gt = rssi_10.to(device), rtt_gt.to(device)
            optimizer.zero_grad()
            pred_rtt = pcn(rssi_10)
            # 僅使用 mc_idx 內的 RTT 進行 Loss 計算與訓練
            loss = criterion(pred_rtt[:, mc_idx], rtt_gt[:, mc_idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    print(f">> {domain_name} PCN Training Finished. Huber Loss: {total_loss/len(data_loader):.4f}")
    pcn.eval()
    return pcn

# ==========================================
# 4. 主程式
# ==========================================
def main():
    results = []
    seed_candidate = [42, 6767, 123456]
    
    for seed in seed_candidate:
        set_seed(seed)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 設定資料路徑
        SOURCE_CSV = os.path.join(args.base_path, '2026_1_1/all/All_Data_With_RSSI_Diff.csv')
        TARGET_CSV = os.path.join(args.base_path, '2026_2_4/All_Data_With_RSSI_Diff_withoutNA.csv')
        SAMPLES_PER_CLASS = 120 
        BATCH_SIZE = 32
        
        # 1. 載入資料 (包含 10 維 RSSI 用於 PCN, 4 維 RSSI 用於 DNN, 4 維真實 RTT)
        s_rssi_10, s_rssi_4, s_rtt_all, s_labels = load_wifi_data(SOURCE_CSV, is_source=True, samples_per_label=SAMPLES_PER_CLASS)
        t_rssi_10, t_rssi_4, t_rtt_all, t_labels = load_wifi_data(TARGET_CSV, is_source=False, samples_per_label=SAMPLES_PER_CLASS)

        # 2. 準備 PCN 的 DataLoader
        pcn_source_ds = TensorDataset(s_rssi_10, s_rtt_all, s_labels)
        pcn_source_loader = DataLoader(pcn_source_ds, batch_size=BATCH_SIZE, shuffle=True)
        
        # 根據您的需求：Source+Target Domain 共同訓練第二個 PCN
        st_rssi_10 = torch.cat((s_rssi_10, t_rssi_10), dim=0)
        st_rtt_all = torch.cat((s_rtt_all, t_rtt_all), dim=0)
        st_labels  = torch.cat((s_labels, t_labels), dim=0)
        pcn_target_ds = TensorDataset(st_rssi_10, st_rtt_all, st_labels)
        pcn_target_loader = DataLoader(pcn_target_ds, batch_size=BATCH_SIZE, shuffle=True)

        # 3. 獨立訓練兩個 PCN
        pcn_source = train_pcn(pcn_source_loader, mc_idx, DEVICE, epochs=50, domain_name="Source")
        pcn_target = train_pcn(pcn_target_loader, mc_idx, DEVICE, epochs=50, domain_name="Source+Target")

        # 4. 利用 PCN 預測並補全未知的 RTT 虛擬距離
        pcn_source.eval()
        pcn_target.eval()
        with torch.no_grad():
            # Source 替換
            s_rtt_virtual = pcn_source(s_rssi_10.to(DEVICE)).cpu()
            s_rtt_fused = s_rtt_all.clone()
            for i in range(4):
                if i not in mc_idx:
                    s_rtt_fused[:, i] = s_rtt_virtual[:, i]
            
            # Target 替換
            t_rtt_virtual = pcn_target(t_rssi_10.to(DEVICE)).cpu()
            t_rtt_fused = t_rtt_all.clone()
            for i in range(4):
                if i not in mc_idx:
                    t_rtt_fused[:, i] = t_rtt_virtual[:, i]

        # 處理消融實驗的 Mode (保持原意：若純測量模式，將另一個特徵歸零)
        if args.mode == 'rtt':
            s_rssi_4 = torch.zeros_like(s_rssi_4)
            t_rssi_4 = torch.zeros_like(t_rssi_4)
        elif args.mode == 'rssi':
            s_rtt_fused = torch.zeros_like(s_rtt_fused)
            t_rtt_fused = torch.zeros_like(t_rtt_fused)

        # 5. 準備最終進入 Fusion DNN 的資料集 (現在 RTT 維度皆為補全後的 4 維)
        full_source_dataset = TensorDataset(s_rtt_fused, s_rssi_4, s_labels)
        train_size = int(0.8 * len(full_source_dataset))
        val_size = len(full_source_dataset) - train_size
        source_train_ds, source_val_ds = random_split(
            full_source_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        
        target_test_dataset = TensorDataset(t_rtt_fused, t_rssi_4, t_labels)
        source_test_dataset = TensorDataset(s_rtt_fused, s_rssi_4, s_labels)

        NUM_WORKERS = 0
        train_loader = DataLoader(source_train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS, pin_memory=True)
        val_loader   = DataLoader(source_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        target_test_loader  = DataLoader(target_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        source_test_loader  = DataLoader(source_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        class_names = label_encoder.classes_
        COORD_TENSOR = create_coord_tensor(class_names, DEVICE)

        # Init Fusion DNN Model (rtt_dim 固定為 4)
        model = BaselineDNN(rtt_dim=RTT_DIM, rssi_dim=RSSI_DIM, num_classes=len(class_names)).to(DEVICE)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        # Training DNN
        num_epochs = 400
        best_val_acc = 0.0 
        best_epoch = 0
        temp_model_name = f"temp_dnn_{COMBO_NAME}_seed{seed}.pth"
        
        for epoch in range(num_epochs):
            model.train()
            for x_rtt_b, x_rssi_b, label_b in train_loader:
                x_rtt_b, x_rssi_b, label_b = x_rtt_b.to(DEVICE), x_rssi_b.to(DEVICE), label_b.to(DEVICE)
                output = model(x_rtt_b, x_rssi_b)
                loss = criterion(output, label_b)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            val_acc, val_mde, _ = evaluate(model, val_loader, COORD_TENSOR, DEVICE)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                torch.save(model.state_dict(), temp_model_name)

        # Final Test
        if best_epoch != 0:
            model.load_state_dict(torch.load(temp_model_name))
            os.remove(temp_model_name)
        
        t_acc, t_mde, t_err = evaluate(model, target_test_loader, COORD_TENSOR, DEVICE, return_all_errors=True)
        s_acc, s_mde, s_err = evaluate(model, source_test_loader, COORD_TENSOR, DEVICE, return_all_errors=True)
        
        print(f"Seed {seed} | Best Epoch {best_epoch} | Tgt MDE: {t_mde:.4f}m")
        
        results.append({
            "Combo": COMBO_NAME, "Seed": seed,
            "Source_Acc": s_acc, "Source_MDE": s_mde,
            "Target_Acc": t_acc, "Target_MDE": t_mde
        })
        
        np.save(os.path.join(CDF_DIR, f"error_{COMBO_NAME}_seed{seed}.npy"), t_err)

    # 計算平均並寫入 Summary CSV
    df_res = pd.DataFrame(results)
    avg_s_acc = df_res['Source_Acc'].mean()
    avg_s_acc_std = df_res['Source_Acc'].std()
    avg_s_mde = df_res['Source_MDE'].mean()
    avg_s_mde_std = df_res['Source_MDE'].std()
    avg_t_acc = df_res['Target_Acc'].mean()
    avg_t_acc_std = df_res['Target_Acc'].std()
    avg_t_mde = df_res['Target_MDE'].mean()
    avg_t_mde_std = df_res['Target_MDE'].std()
    
    summary_file = os.path.join(RESULT_DIR, "experiment_summary_pcn.csv")
    file_exists = os.path.isfile(summary_file)
    
    with open(summary_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Combo", "Avg_Src_Acc", "Avg_Src_Acc_STD", "Avg_Src_MDE", "Avg_Src_MDE_STD", "Avg_Tgt_Acc", "Avg_Tgt_Acc_STD", "Avg_Tgt_MDE", "Avg_Tgt_MDE_STD", "Seeds_Detail"])
        writer.writerow([COMBO_NAME, f"{avg_s_acc:.4f}", f"{avg_s_acc_std:.4f}", f"{avg_s_mde:.4f}", f"{avg_s_mde_std:.4f}", f"{avg_t_acc:.4f}", f"{avg_t_acc_std:.4f}", f"{avg_t_mde:.4f}", f"{avg_t_mde_std:.4f}", str(seed_candidate)])
        
    print(f"Finished Combo {COMBO_NAME}. Results saved.")

if __name__ == '__main__':
    main()