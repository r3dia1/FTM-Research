import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch.nn.functional as F
import os
import random
import argparse
import csv

# ==========================================
# 0. 參數解析與設置
# ==========================================
parser = argparse.ArgumentParser(description='Baseline DNN Ablation')
parser.add_argument('--rtt_indices', type=str, default="1 2 3 4", help='Space separated indices')
parser.add_argument('--base_path', type=str, default='..', help='Base path for data')
parser.add_argument('--mode', type=str, required=True, choices=['fusion', 'rtt', 'rssi'], 
                    help='Input mode: fusion (RSSI+RTT), rtt (RTT only), rssi (RSSI only)')
parser.add_argument('--future_csv', type=str, default='', help='Path for the future date test dataset (optional)')
args = parser.parse_args()

# 解析 Columns
rtt_indices = args.rtt_indices.strip().split()
RTT_COLS = [f'Dist_mm_{i}' for i in rtt_indices]
RSSI_COLS = ['RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4'] 

# [核心修改 1] 分別取得 RTT 與 RSSI 的維度，不再合併 INPUT_DIM
RTT_DIM = len(RTT_COLS)
RSSI_DIM = len(RSSI_COLS)

if args.mode == 'fusion':
    COMBO_NAME = f"FusionDNN_RTT_{'_'.join(rtt_indices)}"
elif args.mode == 'rtt':
    COMBO_NAME = f"FusionDNN_Only_RTT_{'_'.join(rtt_indices)}"
elif args.mode == 'rssi':
    COMBO_NAME = "FusionDNN_Only_RSSI_Fixed"

print(f"==========================================")
print(f"Experiment: {COMBO_NAME}")
print(f"Mode: {args.mode} | RTT Dim: {RTT_DIM} | RSSI Dim: {RSSI_DIM}")
print(f"Features: RSSI({RSSI_DIM}) + RTT({RTT_DIM})" if args.mode == 'fusion' else f"Features: {args.mode.upper()}")
print(f"==========================================")

# 建立結果資料夾
RESULT_DIR = "results"
CDF_DIR = os.path.join(RESULT_DIR, "cdf_data")
os.makedirs(CDF_DIR, exist_ok=True)

# ==========================================
# 1. 模型架構：標準 DNN (雙萃取器版本)
# ==========================================
class BaselineDNN(nn.Module):
    # [核心修改 2] 接收獨立的 rtt_dim 與 rssi_dim
    def __init__(self, rtt_dim=4, rssi_dim=4, num_classes=5, hidden_dim=64):
        super(BaselineDNN, self).__init__()

        # --- 特徵提取器 ---
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

        # --- 標籤分類器 ---
        self.class_classifier = nn.Sequential(
            nn.Linear(hidden_dim*2, 64), # 這裡維持 hidden_dim*2，因為後續用拼接
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x_rtt, x_rssi):
        rtt_features = self.feature_extractor_rtt(x_rtt)
        rssi_features = self.feature_extractor_rssi(x_rssi)
        
        # [核心修改 3] 原本為 '+'，改為 'cat' 以符合後方 classifier 要求 input 維度為 hidden_dim*2
        features = torch.cat((rtt_features, rssi_features), dim=1) 
        class_output = self.class_classifier(features)
        return class_output

# ==========================================
# 資料處理 (保持不變)
# ==========================================
rssi_scaler = MinMaxScaler(feature_range=(-1, 1))
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

def load_wifi_data(csv_path, is_source=True, samples_per_label=None, rtt_cols_to_use=None):
    global is_scaler_fitted
    df = pd.read_csv(csv_path)
    
    rssi_cols = RSSI_COLS
    rtt_cols = rtt_cols_to_use 
    
    for col in rssi_cols:
        df[col] = df[col].replace(-100, np.nan)
    for col in rtt_cols:
        df[col] = df[col].replace([0, -1], np.nan)

    def fill_with_mean(x):
        return x.fillna(x.mean())
    
    cols_to_fix = rssi_cols + rtt_cols
    if is_source:
        df[cols_to_fix] = df.groupby('Label')[cols_to_fix].transform(fill_with_mean)
    else:
        pass
    
    df[rssi_cols] = df[rssi_cols].fillna(-100)
    df[rtt_cols] = df[rtt_cols].fillna(-1)

    if samples_per_label is not None:
        df = df.groupby('Label').apply(
            lambda x: x.sample(n=samples_per_label, replace=True) if len(x) < samples_per_label else x.sample(n=samples_per_label, replace=False)
        ).reset_index(drop=True)

    rssi_data = df[rssi_cols].values.astype(np.float32)
    rtt_data = df[rtt_cols].values.astype(np.float32)
    raw_labels = df['Label'].values

    if is_source:
        rssi_data = rssi_scaler.fit_transform(rssi_data)
        rtt_data = rtt_scaler.fit_transform(rtt_data)
        labels = label_encoder.fit_transform(raw_labels)
        is_scaler_fitted = True
    else:
        if not is_scaler_fitted: raise ValueError("Error: Scaler not fitted.")
        rssi_data = rssi_scaler.transform(rssi_data)
        rtt_data = rtt_scaler.transform(rtt_data)
        try: labels = label_encoder.transform(raw_labels)
        except: labels = np.zeros(len(df))
    return torch.tensor(rssi_data), torch.tensor(rtt_data), torch.tensor(labels, dtype=torch.long)

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

# [核心修改 4] evaluate 迴圈改為接收 rtt 與 rssi 兩種 tensor
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
# 3. 主程式
# ==========================================
def main():
    results = []
    seed_candidate = [42, 6767, 123456]
    
    for seed in seed_candidate:
        set_seed(seed)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 設定資料路徑
        SOURCE_CSV = os.path.join(args.base_path, '2026_1_1/all/All_Data_With_RSSI_Diff.csv')
        # TARGET_CSV = os.path.join(args.base_path, '2026_3_17/All_Data_With_RSSI_Diff_withoutNA.csv')
        TARGET_CSV = os.path.join(args.base_path, '2026_2_4/All_Data_With_RSSI_Diff_withoutNA.csv')
        FUTURE_CSV = args.future_csv
        has_future_test = bool(FUTURE_CSV)

        SAMPLES_PER_CLASS = 120 
        
        # Load Data
        s_rssi, s_rtt, s_labels = load_wifi_data(SOURCE_CSV, is_source=True, samples_per_label=SAMPLES_PER_CLASS, rtt_cols_to_use=RTT_COLS)
        t_rssi, t_rtt, t_labels = load_wifi_data(TARGET_CSV, is_source=False, samples_per_label=SAMPLES_PER_CLASS, rtt_cols_to_use=RTT_COLS)

        if has_future_test:
            f_rssi, f_rtt, f_labels = load_wifi_data(FUTURE_CSV, is_source=False, samples_per_label=None, rtt_cols_to_use=RTT_COLS)

        # [核心修改 5] 利用 zero tensor 技巧來保留雙路徑模型，並完美對應不同的訓練模式(mode)
        if args.mode == 'rtt':
            s_rssi = torch.zeros_like(s_rssi)
            t_rssi = torch.zeros_like(t_rssi)
            if has_future_test:
                f_rssi = torch.zeros_like(f_rssi)
        elif args.mode == 'rssi':
            s_rtt = torch.zeros_like(s_rtt)
            t_rtt = torch.zeros_like(t_rtt)
            if has_future_test:
                f_rtt = torch.zeros_like(f_rtt)

        # 將 s_rtt 和 s_rssi 分別放入 TensorDataset
        full_source_dataset = TensorDataset(s_rtt, s_rssi, s_labels)
        train_size = int(0.8 * len(full_source_dataset))
        val_size = len(full_source_dataset) - train_size
        print(f"train size = {train_size}, val size = {val_size}")
        source_train_ds, source_val_ds = random_split(
            full_source_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        
        # Target Test 與 Source Test 也獨立切分
        target_test_dataset = TensorDataset(t_rtt, t_rssi, t_labels)
        source_test_dataset = TensorDataset(s_rtt, s_rssi, s_labels)
        # Future Test
        if has_future_test:
            future_test_dataset = TensorDataset(f_rtt, f_rssi, f_labels)

        BATCH_SIZE = 32
        NUM_WORKERS = 0
        train_loader = DataLoader(source_train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS, pin_memory=True)
        val_loader   = DataLoader(source_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        target_test_loader  = DataLoader(target_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        source_test_loader  = DataLoader(source_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        if has_future_test:
            future_test_loader = DataLoader(future_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        
        class_names = label_encoder.classes_
        COORD_TENSOR = create_coord_tensor(class_names, DEVICE)

        # Init Model
        model = BaselineDNN(rtt_dim=RTT_DIM, rssi_dim=RSSI_DIM, num_classes=len(class_names)).to(DEVICE)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        # Training
        num_epochs = 400
        best_val_acc = 0.0 
        best_epoch = 0
        temp_model_name = f"temp_dnn_{COMBO_NAME}_seed{seed}.pth"
        
        for epoch in range(num_epochs):
            model.train()
            # [核心修改 6] 訓練迴圈接收雙 tensor
            for x_rtt_b, x_rssi_b, label_b in train_loader:
                x_rtt_b, x_rssi_b, label_b = x_rtt_b.to(DEVICE), x_rssi_b.to(DEVICE), label_b.to(DEVICE)
                output = model(x_rtt_b, x_rssi_b)
                loss = criterion(output, label_b)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Validation
            val_acc, val_mde, _ = evaluate(model, val_loader, COORD_TENSOR, DEVICE)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                torch.save(model.state_dict(), temp_model_name)

        # Final Test
        if best_epoch != 0:
            model.load_state_dict(torch.load(temp_model_name))
            os.remove(temp_model_name)
        
        # 評估 Target 和 Source
        t_acc, t_mde, t_err = evaluate(model, target_test_loader, COORD_TENSOR, DEVICE, return_all_errors=True)
        s_acc, s_mde, s_err = evaluate(model, source_test_loader, COORD_TENSOR, DEVICE, return_all_errors=True)
        
        print(f"Seed {seed} | Best Epoch {best_epoch} | Tgt MDE: {t_mde:.4f}m")
        
        res_dict = {
            "Combo": COMBO_NAME, "Seed": seed,
            "Source_Acc": s_acc, "Source_MDE": s_mde,
            "Target_Acc": t_acc, "Target_MDE": t_mde
        }

        # 評估未來測試集
        if has_future_test:
            f_acc, f_mde, f_err = evaluate(model, future_test_loader, COORD_TENSOR, DEVICE, return_all_errors=True)
            res_dict["Future_Acc"] = f_acc
            res_dict["Future_MDE"] = f_mde
            print(f"Seed {seed} | Best Epoch {best_epoch} | Tgt MDE: {t_mde:.4f}m | Fut MDE: {f_mde:.4f}m")
            # 儲存 CDF 用的 Error Array
            np.save(os.path.join(CDF_DIR, f"error_{COMBO_NAME}_future_seed{seed}.npy"), f_err)
        else:
            print(f"Seed {seed} | Best Epoch {best_epoch} | Tgt MDE: {t_mde:.4f}m")
        
        results.append(res_dict)
        
        # 儲存 CDF Error Data
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
    
    summary_file = os.path.join(RESULT_DIR, "experiment_summary.csv")
    file_exists = os.path.isfile(summary_file)
    
    with open(summary_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            # 動態建立標頭
            headers = ["Combo", "Avg_Src_Acc", "Avg_Src_Acc_STD", "Avg_Src_MDE", "Avg_Src_MDE_STD", "Avg_Tgt_Acc", "Avg_Tgt_Acc_STD", "Avg_Tgt_MDE", "Avg_Tgt_MDE_STD"]
            if has_future_test:
                headers.extend(["Avg_Fut_Acc", "Avg_Fut_Acc_STD", "Avg_Fut_MDE", "Avg_Fut_MDE_STD"])
            headers.append("Seeds_Detail")
            writer.writerow(headers)
                    
        row_data = [COMBO_NAME, f"{avg_s_acc:.4f}", f"{avg_s_acc_std:.4f}", f"{avg_s_mde:.4f}", f"{avg_s_mde_std:.4f}", f"{avg_t_acc:.4f}", f"{avg_t_acc_std:.4f}", f"{avg_t_mde:.4f}", f"{avg_t_mde_std:.4f}"]
                
        if has_future_test:
            avg_f_acc = df_res['Future_Acc'].mean()
            avg_f_acc_std = df_res['Future_Acc'].std()
            avg_f_mde = df_res['Future_MDE'].mean()
            avg_f_mde_std = df_res['Future_MDE'].std()
            row_data.extend([f"{avg_f_acc:.4f}", f"{avg_f_acc_std:.4f}", f"{avg_f_mde:.4f}", f"{avg_f_mde_std:.4f}"])
            
        row_data.append(str(seed_candidate))
        writer.writerow(row_data)
        
    print(f"Finished Combo {COMBO_NAME}. Results saved.")

if __name__ == '__main__':
    main()