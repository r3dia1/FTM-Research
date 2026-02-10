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
args = parser.parse_args()

# 解析 Columns
rtt_indices = args.rtt_indices.strip().split()
RTT_COLS = [f'Dist_mm_{i}' for i in rtt_indices]
RSSI_COLS = ['RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4'] 

# [核心修改] 根據模式決定 INPUT_DIM 與 COMBO_NAME
if args.mode == 'fusion':
    INPUT_DIM = len(RSSI_COLS) + len(RTT_COLS)
    COMBO_NAME = f"DNN_Fusion_RTT_{'_'.join(rtt_indices)}"
elif args.mode == 'rtt':
    INPUT_DIM = len(RTT_COLS)
    COMBO_NAME = f"DNN_Only_RTT_{'_'.join(rtt_indices)}"
elif args.mode == 'rssi':
    INPUT_DIM = len(RSSI_COLS)
    COMBO_NAME = "DNN_Only_RSSI_Fixed"

print(f"==========================================")
print(f"Experiment: {COMBO_NAME}")
print(f"Mode: {args.mode} | Input Dim: {INPUT_DIM}")
print(f"Features: RSSI({len(RSSI_COLS)}) + RTT({len(RTT_COLS)})" if args.mode == 'fusion' else f"Features: {args.mode.upper()}")
print(f"==========================================")

# 建立結果資料夾
RESULT_DIR = "results"
CDF_DIR = os.path.join(RESULT_DIR, "cdf_data")
os.makedirs(CDF_DIR, exist_ok=True)

# ==========================================
# 1. 模型架構：標準 DNN
# ==========================================
class BaselineDNN(nn.Module):
    def __init__(self, input_dim=4, num_classes=5, hidden_dim=64):
        super(BaselineDNN, self).__init__()

        # --- 特徵提取器 ---
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 32), # 動態調整輸入層
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True)
        )

        # --- 標籤分類器 ---
        self.class_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        class_output = self.class_classifier(features)
        return class_output

# ==========================================
# 資料處理
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
    # 使用動態傳入的 RTT columns
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

# 座標映射 (保持不變)
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
        for x_b, labels_b in data_loader:
            x_b, labels_b = x_b.to(device), labels_b.to(device)
            class_out = model(x_b)
            
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
        # 請確保這裡的路徑對應到您存放資料的結構，或從外部 args 傳入
        SOURCE_CSV = os.path.join(args.base_path, '2026_1_1/all/All_Data_With_RSSI_Diff.csv')
        TARGET_CSV = os.path.join(args.base_path, '2026_1_14/All_Data_With_RSSI_Diff_withoutNA.csv')

        SAMPLES_PER_CLASS = 120 # 根據需求調整
        
        # Load Data
        s_rssi, s_rtt, s_labels = load_wifi_data(SOURCE_CSV, is_source=True, samples_per_label=SAMPLES_PER_CLASS, rtt_cols_to_use=RTT_COLS)
        t_rssi, t_rtt, t_labels = load_wifi_data(TARGET_CSV, is_source=False, samples_per_label=SAMPLES_PER_CLASS, rtt_cols_to_use=RTT_COLS)

        # [核心修改] 根據模式準備輸入資料
        if args.mode == 'fusion':
            s_data = torch.cat((s_rssi, s_rtt), dim=1)
            t_data = torch.cat((t_rssi, t_rtt), dim=1)
        elif args.mode == 'rtt':
            s_data = s_rtt
            t_data = t_rtt
        elif args.mode == 'rssi':
            s_data = s_rssi
            t_data = t_rssi

        # Source Split (Train/Val)
        full_source_dataset = TensorDataset(s_data, s_labels)
        train_size = int(0.8 * len(full_source_dataset))
        val_size = len(full_source_dataset) - train_size
        source_train_ds, source_val_ds = random_split(
            full_source_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        
        # Target Test
        target_test_dataset = TensorDataset(t_data, t_labels)
        # Source Test (Optional, to check source performance)
        source_test_dataset = TensorDataset(s_data, s_labels)

        BATCH_SIZE = 32
        train_loader = DataLoader(source_train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader   = DataLoader(source_val_ds, batch_size=BATCH_SIZE, shuffle=False)
        target_test_loader  = DataLoader(target_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        source_test_loader  = DataLoader(source_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        class_names = label_encoder.classes_
        COORD_TENSOR = create_coord_tensor(class_names, DEVICE)

        # Init Model
        model = BaselineDNN(input_dim=INPUT_DIM, num_classes=len(class_names)).to(DEVICE)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        # Training
        num_epochs = 400
        best_val_acc = 0.0 
        best_epoch = 0
        temp_model_name = f"temp_dnn_{COMBO_NAME}_seed{seed}.pth"
        
        for epoch in range(num_epochs):
            model.train()
            for x_b, label_b in train_loader:
                x_b, label_b = x_b.to(DEVICE), label_b.to(DEVICE)
                output = model(x_b)
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
        s_acc, s_mde, s_err = evaluate(model, source_test_loader, COORD_TENSOR, DEVICE, return_all_errors=True) # 使用全量 source 作為參考
        
        print(f"Seed {seed} | Best Epoch {best_epoch} | Tgt MDE: {t_mde:.4f}m")
        
        results.append({
            "Combo": COMBO_NAME, "Seed": seed,
            "Source_Acc": s_acc, "Source_MDE": s_mde,
            "Target_Acc": t_acc, "Target_MDE": t_mde
        })
        
        # 儲存 CDF Error Data
        np.save(os.path.join(CDF_DIR, f"error_{COMBO_NAME}_seed{seed}.npy"), t_err)

    # Summary
    df_res = pd.DataFrame(results)
    avg_s_acc = df_res['Source_Acc'].mean()
    avg_s_mde = df_res['Source_MDE'].mean()
    avg_t_acc = df_res['Target_Acc'].mean()
    avg_t_mde = df_res['Target_MDE'].mean()
    
    summary_file = os.path.join(RESULT_DIR, "dnn_experiment_summary.csv")
    file_exists = os.path.isfile(summary_file)
    with open(summary_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Combo", "Avg_Src_Acc", "Avg_Src_MDE", "Avg_Tgt_Acc", "Avg_Tgt_MDE", "Seeds_Detail"])
        writer.writerow([COMBO_NAME, f"{avg_s_acc:.4f}", f"{avg_s_mde:.4f}", f"{avg_t_acc:.4f}", f"{avg_t_mde:.4f}", str(seed_candidate)])

    print(f"Finished {COMBO_NAME}. Results saved.")

if __name__ == '__main__':
    main()