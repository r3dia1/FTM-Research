import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
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

def load_raw_data(csv_path, rtt_cols_to_use=None):
    """只負責讀取與基本清理，絕對不做 Scaling"""
    df = pd.read_csv(csv_path)
    rssi_cols = RSSI_COLS
    rtt_cols = rtt_cols_to_use 
    
    for col in rssi_cols: df[col] = df[col].replace(-100, np.nan)
    for col in rtt_cols: df[col] = df[col].replace([0, -1], np.nan)

    df[rssi_cols] = df[rssi_cols].fillna(-100)
    df[rtt_cols] = df[rtt_cols].fillna(-1)

    rssi_raw = df[rssi_cols].values.astype(np.float32)
    rtt_raw = df[rtt_cols].values.astype(np.float32)
    raw_labels = df['Label'].values
    
    return rssi_raw, rtt_raw, raw_labels

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
        
        # 1. 讀取 Raw Data
        s_rssi_raw, s_rtt_raw, s_labels_raw = load_raw_data(SOURCE_CSV, rtt_cols_to_use=RTT_COLS)
        t_rssi_raw, t_rtt_raw, t_labels_raw = load_raw_data(TARGET_CSV, rtt_cols_to_use=RTT_COLS)

        # 2. 先切分 Source (Train / Val) - 保持類別比例 (Stratify)
        s_tr_idx, s_val_idx = train_test_split(
            np.arange(len(s_labels_raw)), 
            test_size=0.2, 
            random_state=seed, 
            stratify=s_labels_raw
        )

        # 3. 初始化 Scaler (每個 Seed 重新初始化)
        rssi_scaler = MinMaxScaler(feature_range=(-1, 1))
        rtt_scaler = MinMaxScaler(feature_range=(-1, 1))
        label_encoder = LabelEncoder()

        # 4. [防洩漏核心] 只有 Source Train 參與 fit
        rssi_scaler.fit(s_rssi_raw[s_tr_idx])
        rtt_scaler.fit(s_rtt_raw[s_tr_idx])
        label_encoder.fit(s_labels_raw[s_tr_idx])

        # 5. 特徵組合與 Dataset 建立函式
        def create_dataset(rssi, rtt, labels, indices=None):
            # 若有指定 indices，代表是 source，只取部分資料
            if indices is not None:
                r_t = rssi_scaler.transform(rssi[indices])
                rt_t = rtt_scaler.transform(rtt[indices])
                l_t = labels[indices]
            # 若無指定，代表是 Target，全取
            else:
                r_t = rssi_scaler.transform(rssi)
                rt_t = rtt_scaler.transform(rtt)
                l_t = labels
                
            # 根據 mode 組合特徵
            if args.mode == 'fusion':
                x_data = np.concatenate((r_t, rt_t), axis=1)
            elif args.mode == 'rtt':
                x_data = rt_t
            elif args.mode == 'rssi':
                x_data = r_t
                
            try: y_data = label_encoder.transform(l_t)
            except: y_data = np.zeros(len(l_t))
                
            return TensorDataset(torch.tensor(x_data, dtype=torch.float32), torch.tensor(y_data, dtype=torch.long))

        # 6. 建立乾淨的 TensorDataset
        source_train_ds = create_dataset(s_rssi_raw, s_rtt_raw, s_labels_raw, s_tr_idx)
        source_val_ds = create_dataset(s_rssi_raw, s_rtt_raw, s_labels_raw, s_val_idx)
        source_test_dataset = create_dataset(s_rssi_raw, s_rtt_raw, s_labels_raw) # 全量 source 當作參考
        target_test_dataset = create_dataset(t_rssi_raw, t_rtt_raw, t_labels_raw) # 全量 target 當作 test

        # ====================================# 1. 讀取 Raw Data
        s_rssi_raw, s_rtt_raw, s_labels_raw = load_raw_data(SOURCE_CSV, rtt_cols_to_use=RTT_COLS)
        t_rssi_raw, t_rtt_raw, t_labels_raw = load_raw_data(TARGET_CSV, rtt_cols_to_use=RTT_COLS)

        # 2. 先切分 Source (Train / Val) - 保持類別比例 (Stratify)
        s_tr_idx, s_val_idx = train_test_split(
            np.arange(len(s_labels_raw)), 
            test_size=0.2, 
            random_state=seed, 
            stratify=s_labels_raw
        )

        # 3. 初始化 Scaler (每個 Seed 重新初始化)
        rssi_scaler = MinMaxScaler(feature_range=(-1, 1))
        rtt_scaler = MinMaxScaler(feature_range=(-1, 1))
        label_encoder = LabelEncoder()

        # 4. [防洩漏核心] 只有 Source Train 參與 fit
        rssi_scaler.fit(s_rssi_raw[s_tr_idx])
        rtt_scaler.fit(s_rtt_raw[s_tr_idx])
        label_encoder.fit(s_labels_raw[s_tr_idx])

        # 5. 特徵組合與 Dataset 建立函式
        def create_dataset(rssi, rtt, labels, indices=None):
            # 若有指定 indices，代表是 source，只取部分資料
            if indices is not None:
                r_t = rssi_scaler.transform(rssi[indices])
                rt_t = rtt_scaler.transform(rtt[indices])
                l_t = labels[indices]
            # 若無指定，代表是 Target，全取
            else:
                r_t = rssi_scaler.transform(rssi)
                rt_t = rtt_scaler.transform(rtt)
                l_t = labels
                
            # 根據 mode 組合特徵
            if args.mode == 'fusion':
                x_data = np.concatenate((r_t, rt_t), axis=1)
            elif args.mode == 'rtt':
                x_data = rt_t
            elif args.mode == 'rssi':
                x_data = r_t
                
            try: y_data = label_encoder.transform(l_t)
            except: y_data = np.zeros(len(l_t))
                
            return TensorDataset(torch.tensor(x_data, dtype=torch.float32), torch.tensor(y_data, dtype=torch.long))

        # 6. 建立乾淨的 TensorDataset
        source_train_ds = create_dataset(s_rssi_raw, s_rtt_raw, s_labels_raw, s_tr_idx)
        source_val_ds = create_dataset(s_rssi_raw, s_rtt_raw, s_labels_raw, s_val_idx)
        source_test_dataset = create_dataset(s_rssi_raw, s_rtt_raw, s_labels_raw) # 全量 source 當作參考
        target_test_dataset = create_dataset(t_rssi_raw, t_rtt_raw, t_labels_raw) # 全量 target 當作 test

        # ====================================

        BATCH_SIZE = 32
        NUM_WORKERS = 0
        train_loader = DataLoader(source_train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS, pin_memory=True)
        val_loader   = DataLoader(source_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        target_test_loader  = DataLoader(target_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        source_test_loader  = DataLoader(source_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

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