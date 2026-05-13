# ===================== Version Info =============================
# 根據 version 4 FIX 的版本做 MinMaxScaler 的修正(改善資料洩漏)
# target_split_counts = [80, 20, 20]
# target_split_counts = [40, 20, 20] (now)
# ================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
from torch.autograd import Function
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
parser = argparse.ArgumentParser(description='Single Stream DANN Ablation')
parser.add_argument('--rtt_indices', type=str, default="1 2 3 4", help='Space separated indices')
parser.add_argument('--base_path', type=str, default='..', help='Base path for data')
# [新增] 模式選擇參數
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
    COMBO_NAME = f"Fusion_RTT_{'_'.join(rtt_indices)}"
    print(f"Mode: Fusion | Features: RSSI(4) + RTT({len(RTT_COLS)})")

elif args.mode == 'rtt':
    INPUT_DIM = len(RTT_COLS)
    COMBO_NAME = f"Only_RTT_{'_'.join(rtt_indices)}"
    print(f"Mode: Pure RTT | Features: RTT({len(RTT_COLS)})")

elif args.mode == 'rssi':
    INPUT_DIM = len(RSSI_COLS)
    COMBO_NAME = "Only_RSSI_Fixed"
    print(f"Mode: Pure RSSI | Features: RSSI(4)")

print(f"Model Input Dimension: {INPUT_DIM}")

# 建立結果資料夾
RESULT_DIR = "results"
CDF_DIR = os.path.join(RESULT_DIR, "cdf_data")
os.makedirs(CDF_DIR, exist_ok=True)

# ==========================================
# 1. 核心組件：梯度反轉層 (GRL)
# ==========================================
class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()

    def forward(self, x, alpha=1.0):
        return GradientReversalFn.apply(x, alpha)

# ==========================================
# 2. 模型架構：單分支 DANN (Single Stream)
# ==========================================
class SingleStreamDANN(nn.Module):
    def __init__(self, input_dim=4, num_classes=5, hidden_dim=64):
        super(SingleStreamDANN, self).__init__()

        # --- 單一特徵提取器 (動態 Input Dim) ---
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 32), # 這裡會自動調整
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True)
        )

        # --- 標籤分類器 (Task Classifier) ---
        self.class_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

        # --- 單一域分類器 ---
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 2) # Source=0, Target=1
        )
        
        self.grl = GradientReversalLayer()
    
    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        class_output = self.class_classifier(features)
        r_features = self.grl(features, alpha)
        domain_output = self.domain_classifier(r_features)
        return class_output, domain_output

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
    
    # 針對無效值處理
    for col in rssi_cols: df[col] = df[col].replace(-100, np.nan)
    for col in rtt_cols: df[col] = df[col].replace([0, -1], np.nan)

    # 填補缺失值為常數 (根據您註解中說的資料集無 NA，這步只是保險)
    df[rssi_cols] = df[rssi_cols].fillna(-100)
    df[rtt_cols] = df[rtt_cols].fillna(-1)

    rssi_raw = df[rssi_cols].values.astype(np.float32)
    rtt_raw = df[rtt_cols].values.astype(np.float32)
    raw_labels = df['Label'].values
    
    return rssi_raw, rtt_raw, raw_labels

def get_stratified_indices(labels, split_counts):
    """回傳 Train/Val/Test 的 Index，而不是 Dataset"""
    train_idx, val_idx, test_idx = [], [], []
    unique_labels = np.unique(labels)
    n_train, n_val, n_test = split_counts
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        np.random.shuffle(label_indices)
        train_idx.extend(label_indices[:n_train])
        val_idx.extend(label_indices[n_train : n_train + n_val])
        test_idx.extend(label_indices[n_train + n_val : n_train + n_val + n_test])
    return train_idx, val_idx, test_idx

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

def validate_process(model, source_val_loader, target_val_loader, device):
    model.eval()
    total_cls_loss = 0.0
    total_dom_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for (s_data, s_label), (t_data,_) in zip(source_val_loader, target_val_loader):
            s_data, s_label = s_data.to(device), s_label.to(device)
            t_data = t_data.to(device)
            
            class_out_s, d_s = model(s_data, alpha=0) 
            _, d_t = model(t_data, alpha=0)
            
            loss_cls = F.cross_entropy(class_out_s, s_label, reduction='sum')
            d_label_s = torch.zeros(s_data.size(0), dtype=torch.long).to(device)
            d_label_t = torch.ones(t_data.size(0), dtype=torch.long).to(device)
            
            loss_dom_s = F.cross_entropy(d_s, d_label_s, reduction='sum')
            loss_dom_t = F.cross_entropy(d_t, d_label_t, reduction='sum')
            
            total_cls_loss += loss_cls.item()
            total_dom_loss += (loss_dom_s.item() + loss_dom_t.item())
            num_batches += 1

    if num_batches == 0: return 0, 0
    avg_cls = total_cls_loss / (num_batches * source_val_loader.batch_size)
    avg_dom_loss = total_dom_loss / (num_batches * source_val_loader.batch_size * 2) 
    return avg_cls, avg_dom_loss

def evaluate_test(model, data_loader, coord_tensor, device, return_all_errors=False):
    model.eval()
    correct = 0; total = 0; total_dist = 0.0; all_dists = []
    with torch.no_grad():
        for x, labels in data_loader:
            x, labels = x.to(device), labels.to(device)
            class_out, _ = model(x, alpha=0)
            preds = torch.argmax(class_out, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            dist = torch.norm(coord_tensor[preds] - coord_tensor[labels], p=2, dim=1)
            total_dist += dist.sum().item()
            if return_all_errors: all_dists.extend(dist.cpu().numpy())
    if total == 0: return 0, 0, []
    return 100.*correct/total, total_dist/total, np.array(all_dists)

# ==========================================
# 3. 主程式 (部分修改)
# ==========================================
def main():
    # ... [前段設定保持不變] ...
    results = []
    seed_candidate = [42, 6767, 123456]
    
    for seed in seed_candidate:
        set_seed(seed)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        SOURCE_CSV = os.path.join(args.base_path, '2026_1_1/all/All_Data_With_RSSI_Diff.csv')
        TARGET_CSV = os.path.join(args.base_path, '2026_2_4/All_Data_With_RSSI_Diff_withoutNA.csv')
        # TARGET_CSV = os.path.join(args.base_path, '2026_3_17/All_Data_With_RSSI_Diff_withoutNA.csv')
        # TARGET_CSV = os.path.join(args.base_path, '2026_4_1/All_Data_With_RSSI_Diff_withoutNA.csv')
        
        # 讀取 Raw Data
        s_rssi_raw, s_rtt_raw, s_labels_raw = load_raw_data(SOURCE_CSV, rtt_cols_to_use=RTT_COLS)
        t_rssi_raw, t_rtt_raw, t_labels_raw = load_raw_data(TARGET_CSV, rtt_cols_to_use=RTT_COLS)

        # 1. 取得切割索引
        source_split_counts = [80, 20, 20] 
        target_split_counts = [40, 20, 20]
        s_tr_idx, s_val_idx, s_test_idx = get_stratified_indices(s_labels_raw, source_split_counts)
        t_tr_idx, t_val_idx, t_test_idx = get_stratified_indices(t_labels_raw, target_split_counts)

        # 2. 初始化 Scaler (每個 Seed 重新初始化，保證乾淨)
        rssi_scaler = MinMaxScaler(feature_range=(-1, 1))
        rtt_scaler = MinMaxScaler(feature_range=(-1, 1))
        label_encoder = LabelEncoder()

        # 3. [關鍵] 只有 Source Train 參與 fit !
        rssi_scaler.fit(s_rssi_raw[s_tr_idx])
        rtt_scaler.fit(s_rtt_raw[s_tr_idx])
        label_encoder.fit(s_labels_raw[s_tr_idx])

        # 4. 改寫的 create_dataset: 負責縮放並依據 mode 合併特徵為「單一 Tensor」
        def create_dataset(rssi, rtt, labels, indices):
            r_t = rssi_scaler.transform(rssi[indices])
            rt_t = rtt_scaler.transform(rtt[indices])
            
            # [核心修改] 在這裡根據 mode 組合特徵
            if args.mode == 'fusion':
                x_data = np.concatenate((r_t, rt_t), axis=1) # 變成 (Batch, RSSI_DIM + RTT_DIM)
            elif args.mode == 'rtt':
                x_data = rt_t
            elif args.mode == 'rssi':
                x_data = r_t
                
            # 防止 target 出現未知的 label 報錯
            try: 
                l_t = label_encoder.transform(labels[indices])
            except: 
                l_t = np.zeros(len(indices))
                
            # 只回傳 (X, Y) 兩個變數，對應單分支
            return TensorDataset(torch.tensor(x_data, dtype=torch.float32), torch.tensor(l_t, dtype=torch.long))

        # 5. 建立 TensorDataset
        s_train = create_dataset(s_rssi_raw, s_rtt_raw, s_labels_raw, s_tr_idx)
        s_val = create_dataset(s_rssi_raw, s_rtt_raw, s_labels_raw, s_val_idx)
        s_test = create_dataset(s_rssi_raw, s_rtt_raw, s_labels_raw, s_test_idx)
        
        t_train = create_dataset(t_rssi_raw, t_rtt_raw, t_labels_raw, t_tr_idx)
        t_val = create_dataset(t_rssi_raw, t_rtt_raw, t_labels_raw, t_val_idx)
        t_test = create_dataset(t_rssi_raw, t_rtt_raw, t_labels_raw, t_test_idx)
        # ====================================

        BATCH_SIZE = 32
        source_loader = DataLoader(s_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        target_train_loader = DataLoader(t_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        source_val_loader = DataLoader(s_val, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
        target_val_loader = DataLoader(t_val, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
        source_test_loader = DataLoader(s_test, batch_size=BATCH_SIZE, shuffle=False)
        target_test_loader = DataLoader(t_test, batch_size=BATCH_SIZE, shuffle=False)

        class_names = label_encoder.classes_
        COORD_TENSOR = create_coord_tensor(class_names, DEVICE)

        # 建立模型 (傳入正確的 INPUT_DIM)
        model = SingleStreamDANN(input_dim=INPUT_DIM, num_classes=len(class_names)).to(DEVICE)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
        
        W_CLS = 1; W_DOM = 1    
        num_epochs = 400
        best_epoch = -1
        best_adv_score = float('-inf')
        
        WARMUP_EPOCHS = 10
        CLS_THRESHOLD = 0.5 
        W_SCORE_CLS = 0.3
        W_SCORE_DOM = 0.1
        
        temp_model_name = f"temp_dann_{COMBO_NAME}_seed{seed}.pth"

        print(f"Start Training Seed {seed}...")
        
        for epoch in range(num_epochs):
            model.train()
            p = float(epoch) / num_epochs
            alpha = 2. / (1. + np.exp(-5 * p)) - 1
            alpha = min(alpha, 0.3)
            
            for (s_data_b, s_label_b), (t_data_b,_) in zip(source_loader, target_train_loader):
                s_data_b, s_label_b = s_data_b.to(DEVICE), s_label_b.to(DEVICE)
                t_data_b = t_data_b.to(DEVICE)
                
                class_out, d_out_s = model(s_data_b, alpha=alpha)
                _, d_out_t = model(t_data_b, alpha=alpha)
                
                loss_class = F.cross_entropy(class_out, s_label_b)
                d_label_s = torch.zeros(s_data_b.size(0), dtype=torch.long).to(DEVICE)
                d_label_t = torch.ones(t_data_b.size(0), dtype=torch.long).to(DEVICE)
                loss_d = F.cross_entropy(d_out_s, d_label_s) + F.cross_entropy(d_out_t, d_label_t)
                loss = (W_CLS * loss_class) + (W_DOM * loss_d)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation
            val_cls, val_d_loss = validate_process(model, source_val_loader, target_val_loader, DEVICE)
            current_score = (W_SCORE_DOM * val_d_loss) - (W_SCORE_CLS * val_cls)

            if (epoch + 1) > WARMUP_EPOCHS and val_cls < CLS_THRESHOLD:
                if current_score > best_adv_score:
                    best_adv_score = current_score
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), temp_model_name)

        # Final Test
        if best_epoch != -1:
            model.load_state_dict(torch.load(temp_model_name))
            os.remove(temp_model_name)

        t_acc, t_mde, t_err = evaluate_test(model, target_test_loader, COORD_TENSOR, DEVICE, return_all_errors=True)
        s_acc, s_mde, s_err = evaluate_test(model, source_test_loader, COORD_TENSOR, DEVICE, return_all_errors=True)
        
        print(f"Seed {seed} | Src MDE: {s_mde:.4f} | Tgt MDE: {t_mde:.4f}")
        
        results.append({
            "Combo": COMBO_NAME,
            "Seed": seed,
            "Source_Acc": s_acc, "Source_MDE": s_mde,
            "Target_Acc": t_acc, "Target_MDE": t_mde
        })
        
        # 儲存 CDF 用的 Error Array
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
            writer.writerow(["Combo", "Avg_Src_Acc", "Avg_Src_Acc_STD", "Avg_Src_MDE", "Avg_Src_MDE_STD", "Avg_Tgt_Acc", "Avg_Tgt_Acc_STD", "Avg_Tgt_MDE", "Avg_Tgt_MDE_STD", "Seeds_Detail"])
        writer.writerow([COMBO_NAME, f"{avg_s_acc:.4f}", f"{avg_s_acc_std:.4f}", f"{avg_s_mde:.4f}", f"{avg_s_mde_std:.4f}", f"{avg_t_acc:.4f}", f"{avg_t_acc_std:.4f}", f"{avg_t_mde:.4f}", f"{avg_t_mde_std:.4f}", str(seed_candidate)])
        
    print(f"Finished Combo {COMBO_NAME}. Results saved.")

if __name__ == '__main__':
    main()