# ===================== Version Info =============================
# 修改自 version 4 fix
# 原本的 load_wifi_data 在做 fit transform 的時候
# 沒有區隔 train/val/test，現已修正
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
parser = argparse.ArgumentParser(description='Dual Stream DANN Ablation')
parser.add_argument('--rtt_indices', type=str, required=True, help='Space separated indices (e.g., "1 2 4")')
parser.add_argument('--base_path', type=str, default='..', help='Base path for data')
args = parser.parse_args()

# 解析 RTT Columns
rtt_indices = args.rtt_indices.strip().split()
RTT_COLS = [f'Dist_mm_{i}' for i in rtt_indices]

# 固定 RSSI 輸入 (根據您的需求)
# RSSI_COLS = ['Diff_RSSI_1_2', 'Diff_RSSI_1_3', 'Diff_RSSI_1_4', 'Diff_RSSI_2_3', 'Diff_RSSI_2_4', 'Diff_RSSI_3_4']
RSSI_COLS = ['RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4'] 

# 計算輸入維度
RTT_INPUT_DIM = len(RTT_COLS)
RSSI_INPUT_DIM = len(RSSI_COLS)
COMBO_NAME = "FusionDANN_" + "_".join(rtt_indices)

print(f"==========================================")
print(f"Experiment: {COMBO_NAME}")
print(f"RSSI Features ({RSSI_INPUT_DIM}): Fixed")
print(f"RTT Features ({RTT_INPUT_DIM}): {RTT_COLS}")
print(f"==========================================")

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
# 2. 模型架構：雙分支 DANN (動態維度)
# ==========================================
class DualStreamDANN(nn.Module):
    def __init__(self, rssi_dim=4, rtt_dim=3, num_classes=49, hidden_dim=64):
        super(DualStreamDANN, self).__init__()

        # --- 分支 B: RTT 特徵提取器 (動態輸入) ---
        self.rtt_extractor = nn.Sequential(
            nn.Linear(rtt_dim, 32), # [修改] 使用傳入的 rtt_dim
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True)
        )

        # --- 分支 A: RSSI 特徵提取器 ---
        self.rssi_extractor = nn.Sequential(
            nn.Linear(rssi_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True)
        )

        # --- 融合後的標籤分類器 ---
        self.class_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

        # --- 域分類器 ---
        self.domain_classifier_rssi = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.BatchNorm1d(32), nn.ReLU(True),
            nn.Linear(32, 2)
        )
        self.domain_classifier_rtt = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.BatchNorm1d(32), nn.ReLU(True),
            nn.Linear(32, 2)
        )
        
        self.grl = GradientReversalLayer()
    
    def forward(self, rssi, rtt, alpha=1.0):
        # 1. 提取特徵
        f_rssi = self.rssi_extractor(rssi)
        f_rtt = self.rtt_extractor(rtt)

        # 2. 標籤預測 (Concatenate)
        f_cat = torch.cat((f_rssi, f_rtt), dim=1)
        class_output = self.class_classifier(f_cat)

        # 3. 域預測
        r_rssi = self.grl(f_rssi, alpha)
        domain_output_rssi = self.domain_classifier_rssi(r_rssi)

        r_rtt = self.grl(f_rtt, alpha)
        domain_output_rtt = self.domain_classifier_rtt(r_rtt)

        return class_output, domain_output_rssi, domain_output_rtt

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

def fill_nan_with_train_mean(train_data, val_data, test_data, train_labels, val_labels, test_labels, fallback_val):
    """
    嚴格使用 Train Set 的類別平均值來填補 Train/Val/Test 中的 NaN。
    """
    # 複製資料，避免改動到原始全域變數
    t_data = np.copy(train_data)
    v_data = np.copy(val_data)
    te_data = np.copy(test_data)
    
    unique_labels = np.unique(train_labels)
    
    # 1. 計算 Global Mean (作為最後的備案：如果某個類別在 Train Set 裡某特徵全為 NaN)
    with np.errstate(invalid='ignore'):
        global_mean = np.nanmean(t_data, axis=0)
    # 如果連 Global Mean 都是 NaN (整個特徵壞掉)，就用常數 (如 -100 或 -1)
    global_mean = np.nan_to_num(global_mean, nan=fallback_val)
    
    # 2. 建立 Train Set 的類別平均值字典
    class_means = {}
    for label in unique_labels:
        mask = (train_labels == label)
        class_data = t_data[mask]
        
        with np.errstate(invalid='ignore'):
            c_mean = np.nanmean(class_data, axis=0)
        
        # 針對該類別全為 NaN 的特徵，使用 Global Mean 補救
        c_mean_nan_mask = np.isnan(c_mean)
        c_mean[c_mean_nan_mask] = global_mean[c_mean_nan_mask]
        class_means[label] = c_mean

    # 3. 定義填補邏輯
    def apply_imputation(data, labels):
        for i in range(len(data)):
            label = labels[i]
            nan_mask = np.isnan(data[i])
            if np.any(nan_mask): # 如果這筆資料有缺值
                # 取得該類別的平均值 (若遇到 Train 沒看過的奇葩 Label，用 Global Mean)
                fill_values = class_means.get(label, global_mean)
                data[i][nan_mask] = fill_values[nan_mask]
        return data

    return apply_imputation(t_data, train_labels), \
           apply_imputation(v_data, val_labels), \
           apply_imputation(te_data, test_labels)

# 座標映射
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
    total_dom_rssi = 0.0
    total_dom_rtt = 0.0
    num_batches = 0
    with torch.no_grad():
        for (s_rssi, s_rtt, s_label), (t_rssi, t_rtt, _) in zip(source_val_loader, target_val_loader):
            s_rssi, s_rtt, s_label = s_rssi.to(device), s_rtt.to(device), s_label.to(device)
            t_rssi, t_rtt = t_rssi.to(device), t_rtt.to(device)
            
            class_out_s, d_rssi_s, d_rtt_s = model(s_rssi, s_rtt, alpha=0) 
            _, d_rssi_t, d_rtt_t = model(t_rssi, t_rtt, alpha=0)
            
            loss_cls = F.cross_entropy(class_out_s, s_label, reduction='sum')
            d_label_s = torch.zeros(s_rssi.size(0), dtype=torch.long).to(device)
            d_label_t = torch.ones(t_rssi.size(0), dtype=torch.long).to(device)
            
            loss_dom_rssi = F.cross_entropy(d_rssi_s, d_label_s, reduction='sum') + F.cross_entropy(d_rssi_t, d_label_t, reduction='sum')
            loss_dom_rtt = F.cross_entropy(d_rtt_s, d_label_s, reduction='sum') + F.cross_entropy(d_rtt_t, d_label_t, reduction='sum')
            
            total_cls_loss += loss_cls.item()
            total_dom_rssi += loss_dom_rssi.item()
            total_dom_rtt += loss_dom_rtt.item()
            num_batches += 1

    if num_batches == 0: return 0, 0, 0
    avg_cls = total_cls_loss / (num_batches * source_val_loader.batch_size)
    avg_dom_rssi = total_dom_rssi / (num_batches * source_val_loader.batch_size * 2)
    avg_dom_rtt = total_dom_rtt / (num_batches * source_val_loader.batch_size * 2)
    return avg_cls, avg_dom_rssi, avg_dom_rtt

def evaluate_test(model, data_loader, coord_tensor, device, return_all_errors=False):
    model.eval()
    correct = 0; total = 0; total_dist = 0.0; all_dists = []
    with torch.no_grad():
        for rssi, rtt, labels in data_loader:
            rssi, rtt, labels = rssi.to(device), rtt.to(device), labels.to(device)
            class_out, _, _ = model(rssi, rtt, alpha=0)
            preds = torch.argmax(class_out, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            dist = torch.norm(coord_tensor[preds] - coord_tensor[labels], p=2, dim=1)
            total_dist += dist.sum().item()
            if return_all_errors: all_dists.extend(dist.cpu().numpy())
    if total == 0: return 0, 0, []
    return 100.*correct/total, total_dist/total, np.array(all_dists)

# ==========================================
# 3. 主程式
# ==========================================
def main():
    results = []
    seed_candidate = [42, 6767, 123456]
    
    for seed in seed_candidate:
        set_seed(seed)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 設定路徑
        SOURCE_CSV = os.path.join(args.base_path, '2026_1_1/all/All_Data_With_RSSI_Diff.csv')
        TARGET_CSV = os.path.join(args.base_path, '2026_2_4/All_Data_With_RSSI_Diff_withoutNA.csv')

        SAMPLES_PER_LABEL = 120
        # Load Data
        s_rssi_raw, s_rtt_raw, s_labels_raw = load_raw_data(SOURCE_CSV, rtt_cols_to_use=RTT_COLS)
        t_rssi_raw, t_rtt_raw, t_labels_raw = load_raw_data(TARGET_CSV, rtt_cols_to_use=RTT_COLS)
        
        # 1. 取得切割索引
        source_split_counts = [80, 20, 20] 
        target_split_counts = [40, 20, 20]
        s_tr_idx, s_val_idx, s_test_idx = get_stratified_indices(s_labels_raw, source_split_counts)
        t_tr_idx, t_val_idx, t_test_idx = get_stratified_indices(t_labels_raw, target_split_counts)

        # 處理 Source RSSI (常數備案為 -100)
        s_rssi_train, s_rssi_val, s_rssi_test = fill_nan_with_train_mean(
            s_rssi_raw[s_tr_idx], s_rssi_raw[s_val_idx], s_rssi_raw[s_test_idx],
            s_labels_raw[s_tr_idx], s_labels_raw[s_val_idx], s_labels_raw[s_test_idx], fallback_val=-100.0)
        
        # 處理 Source RTT (常數備案為 -1)
        s_rtt_train, s_rtt_val, s_rtt_test = fill_nan_with_train_mean(
            s_rtt_raw[s_tr_idx], s_rtt_raw[s_val_idx], s_rtt_raw[s_test_idx],
            s_labels_raw[s_tr_idx], s_labels_raw[s_val_idx], s_labels_raw[s_test_idx], fallback_val=-1.0)

        # t_rssi_train = t_rssi_raw[t_tr_idx]
        # t_rssi_val   = t_rssi_raw[t_val_idx]
        # t_rssi_test  = t_rssi_raw[t_test_idx]

        # t_rtt_train  = t_rtt_raw[t_tr_idx]
        # t_rtt_val    = t_rtt_raw[t_val_idx]
        # t_rtt_test   = t_rtt_raw[t_test_idx]
        
        # 處理 Target RSSI
        t_rssi_train, t_rssi_val, t_rssi_test = fill_nan_with_train_mean(
            t_rssi_raw[t_tr_idx], t_rssi_raw[t_val_idx], t_rssi_raw[t_test_idx],
            t_labels_raw[t_tr_idx], t_labels_raw[t_val_idx], t_labels_raw[t_test_idx], fallback_val=-100.0)

        # 處理 Target RTT
        t_rtt_train, t_rtt_val, t_rtt_test = fill_nan_with_train_mean(
            t_rtt_raw[t_tr_idx], t_rtt_raw[t_val_idx], t_rtt_raw[t_test_idx],
            t_labels_raw[t_tr_idx], t_labels_raw[t_val_idx], t_labels_raw[t_test_idx], fallback_val=-1.0)

        # 初始化 Scaler 與 LabelEncoder
        rssi_scaler = MinMaxScaler(feature_range=(-1, 1))
        rtt_scaler = MinMaxScaler(feature_range=(-1, 1))
        label_encoder = LabelEncoder()

        # 針對 s_labels_raw[s_tr_idx] 進行 fit 的做法
        label_encoder.fit(s_labels_raw[s_tr_idx])

        # 只有 Source Train 參與 fit !
        rssi_scaler.fit(s_rssi_train)
        rtt_scaler.fit(s_rtt_train)

        def create_dataset(rssi_split, rtt_split, labels_split):
            r_t = rssi_scaler.transform(rssi_split)
            rt_t = rtt_scaler.transform(rtt_split)
            # 保留原有的容錯機制
            try: 
                l_t = label_encoder.transform(labels_split)
            except: 
                l_t = np.zeros(len(labels_split))
            return TensorDataset(torch.tensor(r_t), torch.tensor(rt_t), torch.tensor(l_t, dtype=torch.long))

        s_train = create_dataset(s_rssi_train, s_rtt_train, s_labels_raw[s_tr_idx])
        s_val   = create_dataset(s_rssi_val, s_rtt_val, s_labels_raw[s_val_idx])
        s_test  = create_dataset(s_rssi_test, s_rtt_test, s_labels_raw[s_test_idx])
        
        t_train = create_dataset(t_rssi_train, t_rtt_train, t_labels_raw[t_tr_idx])
        t_val   = create_dataset(t_rssi_val, t_rtt_val, t_labels_raw[t_val_idx])
        t_test  = create_dataset(t_rssi_test, t_rtt_test, t_labels_raw[t_test_idx])

        BATCH_SIZE = 32
        source_loader = DataLoader(s_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        target_train_loader = DataLoader(t_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        source_val_loader = DataLoader(s_val, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
        target_val_loader = DataLoader(t_val, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
        source_test_loader = DataLoader(s_test, batch_size=BATCH_SIZE, shuffle=False)
        target_test_loader = DataLoader(t_test, batch_size=BATCH_SIZE, shuffle=False)

        class_names = label_encoder.classes_
        COORD_TENSOR = create_coord_tensor(class_names, DEVICE)

        # [核心] 初始化模型，傳入動態 RTT 維度
        model = DualStreamDANN(rssi_dim=RSSI_INPUT_DIM, rtt_dim=RTT_INPUT_DIM, num_classes=len(class_names)).to(DEVICE)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
        
        W_CLS = 1; W_DOM_RSSI = 1; W_DOM_RTT = 1    
        num_epochs = 400
        best_epoch = -1
        best_adv_score = float('-inf')
        
        WARMUP_EPOCHS = 10
        CLS_THRESHOLD = 0.5 
        # W_SCORE_CLS = 0.3
        # W_SCORE_DOM_1 = 0.1
        # W_SCORE_DOM_2 = 0.3
        W_SCORE_CLS = 1
        W_SCORE_DOM_1 = 1
        W_SCORE_DOM_2 = 1

        temp_model_name = f"temp_dual_{COMBO_NAME}_seed{seed}.pth"

        print(f"Start Training Seed {seed}...")
        print(f"\nStart Training... (Validate on Source Val & Target Val)")
        print(f"{'Epoch':<6} | {'Total Loss':<10} | {'Train Cls':<10} | {'Train D_RSSI':<12} | {'Train D_RTT':<11} | {'Val Cls':<10} | {'Val D_RSSI':<10} | {'Val D_RTT':<10} | {'Save':<11} | {'Test MDE':<8} | {'Score':<7}")
        print("-" * 135)
        
        for epoch in range(num_epochs):
            model.train()
            total_loss_sum = 0.0
            train_cls_sum = 0
            dom_rssi_sum = 0.0
            dom_rtt_sum = 0.0
            num_batches = 0

            p = float(epoch) / num_epochs
            alpha = 2. / (1. + np.exp(-5 * p)) - 1
            alpha = min(alpha, 0.3)
            
            for (s_rssi, s_rtt, s_lbl), (t_rssi, t_rtt, _) in zip(source_loader, target_train_loader):
                s_rssi, s_rtt, s_lbl = s_rssi.to(DEVICE), s_rtt.to(DEVICE), s_lbl.to(DEVICE)
                t_rssi, t_rtt = t_rssi.to(DEVICE), t_rtt.to(DEVICE)
                
                cls_out, d_rssi_s, d_rtt_s = model(s_rssi, s_rtt, alpha=alpha)
                _, d_rssi_t, d_rtt_t = model(t_rssi, t_rtt, alpha=alpha)
                
                l_cls = F.cross_entropy(cls_out, s_lbl)
                d_lbl_s = torch.zeros(s_rssi.size(0), dtype=torch.long).to(DEVICE)
                d_lbl_t = torch.ones(t_rssi.size(0), dtype=torch.long).to(DEVICE)
                l_d_rssi = F.cross_entropy(d_rssi_s, d_lbl_s) + F.cross_entropy(d_rssi_t, d_lbl_t)
                l_d_rtt = F.cross_entropy(d_rtt_s, d_lbl_s) + F.cross_entropy(d_rtt_t, d_lbl_t)
                loss = (W_CLS * l_cls) + (W_DOM_RSSI * l_d_rssi) + (W_DOM_RTT * l_d_rtt)
                
                optimizer.zero_grad(); loss.backward(); optimizer.step()

                total_loss_sum += loss.item()
                train_cls_sum += l_cls.item()
                dom_rssi_sum += l_d_rssi.item()
                dom_rtt_sum += l_d_rtt.item()
                num_batches += 1

            # 計算 Training Avg Loss
            avg_total = total_loss_sum / num_batches
            avg_cls = train_cls_sum / num_batches
            avg_d_rssi = dom_rssi_sum / num_batches
            avg_d_rtt = dom_rtt_sum / num_batches

            # Validation
            val_cls, val_d_rssi, val_d_rtt = validate_process(model, source_val_loader, target_val_loader, DEVICE)
            
            current_dom_total = (val_d_rssi + val_d_rtt - 1.2)
            current_dom_diff = abs(val_d_rssi - val_d_rtt)
            current_score = (W_SCORE_DOM_1 * current_dom_total) - (W_SCORE_CLS * val_cls) - (W_SCORE_DOM_2 * current_dom_diff)

            save_mark = ""
            if (epoch + 1) > WARMUP_EPOCHS and val_cls < CLS_THRESHOLD:
                if current_score > best_adv_score:
                    best_adv_score = current_score
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), temp_model_name)

            t_acc, t_mde, t_err = evaluate_test(model, target_test_loader, COORD_TENSOR, DEVICE, return_all_errors=True)

            print(f"{epoch+1:<6} | {avg_total:<10.4f} | {avg_cls:<10.4f} | {avg_d_rssi:<12.4f} | {avg_d_rtt:<11.4f} | {val_cls:<10.4f} | {val_d_rssi:<10.4f} | {val_d_rtt:<10.4f} | {save_mark:<11} | {t_mde:<8.4f} | {current_score:<7.3f}")


        if best_epoch != -1:
            model.load_state_dict(torch.load(temp_model_name))
            os.remove(temp_model_name)
        
        t_acc, t_mde, t_err = evaluate_test(model, target_test_loader, COORD_TENSOR, DEVICE, return_all_errors=True)
        s_acc, s_mde, s_err = evaluate_test(model, source_test_loader, COORD_TENSOR, DEVICE, return_all_errors=True)
        
        print(f"Seed {seed} | Src MDE: {s_mde:.4f} | Tgt MDE: {t_mde:.4f}")
        
        results.append({
            "Combo": COMBO_NAME, "Seed": seed,
            "Source_Acc": s_acc, "Source_MDE": s_mde,
            "Target_Acc": t_acc, "Target_MDE": t_mde
        })
        
        # 儲存 CDF Error Data
        np.save(os.path.join(CDF_DIR, f"error_{COMBO_NAME}_seed{seed}.npy"), t_err)

    # 寫入 Summary CSV
    df_res = pd.DataFrame(results)
    avg_s_acc = df_res['Source_Acc'].mean()
    avg_s_mde = df_res['Source_MDE'].mean()
    avg_t_acc = df_res['Target_Acc'].mean()
    avg_t_mde = df_res['Target_MDE'].mean()
    
    summary_file = os.path.join(RESULT_DIR, "dual_experiment_summary.csv")
    file_exists = os.path.isfile(summary_file)
    with open(summary_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Combo", "Avg_Src_Acc", "Avg_Src_MDE", "Avg_Tgt_Acc", "Avg_Tgt_MDE", "Seeds_Detail"])
        writer.writerow([COMBO_NAME, f"{avg_s_acc:.4f}", f"{avg_s_mde:.4f}", f"{avg_t_acc:.4f}", f"{avg_t_mde:.4f}", str(seed_candidate)])
    print(f"Finished {COMBO_NAME}. Results saved.")

if __name__ == '__main__':
    main()