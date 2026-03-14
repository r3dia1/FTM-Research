# ===================== Version Info =============================
# DAFI 架構復現
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
    COMBO_NAME = f"Fusion_DAFI_{'_'.join(rtt_indices)}"
    print(f"Mode: Fusion | Features: RSSI(4) + RTT({len(RTT_COLS)})")

elif args.mode == 'rtt':
    INPUT_DIM = len(RTT_COLS)
    COMBO_NAME = f"DAFI_OnlyRTT_{'_'.join(rtt_indices)}"
    print(f"Mode: Pure RTT | Features: RTT({len(RTT_COLS)})")

elif args.mode == 'rssi':
    INPUT_DIM = len(RSSI_COLS)
    COMBO_NAME = "DAFI_OnlyRSSI_Fixed"
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
# 2. DAFI 模型架構 (遵循論文 Sec 5.1)
# ==========================================
class DAFIModel(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=64):
        super(DAFIModel, self).__init__()
        
        # 特徵提取器 (Feature Extractor theta_f) [cite: 242]
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True)
        )

        # 位置分類器 (Location Classifier theta_c) [cite: 249]
        self.class_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

        # 領域分類器 1 (DC1: Marginal Alignment) [cite: 262, 303]
        self.domain_classifier1 = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 2)
        )

        # 領域分類器 2 (DC2: Conditional Alignment) [cite: 266, 269]
        self.domain_classifier2 = nn.Sequential(
            nn.Linear(hidden_dim + num_classes, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 2)
        )
        
        self.grl = GradientReversalLayer()
    
    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        class_logits = self.class_classifier(features)
        softmax_output = F.softmax(class_logits, dim=1)
        
        # DC1 梯度反轉輸入
        r_feat = self.grl(features, alpha)
        domain_out1 = self.domain_classifier1(r_feat)
        
        # DC2 條件輸入 (特徵 + 位置預測) [cite: 267]
        # 使用 detach 避免 DC2 的梯度直接干擾位置分類器的權重
        cond_input = torch.cat((features, softmax_output.detach()), dim=1)
        r_cond = self.grl(cond_input, alpha)
        domain_out2 = self.domain_classifier2(r_cond)
        
        return class_logits, domain_out1, domain_out2, softmax_output, features

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
    total_sc_loss = 0.0
    total_tc_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for (s_data, s_label), (t_data, _) in zip(source_val_loader, target_val_loader):
            s_data, s_label = s_data.to(device), s_label.to(device)
            t_data = t_data.to(device)
            
            # Forward [cite: 170]
            s_logits, _, _, _, _ = model(s_data, alpha=0)
            _, _, _, t_softmax, _ = model(t_data, alpha=0)
            
            # SC Loss: 源領域位置分類 [cite: 277]
            loss_sc = F.cross_entropy(s_logits, s_label)
            
            # TC Loss: 目標領域熵最小化 [cite: 314]
            loss_tc = -torch.mean(torch.sum(t_softmax * torch.log(t_softmax + 1e-5), dim=1))
            
            total_sc_loss += loss_sc.item()
            total_tc_loss += loss_tc.item()
            num_batches += 1

    if num_batches == 0: return 0, 0
    return total_sc_loss / num_batches, total_tc_loss / num_batches

def evaluate_test(model, data_loader, coord_tensor, device, return_all_errors=False):
    model.eval()
    correct = 0; total = 0; total_dist = 0.0; all_dists = []
    
    with torch.no_grad():
        for x, labels in data_loader:
            x, labels = x.to(device), labels.to(device)
            
            # DAFI 輸出三個值，我們只需要第一個 logits 進行預測 [cite: 201]
            class_out, _, _, _, _ = model(x, alpha=0)
            preds = torch.argmax(class_out, dim=1) # [cite: 327]
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # 計算平均距離誤差 (MDE)
            dist = torch.norm(coord_tensor[preds] - coord_tensor[labels], p=2, dim=1)
            total_dist += dist.sum().item()
            
            if return_all_errors: 
                all_dists.extend(dist.cpu().numpy())
                
    if total == 0: return 0, 0, []
    return 100.*correct/total, total_dist/total, np.array(all_dists)

# ==========================================
# 3. 損失函數計算 (遵循論文 Sec 5.2)
# ==========================================
def calc_dafi_losses(model, s_data, s_label, t_data, alpha, num_classes, device, margin=0.1):
    # Forward pass
    s_logits, s_d1, s_d2, s_softmax, s_feat = model(s_data, alpha=alpha)
    t_logits, t_d1, t_d2, t_softmax, t_feat = model(t_data, alpha=alpha)
    
    # (1) Source Classification (SC) [cite: 277]
    loss_sc = F.cross_entropy(s_logits, s_label)
    
    # (2) Domain Classification (DC = DC1 + DC2) [cite: 300]
    d_label_s = torch.zeros(s_data.size(0), dtype=torch.long).to(device)
    d_label_t = torch.ones(t_data.size(0), dtype=torch.long).to(device)
    
    loss_dc1 = F.cross_entropy(s_d1, d_label_s) + F.cross_entropy(t_d1, d_label_t)
    loss_dc2 = F.cross_entropy(s_d2, d_label_s) + F.cross_entropy(t_d2, d_label_t)
    loss_dc = loss_dc1 + loss_dc2
    
    # (3) Target Classification (TC: Entropy Minimization) [cite: 314]
    loss_tc = -torch.mean(torch.sum(t_softmax * torch.log(t_softmax + 1e-5), dim=1))

    # (4) Class Alignment (CA) - 嚴格遵循論文公式 (12)
    with torch.no_grad():
        pseudo_label_t = torch.argmax(t_softmax, dim=1)

    loss_ca = torch.tensor(0.0).to(device)
    total_triplets = 0

    for c in range(num_classes):
        u_as = s_feat[s_label == c]          # 所有 Anchor (Source label == c)
        u_ps = t_feat[pseudo_label_t == c]   # 所有 Positive (Target pseudo == c)
        u_ns = t_feat[pseudo_label_t != c]   # 所有 Negative (Target pseudo != c)

        if u_as.size(0) > 0 and u_ps.size(0) > 0 and u_ns.size(0) > 0:
            # 論文要求的加總：對於每一組 (a, p, n)
            # 我們利用廣播機制計算所有組合
            # dist_ap: (num_a, num_p), dist_an: (num_a, num_n)
            dist_ap = torch.cdist(u_as, u_ps, p=2).pow(2) 
            dist_an = torch.cdist(u_as, u_ns, p=2).pow(2)

            # 擴展維度以進行對齊計算 (num_a, num_p, num_n)
            # triplet_loss[i, j, k] = max(dist_ap[i, j] - dist_an[i, k] + margin, 0)
            # 這一步會窮舉 Batch 內該類別所有的三元組組合
            triplet_loss = F.relu(dist_ap.unsqueeze(2) - dist_an.unsqueeze(1) + margin)
            
            loss_ca += triplet_loss.sum()
            total_triplets += triplet_loss.numel()

    # 依論文公式 (12) 取平均 (1 / (L * Np))
    if total_triplets > 0:
        loss_ca /= total_triplets

    return loss_sc, loss_dc, loss_tc, loss_ca
    
    # (4) Class Alignment (CA: Center-based) [cite: 337]
    # with torch.no_grad():
    #     pseudo_label_t = torch.argmax(t_softmax, dim=1)
        
    # loss_ca = torch.tensor(0.0).to(device)
    # match_count = 0
    # for c in range(num_classes):
    #     mask_s = (s_label == c)
    #     mask_t = (pseudo_label_t == c)
    #     if mask_s.any() and mask_t.any():
    #         center_s = s_feat[mask_s].mean(0)
    #         center_t = t_feat[mask_t].mean(0)
    #         loss_ca += F.mse_loss(center_t, center_s)
    #         match_count += 1
    # if match_count > 0: loss_ca /= match_count

    # 總損失 (依論文 Sec 5.2.5, 所有 lambda 設為 1) [cite: 341, 344]
    # total_loss = loss_sc + loss_dc + loss_tc + loss_ca
    
    return loss_sc, 2.0 * loss_dc, loss_tc, loss_ca

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
        print(DEVICE)

        SOURCE_CSV = os.path.join(args.base_path, '2026_1_1/all/All_Data_With_RSSI_Diff.csv')
        TARGET_CSV = os.path.join(args.base_path, '2026_2_4/All_Data_With_RSSI_Diff_withoutNA.csv')
        
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
        model = DAFIModel(input_dim=INPUT_DIM, num_classes=len(class_names)).to(DEVICE)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
          
        num_epochs = 300
        best_epoch = -1
        best_adv_score = float('inf')
        
        WARMUP_EPOCHS = 100
        
        temp_model_name = f"temp_dann_{COMBO_NAME}_seed{seed}.pth"

        print(f"Start Training Seed {seed}...")
        print(f"{'Epoch':<6} | {'Total Loss':<10} | {'l_sc':<10} | {'l_dc':<10} | {'l_tc':<10} | {'l_ca':<10} | {'val_l_sc':<12} | {'val_l_dc':<12} | {'Save':<11} | {'Test MDE':<8} | {'Score':<7}")
        print("-" * 135)
        
        for epoch in range(num_epochs):

            total_loss_sum = 0.0
            l_sc_sum = 0.0
            l_dc_sum = 0.0
            l_tc_sum = 0.0
            l_ca_sum = 0.0
            num_batches = 0

            model.train()
            p = float(epoch) / num_epochs
            alpha = 2. / (1. + np.exp(-5 * p)) - 1
            alpha = min(alpha, 1)
            
            for (s_data_b, s_label_b), (t_data_b,_) in zip(source_loader, target_train_loader):
                # --- 核心修正：手動將數據搬到 CUDA ---
                s_data_b = s_data_b.to(DEVICE)
                s_label_b = s_label_b.to(DEVICE)
                t_data_b = t_data_b.to(DEVICE)
                # -----------------------------------
                l_sc, l_dc, l_tc, l_ca = calc_dafi_losses(
                    model, s_data_b, s_label_b, t_data_b, alpha, 
                    len(class_names), DEVICE
                )
                
                # DAFI 總損失函數比例 (預設 lambda 皆為 1) [cite: 341, 344]
                total_loss = l_sc + l_dc + l_tc + l_ca
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                total_loss_sum += total_loss.item()
                l_sc_sum += l_sc.item()
                l_dc_sum += l_dc.item()
                l_tc_sum += l_tc.item()
                l_ca_sum += l_ca.item()
                num_batches += 1
            
            avg_total_loss = total_loss_sum / num_batches
            avg_l_sc = l_sc_sum / num_batches
            avg_l_dc = l_dc_sum / num_batches
            avg_l_tc = l_tc_sum / num_batches
            avg_l_ca = l_ca_sum / num_batches

            # Validation
            val_sc_loss, val_tc_loss = validate_process(model, source_val_loader, target_val_loader, DEVICE)
            current_score = val_sc_loss + val_tc_loss

            save_mark = ""
            if (epoch + 1) > WARMUP_EPOCHS:
                if current_score < best_adv_score:
                    best_adv_score = current_score
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), temp_model_name)
                    save_mark = "(saved)"

            t_acc, t_mde, t_err = evaluate_test(model, target_test_loader, COORD_TENSOR, DEVICE, return_all_errors=True)
            print(f"{epoch+1:<6} | {avg_total_loss:<10.4f} | {avg_l_sc:<10.4f} | {avg_l_dc:<10.4f} | {avg_l_tc:<10.4f} | {avg_l_ca:<10.4f} | {val_sc_loss:<12.4f} | {val_tc_loss:<12.4f} | {save_mark:<11} | {t_mde:<8.4f} | {current_score:<7.3f}")

        # Final Test
        if best_epoch != -1:
            model.load_state_dict(torch.load(temp_model_name))
            os.remove(temp_model_name)
        # model.load_state_dict(torch.load(temp_model_name))
        # os.remove(temp_model_name)

        t_acc, t_mde, t_err = evaluate_test(model, target_test_loader, COORD_TENSOR, DEVICE, return_all_errors=True)
        s_acc, s_mde, s_err = evaluate_test(model, source_test_loader, COORD_TENSOR, DEVICE, return_all_errors=True)
        
        print(f"Seed {seed} | Src Acc: {s_acc:.4f} | Src MDE: {s_mde:.4f} | Tgt Acc: {t_acc:.4f} | Tgt MDE: {t_mde:.4f}")
        
        results.append({
            "Combo": COMBO_NAME,
            "Seed": seed,
            "Source_Acc": s_acc, "Source_MDE": s_mde,
            "Target_Acc": t_acc, "Target_MDE": t_mde
        })
        
        # 儲存 CDF 用的 Error Array
        np.save(os.path.join(CDF_DIR, f"error_{COMBO_NAME}_seed{seed}.npy"), t_err)

    # 寫入 Summary
    df_res = pd.DataFrame(results)
    avg_s_acc = df_res['Source_Acc'].mean()
    avg_s_mde = df_res['Source_MDE'].mean()
    avg_t_acc = df_res['Target_Acc'].mean()
    avg_t_mde = df_res['Target_MDE'].mean()
    
    summary_file = os.path.join(RESULT_DIR, "experiment_summary.csv")
    file_exists = os.path.isfile(summary_file)
    
    with open(summary_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Combo", "Avg_Src_Acc", "Avg_Src_MDE", "Avg_Tgt_Acc", "Avg_Tgt_MDE", "Seeds_Detail"])
        writer.writerow([COMBO_NAME, f"{avg_s_acc:.4f}", f"{avg_s_mde:.4f}", f"{avg_t_acc:.4f}", f"{avg_t_mde:.4f}", str(seed_candidate)])
        
    print(f"Finished Combo {COMBO_NAME}. Results saved.")

if __name__ == '__main__':
    main()