# ===================== Version Info =============================
# 修改自 version 3 fix optimized
# 1. 修正 load_wifi_data(): 改進資料洩漏問題
# 2. 新增 Future Domain 測試，驗證時間漂移對抗能力
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
parser = argparse.ArgumentParser(description='CDAN RTT Ablation Study')
# 接收如 "1 2 3" 這樣的字串，代表使用 Dist_mm_1, Dist_mm_2, Dist_mm_3
parser.add_argument('--rtt_indices', type=str, required=True, help='Space separated AP indices for RTT (e.g., "1 2 4")')
parser.add_argument('--base_path', type=str, default='..', help='Base path for data')
# 新增：接收未來時間點測試集的參數
parser.add_argument('--future_csv', type=str, default='', help='Path for the future date test dataset (optional)')
args = parser.parse_args()

# 解析 RTT Columns
rtt_indices = args.rtt_indices.strip().split()
RTT_COLS = [f'Dist_mm_{i}' for i in rtt_indices]
RTT_INPUT_DIM = len(RTT_COLS)
RTT_COMBO_NAME = "FusionCDAN_" + "_".join(rtt_indices)

print(f"==========================================")
print(f"Current Experiment: RTT Combination: {RTT_COLS}")
print(f"RTT Input Dimension: {RTT_INPUT_DIM}")
if args.future_csv:
    print(f"Future Domain Test: ENABLED ({args.future_csv})")
print(f"==========================================")

# 建立結果資料夾
RESULT_DIR = "results"
CDF_DIR = os.path.join(RESULT_DIR, "cdf_data")
os.makedirs(CDF_DIR, exist_ok=True)

# ==========================================
# 1. 核心組件 (GRL & Map) - 保持不變
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

class RandomizedMultiLinearMap(nn.Module):
    def __init__(self, feature_dim, num_classes, output_dim=1024):
        super(RandomizedMultiLinearMap, self).__init__()
        self.output_dim = output_dim
        self.register_buffer('Rf', torch.randn(feature_dim, output_dim))
        self.register_buffer('Rg', torch.randn(num_classes, output_dim))
        self.output_dim = output_dim

    def forward(self, f, g):
        Rf_f = torch.mm(f, self.Rf)
        Rg_g = torch.mm(g, self.Rg)
        h = (Rf_f * Rg_g) / (self.output_dim ** 0.5)
        return h

# ==========================================
# 2. 模型架構：Dual Stream CDAN (加入座標回歸輔助)
# ==========================================
class DualStreamCDAN(nn.Module):
    def __init__(self, num_aps=4, num_classes=49, hidden_dim=64, rtt_input_dim=1):
        super(DualStreamCDAN, self).__init__()
        self.num_classes = num_classes

        # --- 分支 B: RTT 特徵提取器 ---
        self.rtt_extractor = nn.Sequential(
            nn.Linear(rtt_input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True)
        )

        # --- 分支 A: RSSI 特徵提取器 ---
        self.rssi_extractor = nn.Sequential(
            nn.Linear(6, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True)
        )

        self.feature_dim = hidden_dim * 2

        self.class_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

        self.map_rssi = RandomizedMultiLinearMap(hidden_dim, num_classes + 2, output_dim=512)
        self.disc_rssi = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

        self.map_rtt = RandomizedMultiLinearMap(hidden_dim, num_classes + 2, output_dim=512)
        self.disc_rtt = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        
        self.grl = GradientReversalLayer()

    def forward(self, rssi, rtt, coord_tensor, alpha=1.0):
        f_rssi = self.rssi_extractor(rssi)
        f_rtt = self.rtt_extractor(rtt)
        
        f_cat = torch.cat((f_rssi, f_rtt), dim=1)
        class_logits = self.class_classifier(f_cat)
        softmax_output = F.softmax(class_logits, dim=1)

        expected_coords = torch.mm(softmax_output, coord_tensor) 
        g_cond = torch.cat((softmax_output, expected_coords), dim=1)
        
        # Branch A
        h_rssi = self.map_rssi(f_rssi, g_cond)
        h_rev_rssi = self.grl(h_rssi, alpha)
        d_logits_rssi = self.disc_rssi(h_rev_rssi)

        # Branch B
        h_rtt = self.map_rtt(f_rtt, g_cond)
        h_rev_rtt = self.grl(h_rtt, alpha)
        d_logits_rtt = self.disc_rtt(h_rev_rtt)

        return class_logits, d_logits_rssi, d_logits_rtt, softmax_output

# ==========================================
# 輔助函式
# ==========================================
def calc_entropy(softmax_output):
    epsilon = 1e-5
    entropy = -torch.sum(softmax_output * torch.log(softmax_output + epsilon), dim=1)
    return entropy

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
    df = pd.read_csv(csv_path)
    rssi_cols = ['Diff_RSSI_1_2', 'Diff_RSSI_1_3', 'Diff_RSSI_1_4', 'Diff_RSSI_2_3', 'Diff_RSSI_2_4', 'Diff_RSSI_3_4']
    rtt_cols = rtt_cols_to_use 
    
    for col in rssi_cols: df[col] = df[col].replace(-100, np.nan)
    for col in rtt_cols: df[col] = df[col].replace([0, -1], np.nan)

    df[rssi_cols] = df[rssi_cols].fillna(-100)
    df[rtt_cols] = df[rtt_cols].fillna(-1)

    rssi_raw = df[rssi_cols].values.astype(np.float32)
    rtt_raw = df[rtt_cols].values.astype(np.float32)
    raw_labels = df['Label'].values
    
    return rssi_raw, rtt_raw, raw_labels

def get_stratified_indices(labels, split_counts):
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

def validate_process(model, source_val_loader, target_val_loader, device, coord_tensor):
    model.eval()
    total_correct_s = 0
    total_s = 0
    total_entropy_t = 0.0
    num_batches_t = 0
    
    with torch.no_grad():
        for s_rssi, s_rtt, s_label in source_val_loader:
            s_rssi, s_rtt, s_label = s_rssi.to(device), s_rtt.to(device), s_label.to(device)
            class_out_s, _, _, _ = model(s_rssi, s_rtt, coord_tensor, alpha=0)
            preds = torch.argmax(class_out_s, dim=1)
            total_correct_s += (preds == s_label).sum().item()
            total_s += s_label.size(0)

        for t_rssi, t_rtt, _ in target_val_loader:
            t_rssi, t_rtt = t_rssi.to(device), t_rtt.to(device)
            _, _, _, softmax_t = model(t_rssi, t_rtt, coord_tensor, alpha=0)
            entropy = -torch.sum(softmax_t * torch.log(softmax_t + 1e-5), dim=1)
            total_entropy_t += entropy.mean().item()
            num_batches_t += 1

    avg_s_acc = total_correct_s / total_s
    avg_t_entropy = total_entropy_t / num_batches_t
    return avg_s_acc, avg_t_entropy

def evaluate_test(model, data_loader, coord_tensor, device, return_all_errors=False):
    model.eval()
    correct = 0; total = 0; total_dist = 0.0; all_dists = []
    
    with torch.no_grad():
        for rssi, rtt, labels in data_loader:
            rssi, rtt, labels = rssi.to(device), rtt.to(device), labels.to(device)
            class_out, _, _, _ = model(rssi, rtt, coord_tensor, alpha=0)
            preds = torch.argmax(class_out, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            dist = torch.norm(coord_tensor[preds] - coord_tensor[labels], p=2, dim=1)
            total_dist += dist.sum().item()
            
            if return_all_errors: 
                all_dists.extend(dist.cpu().numpy())
                
    if total == 0: return 0, 0, []
    return 100.*correct/total, total_dist/total, np.array(all_dists)

def distance_weighted_ce(logits, targets, coord_tensor, alpha_dist=0.1):
    ce_loss = F.cross_entropy(logits, targets)
    probs = F.softmax(logits, dim=1) 
    gt_coords = coord_tensor[targets]
    dists = torch.norm(coord_tensor[None, :, :] - gt_coords[:, None, :], p=2, dim=2) 
    spatial_penalty = torch.mean(torch.sum(probs * dists, dim=1))
    return ce_loss + alpha_dist * spatial_penalty

# ==========================================
# 3. 主程式
# ==========================================
def main():
    results = []
    seed_candidate = [42, 67, 1024]

    SOURCE_CSV = os.path.join(args.base_path, '2026_1_1/all/All_Data_With_RSSI_Diff.csv')
    TARGET_CSV = os.path.join(args.base_path, '2026_2_4/All_Data_With_RSSI_Diff_withoutNA.csv')
    
    # 解析未來測試集路徑
    FUTURE_CSV = args.future_csv
    has_future_test = bool(FUTURE_CSV)
    
    for seed in seed_candidate:
        set_seed(seed)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        s_rssi_raw, s_rtt_raw, s_labels_raw = load_raw_data(SOURCE_CSV, rtt_cols_to_use=RTT_COLS)
        t_rssi_raw, t_rtt_raw, t_labels_raw = load_raw_data(TARGET_CSV, rtt_cols_to_use=RTT_COLS)
        
        # 若有提供未來測試集，載入原始資料
        if has_future_test:
            f_rssi_raw, f_rtt_raw, f_labels_raw = load_raw_data(FUTURE_CSV, rtt_cols_to_use=RTT_COLS)
        
        # 1. 取得切割索引
        source_split_counts = [80, 20, 20] 
        target_split_counts = [40, 20, 20]
        s_tr_idx, s_val_idx, s_test_idx = get_stratified_indices(s_labels_raw, source_split_counts)
        t_tr_idx, t_val_idx, t_test_idx = get_stratified_indices(t_labels_raw, target_split_counts)

        # 2. 初始化 Scaler (每個 Seed 重新初始化)
        rssi_scaler = MinMaxScaler(feature_range=(-1, 1))
        rtt_scaler = MinMaxScaler(feature_range=(-1, 1))
        label_encoder = LabelEncoder()

        # 3. [關鍵] 只有 Source Train 參與 fit !
        rssi_scaler.fit(s_rssi_raw[s_tr_idx])
        rtt_scaler.fit(s_rtt_raw[s_tr_idx])
        label_encoder.fit(s_labels_raw[s_tr_idx])

        def create_dataset(rssi, rtt, labels, indices):
            r_t = rssi_scaler.transform(rssi[indices])
            rt_t = rtt_scaler.transform(rtt[indices])
            try: l_t = label_encoder.transform(labels[indices])
            except: l_t = np.zeros(len(indices))
            return TensorDataset(torch.tensor(r_t), torch.tensor(rt_t), torch.tensor(l_t, dtype=torch.long))

        # 4. 建立 TensorDataset
        s_train = create_dataset(s_rssi_raw, s_rtt_raw, s_labels_raw, s_tr_idx)
        s_val = create_dataset(s_rssi_raw, s_rtt_raw, s_labels_raw, s_val_idx)
        s_test = create_dataset(s_rssi_raw, s_rtt_raw, s_labels_raw, s_test_idx)
        
        t_train = create_dataset(t_rssi_raw, t_rtt_raw, t_labels_raw, t_tr_idx)
        t_val = create_dataset(t_rssi_raw, t_rtt_raw, t_labels_raw, t_val_idx)
        t_test = create_dataset(t_rssi_raw, t_rtt_raw, t_labels_raw, t_test_idx)
        
        # 針對未來時間點資料，使用全量進行測試 (不切割)
        if has_future_test:
            f_all_idx = np.arange(len(f_labels_raw))
            f_test = create_dataset(f_rssi_raw, f_rtt_raw, f_labels_raw, f_all_idx)

        # ====================================
        BATCH_SIZE = 32
        NUM_WORKERS = 0
        source_loader = DataLoader(s_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS, pin_memory=True)
        target_train_loader = DataLoader(t_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS, pin_memory=True)
        source_val_loader = DataLoader(s_val, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=NUM_WORKERS, pin_memory=True)
        target_val_loader = DataLoader(t_val, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=NUM_WORKERS, pin_memory=True)
        source_test_loader = DataLoader(s_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        target_test_loader = DataLoader(t_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        
        if has_future_test:
            future_test_loader = DataLoader(f_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        class_names = label_encoder.classes_
        COORD_TENSOR = create_coord_tensor(class_names, DEVICE)

        model = DualStreamCDAN(num_aps=4, num_classes=len(class_names), rtt_input_dim=RTT_INPUT_DIM).to(DEVICE)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
        domain_criterion = nn.BCEWithLogitsLoss(reduction='none')

        num_epochs = 400
        best_epoch = -1
        best_score = float('-inf')
        WARMUP_EPOCHS = 10
        
        temp_model_name = f"temp_model_{RTT_COMBO_NAME}_seed{seed}.pth"

        print(f"Start Training Seed {seed}...")
        for epoch in range(num_epochs):
            model.train()
            # ... 訓練迴圈省略印出細節，保持原樣 ...
            p = float(epoch) / num_epochs
            alpha = 2. / (1. + np.exp(-7.5 * p)) - 1
            
            for (s_rssi_b, s_rtt_b, s_lbl_b), (t_rssi_b, t_rtt_b, _) in zip(source_loader, target_train_loader):
                s_rssi_b, s_rtt_b, s_lbl_b = s_rssi_b.to(DEVICE), s_rtt_b.to(DEVICE), s_lbl_b.to(DEVICE)
                t_rssi_b, t_rtt_b = t_rssi_b.to(DEVICE), t_rtt_b.to(DEVICE)
                
                cls_out_s, d_logits_rssi_s, d_logits_rtt_s, softmax_s = model(s_rssi_b, s_rtt_b, COORD_TENSOR, alpha=alpha)
                _, d_logits_rssi_t, d_logits_rtt_t, softmax_t = model(t_rssi_b, t_rtt_b, COORD_TENSOR, alpha=alpha)
                
                loss_cls = distance_weighted_ce(cls_out_s, s_lbl_b, COORD_TENSOR, alpha_dist=0.1)
                
                entropy_s = calc_entropy(softmax_s)
                entropy_t = calc_entropy(softmax_t)
                weight_s = 1.0 + torch.exp(-entropy_s)
                weight_t = 1.0 + torch.exp(-entropy_t)
                weight_s = weight_s / torch.mean(weight_s)
                weight_t = weight_t / torch.mean(weight_t)

                d_lbl_s = torch.ones(s_rssi_b.size(0), 1).to(DEVICE)
                d_lbl_t = torch.zeros(t_rssi_b.size(0), 1).to(DEVICE)
                
                loss_dom_rssi_s = domain_criterion(d_logits_rssi_s, d_lbl_s)
                loss_dom_rssi_t = domain_criterion(d_logits_rssi_t, d_lbl_t)
                loss_dom_rtt_s = domain_criterion(d_logits_rtt_s, d_lbl_s)
                loss_dom_rtt_t = domain_criterion(d_logits_rtt_t, d_lbl_t)
                
                loss_dom_rssi = torch.mean(weight_s.view(-1, 1) * loss_dom_rssi_s) + torch.mean(weight_t.view(-1, 1) * loss_dom_rssi_t)
                loss_dom_rtt = torch.mean(weight_s.view(-1, 1) * loss_dom_rtt_s) + torch.mean(weight_t.view(-1, 1) * loss_dom_rtt_t)
                
                loss = loss_cls + 1 * (loss_dom_rssi + loss_dom_rtt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation
            val_s_acc, val_t_entropy = validate_process(model, source_val_loader, target_val_loader, DEVICE, COORD_TENSOR)
            current_score = val_s_acc - (0.5 * val_t_entropy)

            if (epoch + 1) > WARMUP_EPOCHS:
                if current_score > best_score:
                    best_score = current_score
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), temp_model_name)

            if (epoch + 1) % 1 == 0:
                t_acc, t_mde, _ = evaluate_test(model, target_test_loader, COORD_TENSOR, DEVICE)
                # print(f"{epoch+1:<6} | {total_loss_sum/num_batches:<10.4f} | {train_cls_sum/num_batches:<10.4f} | {train_dom_sum_rssi/num_batches:<12.4f} | {train_dom_sum_rtt/num_batches:<12.4f} | {val_s_acc:<10.4f} | {val_t_entropy:<12.4f} | {t_mde:<8.4f} | {save_mark:<10}")
                        

        # Final Test & Save Results
        if best_epoch != -1:
            model.load_state_dict(torch.load(temp_model_name))
            os.remove(temp_model_name) 
            
        t_acc, t_mde, t_errors = evaluate_test(model, target_test_loader, COORD_TENSOR, DEVICE, return_all_errors=True)
        s_acc, s_mde, s_errors = evaluate_test(model, source_test_loader, COORD_TENSOR, DEVICE, return_all_errors=True)
        
        # 建立記錄用的字典
        res_dict = {
            "Combo": RTT_COMBO_NAME,
            "Seed": seed,
            "Source_Acc": s_acc,
            "Source_MDE": s_mde,
            "Target_Acc": t_acc,
            "Target_MDE": t_mde
        }

        # 評估未來測試集
        if has_future_test:
            f_acc, f_mde, f_errors = evaluate_test(model, future_test_loader, COORD_TENSOR, DEVICE, return_all_errors=True)
            res_dict["Future_Acc"] = f_acc
            res_dict["Future_MDE"] = f_mde
            print(f"Seed {seed} | Src Acc: {s_acc:.4f} | Src MDE: {s_mde:.4f} | Tgt Acc: {t_acc:.4f} | Tgt MDE: {t_mde:.4f} | Fut Acc: {f_acc:.4f} | Fut MDE: {f_mde:.4f}")
            # 儲存 CDF 用的 Error Array
            np.save(os.path.join(CDF_DIR, f"error_{RTT_COMBO_NAME}_future_seed{seed}.npy"), f_errors)
        else:
            print(f"Seed {seed} | Src Acc: {s_acc:.4f} | Src MDE: {s_mde:.4f} | Tgt Acc: {t_acc:.4f} | Tgt MDE: {t_mde:.4f}")
        
        results.append(res_dict)
        np.save(os.path.join(CDF_DIR, f"error_{RTT_COMBO_NAME}_seed{seed}.npy"), t_errors)

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
            
        row_data = [RTT_COMBO_NAME, f"{avg_s_acc:.4f}", f"{avg_s_acc_std:.4f}", f"{avg_s_mde:.4f}", f"{avg_s_mde_std:.4f}", f"{avg_t_acc:.4f}", f"{avg_t_acc_std:.4f}", f"{avg_t_mde:.4f}", f"{avg_t_mde_std:.4f}"]
        
        if has_future_test:
            avg_f_acc = df_res['Future_Acc'].mean()
            avg_f_acc_std = df_res['Future_Acc'].std()
            avg_f_mde = df_res['Future_MDE'].mean()
            avg_f_mde_std = df_res['Future_MDE'].std()
            row_data.extend([f"{avg_f_acc:.4f}", f"{avg_f_acc_std:.4f}", f"{avg_f_mde:.4f}", f"{avg_f_mde_std:.4f}"])
            
        row_data.append(str(seed_candidate))
        writer.writerow(row_data)
        
    print(f"Finished Combo {RTT_COMBO_NAME}. Results saved.")

if __name__ == '__main__':
    main()