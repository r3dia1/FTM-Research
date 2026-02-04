# ===================== Version Info =============================
# 設計演算法來找最佳模型，詳細看 line 517 comment
# 根據 version 3 的版本做自動化腳本，測試平均 s/t acc/mde
# 每次訓練前檢查：Source/Target資料日期（路徑）、RSSI/RTT資料的選擇、特徵萃取器輸入維度
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
RSSI_COLS = ['Diff_RSSI_1_2', 'Diff_RSSI_1_3', 'Diff_RSSI_1_4', 'Diff_RSSI_2_3', 'Diff_RSSI_2_4', 'Diff_RSSI_3_4']

# 計算輸入維度
RTT_INPUT_DIM = len(RTT_COLS)
RSSI_INPUT_DIM = len(RSSI_COLS)
COMBO_NAME = "Dual_FixedRSSI_" + "RTT_" + "_".join(rtt_indices)

print(f"==========================================")
print(f"Experiment: {COMBO_NAME}")
print(f"RSSI Features ({RSSI_INPUT_DIM}): Fixed")
print(f"RTT Features ({RTT_INPUT_DIM}): {RTT_COLS}")
print(f"==========================================")

# 建立結果資料夾
RESULT_DIR = "results_dual"
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
    def __init__(self, rssi_dim=6, rtt_dim=3, num_classes=49, hidden_dim=64):
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

def load_wifi_data(csv_path, is_source=True, samples_per_label=None, rtt_cols_to_use=None):
    global is_scaler_fitted
    
    df = pd.read_csv(csv_path)
    # 使用全域定義的 RSSI_COLS 和 傳入的 RTT COLS
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
        for col in cols_to_fix:
            if df[col].isnull().all():
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(df[col].mean())

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

def stratified_split(dataset, labels, split_counts):
    if isinstance(labels, torch.Tensor): labels = labels.cpu().numpy()
    num_classes = len(np.unique(labels))
    train_indices, val_indices, test_indices = [], [], []
    unique_labels = np.unique(labels)
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        np.random.shuffle(label_indices)
        n_train, n_val, n_test = split_counts
        train_indices.extend(label_indices[:n_train])
        val_indices.extend(label_indices[n_train : n_train + n_val])
        test_indices.extend(label_indices[n_train + n_val : n_train + n_val + n_test])
    return (Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices))

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
        TARGET_CSV = os.path.join(args.base_path, '2026_1_28/All_Data_With_RSSI_Diff.csv')

        SAMPLES_PER_LABEL = 120
        # Load Data
        s_rssi, s_rtt, s_labels = load_wifi_data(SOURCE_CSV, is_source=True, samples_per_label=SAMPLES_PER_LABEL, rtt_cols_to_use=RTT_COLS)
        t_rssi, t_rtt, t_labels = load_wifi_data(TARGET_CSV, is_source=False, samples_per_label=SAMPLES_PER_LABEL, rtt_cols_to_use=RTT_COLS)

        full_source = TensorDataset(s_rssi, s_rtt, s_labels)
        full_target = TensorDataset(t_rssi, t_rtt, t_labels)

        source_split_counts = [80, 20, 20] 
        target_split_counts = [80, 20, 20]
        s_train, s_val, s_test = stratified_split(full_source, s_labels, source_split_counts)
        t_train, t_val, t_test = stratified_split(full_target, t_labels, target_split_counts)

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
        W_SCORE_CLS = 0.3
        W_SCORE_DOM_1 = 0.1
        W_SCORE_DOM_2 = 0.3
        
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