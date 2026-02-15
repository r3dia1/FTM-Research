# ===================== Version Info =============================
# 修改自 version 3 fixed
# 1. 已完成方向一測試(有效)
# 2. 現在測試方向二: Uncertaintly-Gated Fusion: 讓模型自動調動對 RSSI&RTT 的依賴
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
args = parser.parse_args()

# 解析 RTT Columns
rtt_indices = args.rtt_indices.strip().split()
RTT_COLS = [f'Dist_mm_{i}' for i in rtt_indices]
RTT_INPUT_DIM = len(RTT_COLS)
# RTT_COMBO_NAME = "_".join(rtt_indices) # 用於檔名，例如 "1_2_4"
RTT_COMBO_NAME = "FusionCDAN_" + "_".join(rtt_indices)

print(f"==========================================")
print(f"Current Experiment: RTT Combination: {RTT_COLS}")
print(f"RTT Input Dimension: {RTT_INPUT_DIM}")
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
# 2. 模型架構：Dual Stream CDAN (修改版：加入 Uncertainty-Gated Fusion)
# ==========================================

class UncertaintyGate(nn.Module):
    """
    這是一個輕量級的 Gate Network，用於評估 RSSI 和 RTT 特徵的可靠性。
    輸入: RSSI特徵 + RTT特徵
    輸出: [Weight_RSSI, Weight_RTT] (經過 Softmax 或 Sigmoid)
    """
    def __init__(self, feature_dim, hidden_dim=32):
        super(UncertaintyGate, self).__init__()
        self.gate_fc = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 2), # 輸出兩個權重: [w_rssi, w_rtt]
            nn.Softmax(dim=1)         # 使用 Softmax 讓兩者權重相加為 1 (互補競爭)，
                                      # 或者改用 Sigmoid 讓兩者獨立 (視實驗效果而定)
        )

    def forward(self, x):
        return self.gate_fc(x)

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
        # self.feature_dim = hidden_dim

        # [新增] Uncertainty Gate
        # 輸入維度是 hidden_dim * 2 (因為它同時看 RSSI 和 RTT 來決定誰比較好)
        self.gate_network = UncertaintyGate(self.feature_dim, hidden_dim=32)

        self.class_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

        # 這裡不需要改動，Discriminator 接收的是「加權後」的特徵
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
        """
        coord_tensor: 傳入所有 RP 的真實座標 (C, 2)，用於計算期望座標
        """
        # 1. 提取原始特徵
        f_rssi = self.rssi_extractor(rssi) # (B, 64)
        f_rtt = self.rtt_extractor(rtt)    # (B, 64)
        
        # 2. [新增] 計算 Gated Weights
        # 先將兩者串接，讓 Gate Network 觀察整體資訊
        f_raw_cat = torch.cat((f_rssi, f_rtt), dim=1) # (B, 128)
        weights = self.gate_network(f_raw_cat)        # (B, 2) -> [w_rssi, w_rtt]
        
        # 取出權重並調整形狀以進行廣播 (Broadcasting)
        w_rssi = weights[:, 0].unsqueeze(1) # (B, 1)
        w_rtt  = weights[:, 1].unsqueeze(1) # (B, 1)

        # 3. [新增] 應用權重 (Feature Reweighting)
        # 如果 Gate 覺得 RSSI 不可靠，w_rssi 會變小，f_rssi 的數值就會被壓低
        f_rssi_weighted = f_rssi * w_rssi
        f_rtt_weighted  = f_rtt * w_rtt

        # 4. 串接加權後的特徵進入分類器
        # f_cat = torch.cat((f_rssi_weighted, f_rtt_weighted), dim=1)
        # class_logits = self.class_classifier(f_cat)

        # 改成這樣 (記得 class_classifier 的輸入維度要從 feature_dim 改回 hidden_dim)：
        f_fused = f_rssi_weighted + f_rtt_weighted
        class_logits = self.class_classifier(f_fused)

        softmax_output = F.softmax(class_logits, dim=1)

        # --- 座標回歸輔助 (Coordinate Regression Auxiliary) ---
        expected_coords = torch.mm(softmax_output, coord_tensor) # (B, 2)
        
        # Condition: [g \oplus Coord]
        g_cond = torch.cat((softmax_output, expected_coords), dim=1)

        # 5. Domain Adaptation (使用加權後的特徵)
        # 這裡非常關鍵：如果 w_rssi 很小，Discriminator 看到的 f_rssi_weighted 也接近 0。
        # 這意味著模型主動放棄了混淆這個模態的 Domain，專注於可信的模態。
        
        # Branch A (RSSI)
        # h_rssi = self.map_rssi(f_rssi_weighted, g_cond) # 注意這裡傳入的是 weighted
        h_rssi = self.map_rssi(f_rssi, g_cond)
        h_rev_rssi = self.grl(h_rssi, alpha)
        d_logits_rssi = self.disc_rssi(h_rev_rssi)

        # Branch B (RTT)
        # h_rtt = self.map_rtt(f_rtt_weighted, g_cond)    # 注意這裡傳入的是 weighted
        h_rtt = self.map_rtt(f_rtt, g_cond)
        h_rev_rtt = self.grl(h_rtt, alpha)
        d_logits_rtt = self.disc_rtt(h_rev_rtt)

        return class_logits, d_logits_rssi, d_logits_rtt, softmax_output, weights

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

def load_wifi_data(csv_path, is_source=True, samples_per_label=None, rtt_cols_to_use=None):
    global is_scaler_fitted
    df = pd.read_csv(csv_path)
    
    rssi_cols = ['Diff_RSSI_1_2', 'Diff_RSSI_1_3', 'Diff_RSSI_1_4', 'Diff_RSSI_2_3', 'Diff_RSSI_2_4', 'Diff_RSSI_3_4']
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
        # for col in cols_to_fix:
        #     if df[col].isnull().all():
        #         df[col] = df[col].fillna(0)
        #     else:
        #         df[col] = df[col].fillna(df[col].mean())
    
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

# 坐標 Label Mapping (省略內容，請保持原樣)
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
    total_cls_loss = 0.0
    total_correct_s = 0
    total_s = 0

    # 追蹤 Source 的權重
    total_w_rssi_s, total_w_rtt_s = 0.0, 0.0
    
    # 新增：計算 Target Entropy
    total_entropy_t = 0.0
    num_batches_t = 0
    total_t = 0
    total_w_rssi_t, total_w_rtt_t = 0.0, 0.0
    
    with torch.no_grad():
        # Source Validation
        for s_rssi, s_rtt, s_label in source_val_loader:
            s_rssi, s_rtt, s_label = s_rssi.to(device), s_rtt.to(device), s_label.to(device)
            class_out_s, _, _, _, weights_s = model(s_rssi, s_rtt, coord_tensor, alpha=0)
            
            # 計算 Source Accuracy (比 Loss 更直觀)
            preds = torch.argmax(class_out_s, dim=1)
            total_correct_s += (preds == s_label).sum().item()
            total_s += s_label.size(0)

            # 將這個 Batch 中所有的權重加總
            total_w_rssi_s += weights_s[:, 0].sum().item()
            total_w_rtt_s += weights_s[:, 1].sum().item()

        # Target Validation (只看 Entropy，不看 Domain Loss)
        for t_rssi, t_rtt, _ in target_val_loader:
            t_rssi, t_rtt = t_rssi.to(device), t_rtt.to(device)
            _, _, _, softmax_t, weights_t = model(t_rssi, t_rtt, coord_tensor, alpha=0)
            
            # Entropy 計算: -sum(p * log(p))
            entropy = -torch.sum(softmax_t * torch.log(softmax_t + 1e-5), dim=1)
            total_entropy_t += entropy.mean().item()
            num_batches_t += 1

            # 將這個 Batch 中所有的權重加總
            total_w_rssi_t += weights_t[:, 0].sum().item()
            total_w_rtt_t += weights_t[:, 1].sum().item()

    avg_s_acc = total_correct_s / total_s
    avg_t_entropy = total_entropy_t / num_batches_t

    # 計算平均權重
    avg_w_rssi_s = total_w_rssi_s / total_s
    avg_w_rtt_s = total_w_rtt_s / total_s
    avg_w_rssi_t = total_w_rssi_t / total_t if total_t > 0 else 0
    avg_w_rtt_t = total_w_rtt_t / total_t if total_t > 0 else 0
    
    # 回傳時加入權重資訊
    return avg_s_acc, avg_t_entropy, avg_w_rssi_s, avg_w_rtt_s, avg_w_rssi_t, avg_w_rtt_t

def evaluate_test(model, data_loader, coord_tensor, device, return_all_errors=False):
    model.eval()
    correct = 0; total = 0; total_dist = 0.0; all_dists = []

    # 新增：紀錄權重
    total_w_rssi = 0.0
    total_w_rtt = 0.0
    
    with torch.no_grad():
        for rssi, rtt, labels in data_loader:
            rssi, rtt, labels = rssi.to(device), rtt.to(device), labels.to(device)
            class_out, _, _, _ , weights = model(rssi, rtt, coord_tensor, alpha=0)
            preds = torch.argmax(class_out, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            dist = torch.norm(coord_tensor[preds] - coord_tensor[labels], p=2, dim=1)
            total_dist += dist.sum().item()

            # 累加權重
            total_w_rssi += weights[:, 0].sum().item()
            total_w_rtt += weights[:, 1].sum().item()
            
            if return_all_errors: 
                all_dists.extend(dist.cpu().numpy())
                
    if total == 0: return 0, 0, [], 0.0, 0.0
    
    avg_w_rssi = total_w_rssi / total
    avg_w_rtt = total_w_rtt / total
    
    return 100.*correct/total, total_dist/total, np.array(all_dists), avg_w_rssi, avg_w_rtt

def distance_weighted_ce(logits, targets, coord_tensor, alpha_dist=0.1):
    """
    logits: (B, C) - 模型輸出的類別預測
    targets: (B) - 真實類別索引
    coord_tensor: (C, 2) - 所有類別對應的物理座標
    alpha_dist: 距離懲罰的強度系數
    """
    # 1. 標準的 Cross Entropy (CE)
    ce_loss = F.cross_entropy(logits, targets)
    
    # 2. 計算距離權重 (Distance-based Penalty)
    # 取得當前 batch 中每個樣本預測值的機率分佈
    probs = F.softmax(logits, dim=1) # (B, C)
    
    # 取得當前 batch 中真實位置的座標 (B, 2)
    gt_coords = coord_tensor[targets]
    
    # 計算 batch 內每個預測機率點與真實座標的距離
    # 使用廣播機制計算 (B, C) 維度的距離矩陣
    # coord_tensor[None, :, :] -> (1, C, 2)
    # gt_coords[:, None, :] -> (B, 1, 2)
    dists = torch.norm(coord_tensor[None, :, :] - gt_coords[:, None, :], p=2, dim=2) # (B, C)
    
    # 空間懲罰：對所有類別的預測機率進行加權，距離越遠權重越大
    spatial_penalty = torch.mean(torch.sum(probs * dists, dim=1))
    
    # 最終 Loss = 原始分類誤差 + 空間結構懲罰
    return ce_loss + alpha_dist * spatial_penalty

# ==========================================
# 3. 主程式
# ==========================================
def main():
    # 儲存結果用的 list
    results = []
    
    seed_candidate = [42, 6767, 123456]

    # 根據參數拼接路徑
    # 請確保資料夾路徑正確
    SOURCE_CSV = os.path.join(args.base_path, '2026_1_1/all/All_Data_With_RSSI_Diff.csv')
    TARGET_CSV = os.path.join(args.base_path, '2026_1_23/All_Data_With_RSSI_Diff_withoutNA.csv')

    SAMPLES_PER_LABEL = 120
    # 載入資料時傳入 RTT_COLS
    s_rssi, s_rtt, s_labels = load_wifi_data(SOURCE_CSV, is_source=True, samples_per_label=SAMPLES_PER_LABEL, rtt_cols_to_use=RTT_COLS)
    t_rssi, t_rtt, t_labels = load_wifi_data(TARGET_CSV, is_source=False, samples_per_label=SAMPLES_PER_LABEL, rtt_cols_to_use=RTT_COLS)
    
    for seed in seed_candidate:
        set_seed(seed)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        full_source = TensorDataset(s_rssi, s_rtt, s_labels)
        full_target = TensorDataset(t_rssi, t_rtt, t_labels)
        
        source_split_counts = [80, 20, 20] 
        target_split_counts = [80, 20, 20]
        s_train, s_val, s_test = stratified_split(full_source, s_labels, source_split_counts)
        t_train, t_val, t_test = stratified_split(full_target, t_labels, target_split_counts)

        BATCH_SIZE = 32
        NUM_WORKERS = 8
        source_loader = DataLoader(s_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS, pin_memory=True)
        target_train_loader = DataLoader(t_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS, pin_memory=True)
        source_val_loader = DataLoader(s_val, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=NUM_WORKERS, pin_memory=True)
        target_val_loader = DataLoader(t_val, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=NUM_WORKERS, pin_memory=True)
        source_test_loader = DataLoader(s_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        target_test_loader = DataLoader(t_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        class_names = label_encoder.classes_
        COORD_TENSOR = create_coord_tensor(class_names, DEVICE)

        # 傳入 RTT_INPUT_DIM
        model = DualStreamCDAN(num_aps=4, num_classes=len(class_names), rtt_input_dim=RTT_INPUT_DIM).to(DEVICE)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
        domain_criterion = nn.BCEWithLogitsLoss(reduction='none')

        num_epochs = 400
        best_epoch = -1
        best_score = float('-inf')
        
        WARMUP_EPOCHS = 10
        
        # 臨時模型檔名
        temp_model_name = f"temp_model_{RTT_COMBO_NAME}_seed{seed}.pth"

        print(f"Start Training Seed {seed}...")
        print(f"\nStart CDAN+E Training... (Entropy Conditioning Enabled)")
        print(f"{'Epoch':<6} | {'Total Loss':<10} | {'Tr Cls':<10} | {'Tr Dom RSSI':<12} | {'Tr Dom RTT':<12} | {'Val T Acc':<10} | {'Val T Entropy':<12} | {'Test MDE':<8} | {'Test Target Weights':<18}")
        print("-" * 140)

        # 預先計算距離矩陣
        # # Class Center Distance Matrix: shape (C, C)
        # num_cls = len(class_names)
        # cls_coords = COORD_TENSOR # (C, 2)
        # # 利用廣播計算所有類別兩兩之間的距離
        # dist_matrix = torch.norm(cls_coords[:, None, :] - cls_coords[None, :, :], p=2, dim=2)

        for epoch in range(num_epochs):
            model.train()

            total_loss_sum = 0.0
            train_cls_sum = 0.0
            train_dom_sum_rssi = 0.0
            train_dom_sum_rtt = 0.0
            num_batches = 0
            
            p = float(epoch) / num_epochs
            alpha = 2. / (1. + np.exp(-7.5 * p)) - 1
            
            for (s_rssi_b, s_rtt_b, s_lbl_b), (t_rssi_b, t_rtt_b, _) in zip(source_loader, target_train_loader):
                s_rssi_b, s_rtt_b, s_lbl_b = s_rssi_b.to(DEVICE), s_rtt_b.to(DEVICE), s_lbl_b.to(DEVICE)
                t_rssi_b, t_rtt_b = t_rssi_b.to(DEVICE), t_rtt_b.to(DEVICE)
                
                # 訓練階段
                cls_out_s, d_logits_rssi_s, d_logits_rtt_s, softmax_s, _ = model(s_rssi_b, s_rtt_b, COORD_TENSOR, alpha=alpha)
                _, d_logits_rssi_t, d_logits_rtt_t, softmax_t, _ = model(t_rssi_b, t_rtt_b, COORD_TENSOR, alpha=alpha)
                
                # 修改處：將 CE 替換為 Distance-Weighted CE
                # alpha_dist 可根據需求微調，建議先從 0.1 或 0.5 開始
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

                total_loss_sum += loss.item()
                train_cls_sum += loss_cls.item()
                train_dom_sum_rssi += loss_dom_rssi.item()
                train_dom_sum_rtt += loss_dom_rtt.item()
                num_batches += 1
            
            # Validation
            val_s_acc, val_t_entropy, _, _, _, _ = validate_process(model, source_val_loader, target_val_loader, DEVICE, COORD_TENSOR)
            
            save_mark = ""
            current_score = val_s_acc - (0.5 * val_t_entropy)

            if (epoch + 1) > WARMUP_EPOCHS:
                if current_score > best_score:
                    best_score = current_score
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), temp_model_name)
                    save_mark = f"(ADV {current_score:.2f})"
            
            if (epoch + 1) % 1 == 0:
                t_acc, t_mde, _, avg_w_rssi, avg_w_rtt = evaluate_test(model, target_test_loader, COORD_TENSOR, DEVICE)
                print(f"{epoch+1:<6} | {total_loss_sum/num_batches:<10.4f} | {train_cls_sum/num_batches:<10.4f} | {train_dom_sum_rssi/num_batches:<12.4f} | {train_dom_sum_rtt/num_batches:<12.4f} | {val_s_acc:<10.4f} | {val_t_entropy:<12.4f} | {t_mde:<8.4f} | {avg_w_rssi:<8.4f}, {avg_w_rtt:<8.4f} | {save_mark:<10}")
                # print(f"{epoch+1:<6} | {total_loss_sum/num_batches:<10.4f} | {train_cls_sum/num_batches:<10.4f} | {train_dom_sum_rssi/num_batches:<12.4f} | {train_dom_sum_rtt/num_batches:<12.4f} | {val_s_acc:<10.4f} | {val_t_entropy:<12.4f} | {save_mark:<10}")

        # Final Test & Save Results
        if best_epoch != -1:
            model.load_state_dict(torch.load(temp_model_name))
            os.remove(temp_model_name) # 測試完刪除模型檔
            
        t_acc, t_mde, t_errors, _, _ = evaluate_test(model, target_test_loader, COORD_TENSOR, DEVICE, return_all_errors=True)
        s_acc, s_mde, s_errors, _, _ = evaluate_test(model, source_test_loader, COORD_TENSOR, DEVICE, return_all_errors=True)
        
        print(f"Seed {seed} | Src Acc: {s_acc:.4f} | Src MDE: {s_mde:.4f} | Tgt Acc: {t_acc:.4f} | Tgt MDE: {t_mde:.4f}")
        
        # 1. 儲存該 Seed 的結果到列表
        results.append({
            "Combo": RTT_COMBO_NAME,
            "Seed": seed,
            "Source_Acc": s_acc,
            "Source_MDE": s_mde,
            "Target_Acc": t_acc,
            "Target_MDE": t_mde
        })
        
        # 2. 儲存 CDF 用的 Error Array
        # 檔名格式: error_{Combo}_seed_{Seed}.npy
        np.save(os.path.join(CDF_DIR, f"error_{RTT_COMBO_NAME}_seed{seed}.npy"), t_errors)

    # 計算平均並寫入 Summary CSV
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
        writer.writerow([RTT_COMBO_NAME, f"{avg_s_acc:.4f}", f"{avg_s_mde:.4f}", f"{avg_t_acc:.4f}", f"{avg_t_mde:.4f}", str(seed_candidate)])
        
    print(f"Finished Combo {RTT_COMBO_NAME}. Results saved.")

if __name__ == '__main__':
    main()