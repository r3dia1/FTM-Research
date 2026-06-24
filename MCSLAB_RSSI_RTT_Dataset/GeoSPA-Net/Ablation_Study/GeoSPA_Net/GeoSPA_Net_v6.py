# ===================== Version Info =============================
# Architecture: GeoSPA-Net (Fixed Version)
# Modifications:
# 1. PCN now dynamically calibrates using Target Domain mcAP data.
# 2. Huber Loss delta fixed for MinMaxScaler scale (delta=0.05).
# 3. Explicit Uncertainty Gating applied via exponential weighting.
# 4. Corrected feature dimensions for Extractors.
# 5. PCN training data mask modified.
# ================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
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
# 0. 參數解析與 AP 角色定義
# ==========================================
parser = argparse.ArgumentParser(description='GeoSPA-Net Ablation Study')
parser.add_argument('--rtt_indices', type=str, required=True, help='Space separated mcAP indices for RTT (e.g., "1 2 4")')
parser.add_argument('--base_path', type=str, default='..', help='Base path for data')
args = parser.parse_args()

TOTAL_APS = 4
ALL_RTT_COLS = [f'Dist_mm_{i}' for i in range(1, TOTAL_APS + 1)]

mc_indices_raw = [int(i) for i in args.rtt_indices.strip().split()]
MC_IDX = [i - 1 for i in mc_indices_raw]
LEGACY_IDX = [i for i in range(TOTAL_APS) if i not in MC_IDX]

RTT_COMBO_NAME = "GeoSPA_" + "_".join([str(i) for i in mc_indices_raw])

print(f"==========================================")
print(f"Current Architecture: GeoSPA-Net Version 6")
print(f"mcAPs (Real Anchors): {mc_indices_raw} (Indices: {MC_IDX})")
print(f"Legacy APs (Virtual): {[i+1 for i in LEGACY_IDX]} (Indices: {LEGACY_IDX})")
print(f"==========================================")

RESULT_DIR = "results"
CDF_DIR = os.path.join(RESULT_DIR, "cdf_data")
os.makedirs(CDF_DIR, exist_ok=True)

# ==========================================
# 1. 核心組件 (GRL & Map)
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

    def forward(self, f, g):
        Rf_f = torch.mm(f, self.Rf)
        Rg_g = torch.mm(g, self.Rg)
        h = (Rf_f * Rg_g) / (self.output_dim ** 0.5)
        return h

# ==========================================
# 2. Stage 2: PCN 與 Densification (Target-Centric 修正版)
# ==========================================
def get_target_centric_features(rssi_batch, target_ap_id):
    """
    rssi_batch 維度對應 (共 16 維):
    0-3: RSSI_1 ~ RSSI_4
    4-6: Diff_1_2, Diff_1_3, Diff_1_4
    7-9: Diff_2_3, Diff_2_4, Diff_3_4
    10-12: Diff_2_1, Diff_3_1, Diff_4_1
    13-15: Diff_3_2, Diff_4_2, Diff_4_3
    """
    target_rssi = rssi_batch[:, target_ap_id].unsqueeze(1)
    
    if target_ap_id == 0:   # 目標是 AP_1
        diff_1 = rssi_batch[:, 4].unsqueeze(1)   # 1 - 2
        diff_2 = rssi_batch[:, 5].unsqueeze(1)   # 1 - 3
        diff_3 = rssi_batch[:, 6].unsqueeze(1)   # 1 - 4
        
    elif target_ap_id == 1: # 目標是 AP_2
        diff_1 = rssi_batch[:, 10].unsqueeze(1)  # 2 - 1 (直接取擴增欄位，無負號)
        diff_2 = rssi_batch[:, 7].unsqueeze(1)   # 2 - 3
        diff_3 = rssi_batch[:, 8].unsqueeze(1)   # 2 - 4
        
    elif target_ap_id == 2: # 目標是 AP_3
        diff_1 = rssi_batch[:, 11].unsqueeze(1)  # 3 - 1
        diff_2 = rssi_batch[:, 13].unsqueeze(1)  # 3 - 2
        diff_3 = rssi_batch[:, 9].unsqueeze(1)   # 3 - 4
        
    elif target_ap_id == 3: # 目標是 AP_4
        diff_1 = rssi_batch[:, 12].unsqueeze(1)  # 4 - 1
        diff_2 = rssi_batch[:, 14].unsqueeze(1)  # 4 - 2
        diff_3 = rssi_batch[:, 15].unsqueeze(1)  # 4 - 3
        
    else:
        raise ValueError("Invalid target_ap_id")

    features = torch.cat([target_rssi, diff_1, diff_2, diff_3], dim=1)
    return features

class PathLossCalibrationNetwork(nn.Module):
    def __init__(self, in_dim=4, num_aps=4):
        super(PathLossCalibrationNetwork, self).__init__()
        # 共享網路：負責萃取「環境共性」與「訊號衰減特徵」
        self.shared_net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.ReLU(True)
        )
        # 獨立預測頭：為每台 AP 建立獨立的線性層，記憶空間專屬特性
        self.heads = nn.ModuleList([nn.Linear(16, 1) for _ in range(num_aps)])

    def forward(self, target_features, ap_id):
        shared_feat = self.shared_net(target_features)
        return self.heads[ap_id](shared_feat)

def train_pcn(data_loader, train_ap_indices, device, epochs=50, domain_name="Source", pretrained_model=None):
    # 若有傳入預訓練模型，則使用預訓練模型；否則初始化新模型
    if pretrained_model is not None:
        pcn = pretrained_model
    else:
        pcn = PathLossCalibrationNetwork(in_dim=4, num_aps=TOTAL_APS).to(device)

    if len(train_ap_indices) == 0: return pcn

    optimizer = optim.Adam(pcn.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=0.05) 
    
    pcn.train()
    print(f">> Training Multi-Head PCN (Stage 2) on {domain_name} Domain for APs: {train_ap_indices}...")
    
    for epoch in range(epochs):
        total_loss = 0
        # 接收 DataLoader 吐出的 4 個變數 (多了 rtt_mask)
        for rssi, rtt_gt, rtt_mask, _ in data_loader:
            rssi, rtt_gt, rtt_mask = rssi.to(device), rtt_gt.to(device), rtt_mask.to(device)
            optimizer.zero_grad()
            batch_loss = 0
            
            for ap_id in train_ap_indices:
                features = get_target_centric_features(rssi, ap_id)
                y_true = rtt_gt[:, ap_id]
                valid_mask = rtt_mask[:, ap_id] # 【關鍵修正 3】取出這顆 AP 的真實有效性 Mask
                
                pred_dist = pcn(features, ap_id).squeeze()
                
                # 直接用 valid_mask，不用再判斷 y_true > -0.99
                if valid_mask.sum() > 0:
                    loss = criterion(pred_dist[valid_mask], y_true[valid_mask])
                    batch_loss += loss
                    
            if type(batch_loss) == torch.Tensor: 
                batch_loss.backward()
                optimizer.step()
                total_loss += batch_loss.item()
                
    print(f">> {domain_name} PCN Training Finished. Huber Loss: {total_loss/len(data_loader):.4f}")
    pcn.eval()
    return pcn

def apply_densification_and_uncertainty(rssi, rtt_real, pcn, mc_idx, legacy_idx, device):
    batch_size = rssi.size(0)
    rtt_densified = rtt_real.clone()
    sigma = torch.zeros(batch_size, TOTAL_APS).to(device)
    
    with torch.no_grad():
        if len(legacy_idx) > 0:
            for ap_id in legacy_idx:
                features = get_target_centric_features(rssi, ap_id)
                
                # 【關鍵修改】推論時傳入 ap_id
                pred_dist = pcn(features, ap_id).squeeze()
                rtt_densified[:, ap_id] = pred_dist
                sigma[:, ap_id] = 1.0 # 虛擬距離，給予較高 Uncertainty
                
    if len(mc_idx) > 0:
        sigma[:, mc_idx] = 0.1 # 真實測量，給予較低 Uncertainty
        
    return rtt_densified, sigma

# ==========================================
# 3. Stage 3 模型架構：GeoSPA-Net (修正版)
# ==========================================
class GeoSPANet(nn.Module):
    def __init__(self, num_classes=49, hidden_dim=64):
        super(GeoSPANet, self).__init__()
        self.num_classes = num_classes

        # [修正 4] 取消 concat sigma，恢復正確特徵維度 (diff_rssi 維度為 6)
        self.rssi_extractor = nn.Sequential(
            nn.Linear(6, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True)
        )

        # [修正 4] RTT 維度為 4
        self.rtt_extractor = nn.Sequential(
            nn.Linear(4, 32),
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
        self.disc_rssi = nn.Sequential(nn.Linear(512, 512), nn.ReLU(True), nn.Dropout(0.5), nn.Linear(512, 1))

        self.map_rtt = RandomizedMultiLinearMap(hidden_dim, num_classes + 2, output_dim=512)
        self.disc_rtt = nn.Sequential(nn.Linear(512, 512), nn.ReLU(True), nn.Dropout(0.5), nn.Linear(512, 1))
        
        self.grl = GradientReversalLayer()

    def forward(self, rssi, rtt_densified, sigma, coord_tensor, alpha=1.0):
        # [修正 3] 真正的 Uncertainty Gating！
        # 使用物理意義的權重：sigma=1.0 (虛擬) 時，權重 exp(-1)=0.36，大幅降低影響
        # sigma=0.1 (真實) 時，權重 exp(-0.1)=0.90，保持高信任度
        uncertainty_weights = torch.exp(-sigma)
        rtt_gated = rtt_densified * uncertainty_weights
        
        f_rssi = self.rssi_extractor(rssi)
        f_rtt = self.rtt_extractor(rtt_gated)
        
        f_cat = torch.cat((f_rssi, f_rtt), dim=1)
        class_logits = self.class_classifier(f_cat)
        softmax_output = F.softmax(class_logits, dim=1)

        expected_coords = torch.mm(softmax_output, coord_tensor)
        g_cond = torch.cat((softmax_output, expected_coords), dim=1)
        
        h_rssi = self.map_rssi(f_rssi, g_cond)
        d_logits_rssi = self.disc_rssi(self.grl(h_rssi, alpha))

        h_rtt = self.map_rtt(f_rtt, g_cond)
        d_logits_rtt = self.disc_rtt(self.grl(h_rtt, alpha))

        return class_logits, d_logits_rssi, d_logits_rtt, softmax_output

# ==========================================
# 以下輔助函式與 Main 流程不變 (保留你的原始結構)
# ==========================================
def calc_entropy(softmax_output):
    return -torch.sum(softmax_output * torch.log(softmax_output + 1e-5), dim=1)

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_raw_data(csv_path):
    df = pd.read_csv(csv_path)
    abs_rssi_cols = ['RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4'] 
    
    # 原始 6 維差分
    diff_rssi_cols = ['Diff_RSSI_1_2', 'Diff_RSSI_1_3', 'Diff_RSSI_1_4', 'Diff_RSSI_2_3', 'Diff_RSSI_2_4', 'Diff_RSSI_3_4']
    
    # 擴增 6 維反向差分
    rev_diff_cols = ['Diff_RSSI_2_1', 'Diff_RSSI_3_1', 'Diff_RSSI_4_1', 'Diff_RSSI_3_2', 'Diff_RSSI_4_2', 'Diff_RSSI_4_3']
    
    rssi_cols_original = abs_rssi_cols + diff_rssi_cols
    rtt_cols = ALL_RTT_COLS 
    
    # 1. 先把原本的 -100 換成 NaN，確保數學計算正確
    for col in rssi_cols_original: df[col] = df[col].replace(-100, np.nan)
    
    # 2. 建立擴增資料 (在尚未填補 -100 之前進行物理反轉)
    df['Diff_RSSI_2_1'] = -df['Diff_RSSI_1_2']
    df['Diff_RSSI_3_1'] = -df['Diff_RSSI_1_3']
    df['Diff_RSSI_4_1'] = -df['Diff_RSSI_1_4']
    df['Diff_RSSI_3_2'] = -df['Diff_RSSI_2_3']
    df['Diff_RSSI_4_2'] = -df['Diff_RSSI_2_4']
    df['Diff_RSSI_4_3'] = -df['Diff_RSSI_3_4']

    all_rssi_cols = abs_rssi_cols + diff_rssi_cols + rev_diff_cols

    # 3. 處理 RTT Masking (保留修正版的乾淨 Mask)
    # ----------------------------------------------------------------------
    # 建立布林遮罩（Mask）：RTT 如果是 0、-1 或 NaN，代表該次測量失敗（沒量到距離）。
    # 這段程式碼創造了一個 True/False 的矩陣 rtt_valid_mask。
    # 未來在模型訓練時，我們可以用這個遮罩跳過無效的 RTT 資料，避免模型學到髒資料。
    # ----------------------------------------------------------------------
    
    valid_rtt_condition = (df[rtt_cols].notna()) & (df[rtt_cols] != 0) & (df[rtt_cols] != -1)
    rtt_valid_mask = valid_rtt_condition.values.astype(bool)
    
    for col in rtt_cols: df[col] = df[col].replace([0, -1], np.nan)

    # 4. 補值準備進入 Scaler
    df[all_rssi_cols] = df[all_rssi_cols].fillna(-100)
    df[rtt_cols] = df[rtt_cols].fillna(-1)

    # 輸出為 16 維的 rssi_raw
    rssi_raw = df[all_rssi_cols].values.astype(np.float32)
    rtt_raw = df[rtt_cols].values.astype(np.float32)
    raw_labels = df['Label'].values
    
    return rssi_raw, rtt_raw, rtt_valid_mask, raw_labels

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
    "2-1": (0, 0.6), "2-11": (6.0, 0.6), "3-1": (0, 1.2), "3-11": (6.0, 1.2), "4-1": (0, 1.8), "4-11": (6.0, 1.8), "5-1": (0, 2.4), "5-11": (6.0, 2.4),
    "6-1": (0, 3.0), "6-2": (0.6, 3.0), "6-3": (1.2, 3.0), "6-4": (1.8, 3.0), "6-5": (2.4, 3.0),"6-6": (3.0, 3.0), "6-7": (3.6, 3.0), "6-8": (4.2, 3.0), "6-9": (4.8, 3.0), "6-10": (5.4, 3.0), "6-11": (6.0, 3.0),
    "7-1": (0, 3.6), "7-11": (6.0, 3.6), "8-1": (0, 4.2), "8-11": (6.0, 4.2), "9-1": (0, 4.8), "9-11": (6.0, 4.8), "10-1": (0, 5.4), "10-11": (6.0, 5.4),
    "11-1": (0, 6.0), "11-2": (0.6, 6.0), "11-3": (1.2, 6.0), "11-4": (1.8, 6.0), "11-5": (2.4, 6.0),"11-6": (3.0, 6.0), "11-7": (3.6, 6.0), "11-8": (4.2, 6.0), "11-9": (4.8, 6.0), "11-10": (5.4, 6.0), "11-11": (6.0, 6.0)
}
def create_coord_tensor(dataset_classes, device):
    coords_list = [LABEL_TO_COORDS.get(c, (0,0)) for c in dataset_classes]
    return torch.tensor(coords_list, dtype=torch.float32).to(device)

def distance_weighted_ce(logits, targets, coord_tensor, alpha_dist=0.1):
    ce_loss = F.cross_entropy(logits, targets)
    probs = F.softmax(logits, dim=1) 
    gt_coords = coord_tensor[targets]
    dists = torch.norm(coord_tensor[None, :, :] - gt_coords[:, None, :], p=2, dim=2) 
    spatial_penalty = torch.mean(torch.sum(probs * dists, dim=1))
    return ce_loss + alpha_dist * spatial_penalty

def validate_process(model, pcn_s, pcn_t, source_val_loader, target_val_loader, device, coord_tensor):
    model.eval()
    total_correct_s = 0; total_s = 0; total_entropy_t = 0.0; num_batches_t = 0
    
    with torch.no_grad():
        for s_rssi, s_rtt, s_mask, s_label in source_val_loader:
            s_rssi, s_rtt, s_label = s_rssi.to(device), s_rtt.to(device), s_label.to(device)
            s_rtt_den, s_sig = apply_densification_and_uncertainty(s_rssi, s_rtt, pcn_s, MC_IDX, LEGACY_IDX, device)

            s_diff_rssi = s_rssi[:, 4:10]
            class_out_s, _, _, _ = model(s_diff_rssi, s_rtt_den, s_sig, coord_tensor, alpha=0)
            
            preds = torch.argmax(class_out_s, dim=1)
            total_correct_s += (preds == s_label).sum().item()
            total_s += s_label.size(0)

        for t_rssi, t_rtt, t_mask, _ in target_val_loader:
            t_rssi, t_rtt = t_rssi.to(device), t_rtt.to(device)
            t_rtt_den, t_sig = apply_densification_and_uncertainty(t_rssi, t_rtt, pcn_t, MC_IDX, LEGACY_IDX, device)

            t_diff_rssi = t_rssi[:, 4:10]
            _, _, _, softmax_t = model(t_diff_rssi, t_rtt_den, t_sig, coord_tensor, alpha=0)
            
            entropy = calc_entropy(softmax_t)
            total_entropy_t += entropy.mean().item()
            num_batches_t += 1

    return total_correct_s / total_s, total_entropy_t / num_batches_t

def evaluate_test(model, pcn, data_loader, coord_tensor, device, return_all_errors=False):
    model.eval()
    correct = 0; total = 0; total_dist = 0.0; all_dists = []
    
    with torch.no_grad():
        for rssi, rtt, mask, labels in data_loader:
            rssi, rtt, labels = rssi.to(device), rtt.to(device), labels.to(device)
            rtt_den, sig = apply_densification_and_uncertainty(rssi, rtt, pcn, MC_IDX, LEGACY_IDX, device)

            diff_rssi = rssi[:, 4:10]
            class_out, _, _, _ = model(diff_rssi, rtt_den, sig, coord_tensor, alpha=0)
            preds = torch.argmax(class_out, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            dist = torch.norm(coord_tensor[preds] - coord_tensor[labels], p=2, dim=1)
            total_dist += dist.sum().item()
            if return_all_errors: all_dists.extend(dist.cpu().numpy())
                
    if total == 0: return 0, 0, []
    return 100.*correct/total, total_dist/total, np.array(all_dists)

def main():
    results = []
    seed_candidate = [42, 67, 1024] 
    # seed_candidate = [10, 99, 6767, 423, 123456]

    SOURCE_CSV = os.path.join(args.base_path, '2026_1_1/all/All_Data_With_RSSI_Diff.csv')
    TARGET_CSV = os.path.join(args.base_path, '2026_2_4/All_Data_With_RSSI_Diff_withoutNA.csv')

    for seed in seed_candidate:
        set_seed(seed)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        s_rssi_raw, s_rtt_raw, s_mask_raw, s_labels_raw = load_raw_data(SOURCE_CSV)
        t_rssi_raw, t_rtt_raw, t_mask_raw, t_labels_raw = load_raw_data(TARGET_CSV)
        
        source_split_counts = [80, 20, 20] 
        target_split_counts = [40, 20, 20]
        s_tr_idx, s_val_idx, s_test_idx = get_stratified_indices(s_labels_raw, source_split_counts)
        t_tr_idx, t_val_idx, t_test_idx = get_stratified_indices(t_labels_raw, target_split_counts)

        rssi_scaler = MinMaxScaler(feature_range=(-1, 1))
        rtt_scaler = MinMaxScaler(feature_range=(-1, 1))
        label_encoder = LabelEncoder()

        rssi_scaler.fit(s_rssi_raw[s_tr_idx])
        rtt_scaler.fit(s_rtt_raw[s_tr_idx])
        label_encoder.fit(s_labels_raw[s_tr_idx])

        def create_dataset(rssi, rtt, rtt_mask, labels, indices):
            # 恢復原本的 transform
            r_t = rssi_scaler.transform(rssi[indices])
            rt_t = rtt_scaler.transform(rtt[indices])
            m_t = rtt_mask[indices]
            
            try: l_t = label_encoder.transform(labels[indices])
            except: l_t = np.zeros(len(indices))
            
            return TensorDataset(
                torch.tensor(r_t, dtype=torch.float32), 
                torch.tensor(rt_t, dtype=torch.float32), 
                torch.tensor(m_t, dtype=torch.bool), 
                torch.tensor(l_t, dtype=torch.long)
            )

        # def create_dataset(rssi, rtt, labels, indices):
        #     r_t = rssi_scaler.transform(rssi[indices])
        #     rt_t = rtt_scaler.transform(rtt[indices])
        #     try: l_t = label_encoder.transform(labels[indices])
        #     except: l_t = np.zeros(len(indices))
        #     return TensorDataset(torch.tensor(r_t), torch.tensor(rt_t), torch.tensor(l_t, dtype=torch.long))

        s_train = create_dataset(s_rssi_raw, s_rtt_raw, s_mask_raw, s_labels_raw, s_tr_idx)
        s_val = create_dataset(s_rssi_raw, s_rtt_raw, s_mask_raw, s_labels_raw, s_val_idx)
        s_test = create_dataset(s_rssi_raw, s_rtt_raw, s_mask_raw, s_labels_raw, s_test_idx)
        
        t_train = create_dataset(t_rssi_raw, t_rtt_raw, t_mask_raw, t_labels_raw, t_tr_idx)
        t_val = create_dataset(t_rssi_raw, t_rtt_raw, t_mask_raw, t_labels_raw, t_val_idx)
        t_test = create_dataset(t_rssi_raw, t_rtt_raw, t_mask_raw, t_labels_raw, t_test_idx)

        BATCH_SIZE = 32
        NUM_WORKERS = 0
        source_loader = DataLoader(s_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS)
        target_train_loader = DataLoader(t_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS)
        source_val_loader = DataLoader(s_val, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=NUM_WORKERS)
        target_val_loader = DataLoader(t_val, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=NUM_WORKERS)
        source_test_loader = DataLoader(s_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        target_test_loader = DataLoader(t_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        class_names = label_encoder.classes_
        COORD_TENSOR = create_coord_tensor(class_names, DEVICE)

        # ==========================================
        # 訓練 Stage 2: PCN 領域自適應
        # ==========================================
        ALL_APS_IDX = list(range(TOTAL_APS)) # [0, 1, 2, 3]

        # 1. Source Domain：擁有全部真實數據，訓練所有的 Head
        pcn_source = train_pcn(source_loader, ALL_APS_IDX, DEVICE, epochs=50, domain_name="Source")

        # 2. Target Domain：建立新模型並繼承 Source 的權重 (包含 Shared Net 與 所有的 Heads)
        pcn_target_init = PathLossCalibrationNetwork(in_dim=4, num_aps=TOTAL_APS).to(DEVICE)
        pcn_target_init.load_state_dict(pcn_source.state_dict())

        # 3. Target Domain 微調：只針對有真實標籤的 mcAP 進行訓練 (梯度會更新 Shared Net 與 mcAP 的 Head)
        pcn_target = train_pcn(target_train_loader, MC_IDX, DEVICE, epochs=50, domain_name="Target", pretrained_model=pcn_target_init)

        model = GeoSPANet(num_classes=len(class_names)).to(DEVICE)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
        domain_criterion = nn.BCEWithLogitsLoss(reduction='none')

        num_epochs = 500
        best_epoch = -1
        best_score = float('-inf')
        WARMUP_EPOCHS = 10
        temp_model_name = f"temp_model_{RTT_COMBO_NAME}_seed{seed}.pth"

        print(f"\nStart GeoSPA-Net Training Seed {seed}...")
        print(f"{'Epoch':<6} | {'Total Loss':<10} | {'Tr Cls':<10} | {'Tr Dom RSSI':<12} | {'Tr Dom RTT':<12} | {'Val T Acc':<10} | {'Val T Entropy':<12} | {'Test MDE':<8}")
        print("-" * 120)

        for epoch in range(num_epochs):
            model.train()

            total_loss_sum = 0.0
            train_cls_sum = 0.0
            train_dom_sum_rssi = 0.0
            train_dom_sum_rtt = 0.0
            num_batches = 0
            
            p = float(epoch) / num_epochs
            alpha = 2. / (1. + np.exp(-7.5 * p)) - 1
            
            for (s_rssi_b, s_rtt_b, s_mask_b, s_lbl_b), (t_rssi_b, t_rtt_b, t_mask_b, _) in zip(source_loader, target_train_loader):
                s_rssi_b, s_rtt_b, s_lbl_b = s_rssi_b.to(DEVICE), s_rtt_b.to(DEVICE), s_lbl_b.to(DEVICE)
                t_rssi_b, t_rtt_b = t_rssi_b.to(DEVICE), t_rtt_b.to(DEVICE)
                
                s_rtt_den, s_sig = apply_densification_and_uncertainty(s_rssi_b, s_rtt_b, pcn_source, MC_IDX, LEGACY_IDX, DEVICE)
                t_rtt_den, t_sig = apply_densification_and_uncertainty(t_rssi_b, t_rtt_b, pcn_target, MC_IDX, LEGACY_IDX, DEVICE)

                # 【關鍵修改】餵給 GeoSPANet 時，只取 index 4 到 9 (共 6 維原始差分)
                s_diff_rssi = s_rssi_b[:, 4:10]
                t_diff_rssi = t_rssi_b[:, 4:10]
                
                cls_out_s, d_logits_rssi_s, d_logits_rtt_s, softmax_s = model(s_diff_rssi, s_rtt_den, s_sig, COORD_TENSOR, alpha=alpha)
                _, d_logits_rssi_t, d_logits_rtt_t, softmax_t = model(t_diff_rssi, t_rtt_den, t_sig, COORD_TENSOR, alpha=alpha)

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
            
            val_s_acc, val_t_entropy = validate_process(model, pcn_source, pcn_target, source_val_loader, target_val_loader, DEVICE, COORD_TENSOR)
            
            save_mark = ""
            current_score = val_s_acc - (0.5 * val_t_entropy)

            if (epoch + 1) > WARMUP_EPOCHS:
                if current_score > best_score:
                    best_score = current_score
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), temp_model_name)
                    save_mark = f"(ADV {current_score:.2f})"
            
            if (epoch + 1) % 10 == 0:
                t_acc, t_mde, _ = evaluate_test(model, pcn_target, target_test_loader, COORD_TENSOR, DEVICE)
                print(f"{epoch+1:<6} | {total_loss_sum/num_batches:<10.4f} | {train_cls_sum/num_batches:<10.4f} | {train_dom_sum_rssi/num_batches:<12.4f} | {train_dom_sum_rtt/num_batches:<12.4f} | {val_s_acc:<10.4f} | {val_t_entropy:<12.4f} | {t_mde:<8.4f} | {save_mark:<10}")

        if best_epoch != -1:
            model.load_state_dict(torch.load(temp_model_name))
            if os.path.exists(temp_model_name): os.remove(temp_model_name)
            
        t_acc, t_mde, t_errors = evaluate_test(model, pcn_target, target_test_loader, COORD_TENSOR, DEVICE, return_all_errors=True)
        s_acc, s_mde, s_errors = evaluate_test(model, pcn_source, source_test_loader, COORD_TENSOR, DEVICE, return_all_errors=True)
        
        print(f"Seed {seed} | Src Acc: {s_acc:.4f} | Src MDE: {s_mde:.4f} | Tgt Acc: {t_acc:.4f} | Tgt MDE: {t_mde:.4f}")
        
        results.append({
            "Combo": RTT_COMBO_NAME,
            "Seed": seed,
            "Source_Acc": s_acc, "Source_MDE": s_mde,
            "Target_Acc": t_acc, "Target_MDE": t_mde
        })
        
        np.save(os.path.join(CDF_DIR, f"error_{RTT_COMBO_NAME}_seed{seed}.npy"), t_errors)

    df_res = pd.DataFrame(results)
    avg_s_acc, avg_s_acc_std = df_res['Source_Acc'].mean(), df_res['Source_Acc'].std()
    avg_s_mde, avg_s_mde_std = df_res['Source_MDE'].mean(), df_res['Source_MDE'].std()
    avg_t_acc, avg_t_acc_std = df_res['Target_Acc'].mean(), df_res['Target_Acc'].std()
    avg_t_mde, avg_t_mde_std = df_res['Target_MDE'].mean(), df_res['Target_MDE'].std()
    
    summary_file = os.path.join(RESULT_DIR, "experiment_summary.csv")
    file_exists = os.path.isfile(summary_file)
    
    with open(summary_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Combo", "Avg_Src_Acc", "Avg_Src_Acc_STD", "Avg_Src_MDE", "Avg_Src_MDE_STD", "Avg_Tgt_Acc", "Avg_Tgt_Acc_STD", "Avg_Tgt_MDE", "Avg_Tgt_MDE_STD", "Seeds_Detail"])
        writer.writerow([RTT_COMBO_NAME, f"{avg_s_acc:.4f}", f"{avg_s_acc_std:.4f}", f"{avg_s_mde:.4f}", f"{avg_s_mde_std:.4f}", f"{avg_t_acc:.4f}", f"{avg_t_acc_std:.4f}", f"{avg_t_mde:.4f}", f"{avg_t_mde_std:.4f}", str(seed_candidate)])
        
    print(f"Finished Combo {RTT_COMBO_NAME}. Results saved.")

if __name__ == '__main__':
    main()