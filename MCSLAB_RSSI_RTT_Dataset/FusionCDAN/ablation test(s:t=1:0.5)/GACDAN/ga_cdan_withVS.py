
# ===================== Version Info =============================
# 修改自 version 3 fix optimized
# 1. 修正 load_wifi_data(): 改進資料洩漏問題
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
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.cm as cm

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
RTT_COMBO_NAME = "GACDAN_" + "_".join(rtt_indices)

print(f"==========================================")
print(f"Current Experiment: RTT Combination: {RTT_COLS}")
print(f"RTT Input Dimension: {RTT_INPUT_DIM}")
print(f"==========================================")

# 建立結果資料夾
RESULT_DIR = "results"
CDF_DIR = os.path.join(RESULT_DIR, "cdf_data")
PLOT_DIR = os.path.join(RESULT_DIR, "plots")
os.makedirs(CDF_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

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
            nn.Linear(4, 32),
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

        # 修改處：因為加入了座標 (x, y)，RandomizedMultiLinearMap 的類別維度從 num_classes 變為 num_classes + 2
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
        f_rssi = self.rssi_extractor(rssi)
        f_rtt = self.rtt_extractor(rtt)
        
        f_cat = torch.cat((f_rssi, f_rtt), dim=1)
        class_logits = self.class_classifier(f_cat)
        softmax_output = F.softmax(class_logits, dim=1)

        # --- 座標回歸輔助 (Coordinate Regression Auxiliary) ---
        # 利用機率分佈 softmax_output (B, C) 與 座標矩陣 coord_tensor (C, 2) 相乘
        # 得到預測的物理位置 expected_coords (B, 2)
        expected_coords = torch.mm(softmax_output, coord_tensor) # (B, 2)
        
        # 將類別機率與預測座標串接作為新的 Condition: [g \oplus Coord]
        # 維度變成 (B, C + 2)
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
    """只負責讀取與基本清理，絕對不做 Scaling"""
    df = pd.read_csv(csv_path)
    rssi_cols = ['Diff_RSSI_1_2', 'Diff_RSSI_1_3', 'Diff_RSSI_1_4', 'Diff_RSSI_2_3', 'Diff_RSSI_2_4', 'Diff_RSSI_3_4']
    # rssi_cols = ['RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4']
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
    
    # 新增：計算 Target Entropy
    total_entropy_t = 0.0
    num_batches_t = 0
    
    with torch.no_grad():
        # Source Validation
        for s_rssi, s_rtt, s_label in source_val_loader:
            s_rssi, s_rtt, s_label = s_rssi.to(device), s_rtt.to(device), s_label.to(device)
            class_out_s, _, _, _ = model(s_rssi, s_rtt, coord_tensor, alpha=0)
            
            # 計算 Source Accuracy (比 Loss 更直觀)
            preds = torch.argmax(class_out_s, dim=1)
            total_correct_s += (preds == s_label).sum().item()
            total_s += s_label.size(0)

        # Target Validation (只看 Entropy，不看 Domain Loss)
        for t_rssi, t_rtt, _ in target_val_loader:
            t_rssi, t_rtt = t_rssi.to(device), t_rtt.to(device)
            _, _, _, softmax_t = model(t_rssi, t_rtt, coord_tensor, alpha=0)
            
            # Entropy 計算: -sum(p * log(p))
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
    # dists = dist_matrix[targets]
    
    # 空間懲罰：對所有類別的預測機率進行加權，距離越遠權重越大
    # penalty = sum( prob_i * dist_i )
    spatial_penalty = torch.mean(torch.sum(probs * dists, dim=1))
    
    # 最終 Loss = 原始分類誤差 + 空間結構懲罰
    return ce_loss + alpha_dist * spatial_penalty

def plot_spatial_error_map(model, data_loader, coord_tensor, device, save_path):
    """
    繪製 2D 物理空間誤差向量圖。
    計算每個 RP (參考點) 的平均預測位置，並畫出真實位置指向預測位置的箭頭。
    """
    model.eval()
    true_coords = []
    pred_coords = []
    
    with torch.no_grad():
        for d_rssi, d_rtt, labels in data_loader:
            # 修改 1：修正變數錯置問題
            d_rssi = d_rssi.to(device)
            d_rtt = d_rtt.to(device)
            labels = labels.to(device)
            
            # 確保傳入 coord_tensor
            class_out, _, _, _ = model(d_rssi, d_rtt, coord_tensor, alpha=0.0)
            preds = torch.argmax(class_out, dim=1)

            true_coords.append(coord_tensor[labels].cpu().numpy())
            pred_coords.append(coord_tensor[preds].cpu().numpy())

    true_coords = np.concatenate(true_coords, axis=0)
    pred_coords = np.concatenate(pred_coords, axis=0)

    # 找出所有獨立的真實 RP 座標
    unique_true = np.unique(true_coords, axis=0)
    
    plt.figure(figsize=(10, 6))
    # 畫出實驗室所有 RP 的真實位置 (黑點)
    plt.scatter(unique_true[:, 0], unique_true[:, 1], c='black', marker='o', s=30, label='True RP Location')
    
    # 計算每個 RP 的平均預測位置並畫箭頭
    for rp in unique_true:
        mask = (true_coords == rp).all(axis=1)
        if not np.any(mask):
            continue
        # 該 RP 所有測試樣本的平均預測落點
        mean_pred = pred_coords[mask].mean(axis=0) 
        
        # 畫誤差向量箭頭 (從真實位置指向預測平均位置)
        plt.arrow(rp[0], rp[1], mean_pred[0] - rp[0], mean_pred[1] - rp[1], 
                  head_width=0.1, head_length=0.15, fc='red', ec='red', alpha=0.6, length_includes_head=True)

    plt.title('2D Spatial Error Vector Map (Mean Prediction per RP)')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_spatial_tsne(model, data_loader, coord_tensor, device, save_path):
    """
    繪製具備空間梯度的 t-SNE 特徵分佈圖。
    使用 PyTorch Hook 取得融合後的隱藏層特徵，並根據真實 X/Y 座標進行漸層著色。
    """
    model.eval()
    features = []
    labels_list = []
    
    # 修改 2 & 3：使用 Hook 攔截 class_classifier 的「輸入」
    # 這裡的 input[0] 就是模型 forward 裡的 f_cat (結合 RSSI 和 RTT 的特徵)
    def hook_fn(module, input, output):
        features.append(input[0].detach().cpu().numpy())
        
    # 註冊 Hook 到 class_classifier
    handle = model.class_classifier.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        for d_rssi, d_rtt, labels in data_loader:
            d_rssi = d_rssi.to(device)
            d_rtt = d_rtt.to(device)
            
            # 修改 2：觸發 forward 時，必須補上缺少的 coord_tensor 參數
            _ = model(d_rssi, d_rtt, coord_tensor, alpha=0.0) 
            labels_list.append(labels.cpu().numpy())
            
    # 取消註冊 Hook，避免影響後續訓練或推論操作
    handle.remove()
    
    features = np.concatenate(features, axis=0)
    labels_arr = np.concatenate(labels_list, axis=0)
    
    # 執行 t-SNE 降維
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(features)

    # 假設 labels_arr 裡面存的是參考點的 ID (例如 0, 1, 2... 代表不同的 (x,y) 組合)

    # 1. 挑選你想觀察的特定參考點 (例如隨機挑 6 個，或指定相鄰/容易混淆的幾個點)
    # 這裡假設我們挑選標籤為 0, 5, 10, 15, 20, 25 的參考點
    selected_labels = [0, 5, 10, 15, 20, 25]

    # 2. 過濾出這些特定參考點的特徵和標籤
    mask = np.isin(labels_arr, selected_labels)
    # filtered_tsne = tsne_results[mask]
    # filtered_labels = labels_arr[mask]
    filtered_tsne = tsne_results
    filtered_labels = labels_arr

    # 3. 繪製特徵分群圖
    plt.figure(figsize=(10, 8))

    # 使用具有高對比度的類別型色標 (例如 'tab10')，讓不同點的顏色對比鮮明
    scatter = plt.scatter(
        filtered_tsne[:, 0], 
        filtered_tsne[:, 1], 
        c=filtered_labels, 
        cmap='tab10', 
        alpha=0.8, 
        s=30 # 可以稍微把點放大一點以便觀察
    )

    # 加入圖例，標示哪個顏色代表哪個參考點 (Label)
    legend = plt.legend(*scatter.legend_elements(), title="Reference Point ID", loc="best")
    plt.gca().add_artist(legend)

    plt.title('Target Domain t-SNE (Clustering of Selected Reference Points)')
    plt.axis('off')
    plt.tight_layout()

    # 儲存或顯示圖片
    plt.savefig(save_path, dpi=500)
    # plt.show()
        
    # # 取得每個點對應的物理座標，用於漸層著色
    # phys_coords = coord_tensor[labels_arr].cpu().numpy()
    # x_coords = phys_coords[:, 0]
    # y_coords = phys_coords[:, 1]
    
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # # 子圖 1：依據 X 座標著色 (展示左右空間的潛在特徵連續性)
    # sc1 = ax1.scatter(tsne_results[:, 0], tsne_results[:, 1], c=x_coords, cmap='viridis', alpha=0.8, s=15)
    # ax1.set_title('Target Domain t-SNE (Colored by X Coordinate)')
    # ax1.axis('off') # 隱藏無意義的 t-SNE 軸座標
    # plt.colorbar(sc1, ax=ax1, label='X (meters)')
    
    # # 子圖 2：依據 Y 座標著色 (展示前後空間的潛在特徵連續性)
    # sc2 = ax2.scatter(tsne_results[:, 0], tsne_results[:, 1], c=y_coords, cmap='plasma', alpha=0.8, s=15)
    # ax2.set_title('Target Domain t-SNE (Colored by Y Coordinate)')
    # ax2.axis('off')
    # plt.colorbar(sc2, ax=ax2, label='Y (meters)')
    
    # plt.tight_layout()
    # plt.savefig(save_path, dpi=300)
    # plt.close()

# ==========================================
# 3. 主程式
# ==========================================
def main():
    # 儲存結果用的 list
    results = []
    
    # seed_candidate = [42, 6767, 123456]
    # seed_candidate = [1024, 42, 6767]
    seed_candidate = [10, 42, 423]

    # 根據參數拼接路徑
    # 請確保資料夾路徑正確
    SOURCE_CSV = os.path.join(args.base_path, '2026_1_1/all/All_Data_With_RSSI_Diff.csv')
    TARGET_CSV = os.path.join(args.base_path, '2026_2_4/All_Data_With_RSSI_Diff_withoutNA.csv')
    
    for seed in seed_candidate:
        set_seed(seed)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # === 替換 main() 內的資料載入段落 ===
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

        def create_dataset(rssi, rtt, labels, indices):
            r_t = rssi_scaler.transform(rssi[indices])
            rt_t = rtt_scaler.transform(rtt[indices])
            # 防止 target 出現未知的 label 報錯
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
        # ====================================

        
        BATCH_SIZE = 32
        NUM_WORKERS = 0
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
        print(f"{'Epoch':<6} | {'Total Loss':<10} | {'Tr Cls':<10} | {'Tr Dom RSSI':<12} | {'Tr Dom RTT':<12} | {'Val T Acc':<10} | {'Val T Entropy':<12} | {'Test MDE':<8}")
        print("-" * 120)

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
                cls_out_s, d_logits_rssi_s, d_logits_rtt_s, softmax_s = model(s_rssi_b, s_rtt_b, COORD_TENSOR, alpha=alpha)
                _, d_logits_rssi_t, d_logits_rtt_t, softmax_t = model(t_rssi_b, t_rtt_b, COORD_TENSOR, alpha=alpha)
                
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
            val_s_acc, val_t_entropy = validate_process(model, source_val_loader, target_val_loader, DEVICE, COORD_TENSOR)
            
            save_mark = ""
            current_score = val_s_acc - (0.5 * val_t_entropy)

            if (epoch + 1) > WARMUP_EPOCHS:
                if current_score > best_score:
                    best_score = current_score
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), temp_model_name)
                    save_mark = f"(ADV {current_score:.2f})"
            
            if (epoch + 1) % 1 == 0:
                t_acc, t_mde, _ = evaluate_test(model, target_test_loader, COORD_TENSOR, DEVICE)
                print(f"{epoch+1:<6} | {total_loss_sum/num_batches:<10.4f} | {train_cls_sum/num_batches:<10.4f} | {train_dom_sum_rssi/num_batches:<12.4f} | {train_dom_sum_rtt/num_batches:<12.4f} | {val_s_acc:<10.4f} | {val_t_entropy:<12.4f} | {t_mde:<8.4f} | {save_mark:<10}")
                # print(f"{epoch+1:<6} | {total_loss_sum/num_batches:<10.4f} | {train_cls_sum/num_batches:<10.4f} | {train_dom_sum_rssi/num_batches:<12.4f} | {train_dom_sum_rtt/num_batches:<12.4f} | {val_s_acc:<10.4f} | {val_t_entropy:<12.4f} | {save_mark:<10}")

        # Final Test & Save Results
        if best_epoch != -1:
            model.load_state_dict(torch.load(temp_model_name))
            os.remove(temp_model_name) # 測試完刪除模型檔
            
        t_acc, t_mde, t_errors = evaluate_test(model, target_test_loader, COORD_TENSOR, DEVICE, return_all_errors=True)
        s_acc, s_mde, s_errors = evaluate_test(model, source_test_loader, COORD_TENSOR, DEVICE, return_all_errors=True)
        
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

        # ---------------------------------------------------------
        # 新增：3. 執行視覺化繪圖 (針對 Target Test Data)
        # ---------------------------------------------------------
        print(f"Generating Visualizations for Seed {seed}...")
        
        # 繪製 2D 誤差向量圖
        map_save_path = os.path.join(PLOT_DIR, f"errormap_{RTT_COMBO_NAME}_seed{seed}.png")
        plot_spatial_error_map(model, target_test_loader, COORD_TENSOR, DEVICE, map_save_path)
        
        # 繪製空間著色的 t-SNE 圖
        tsne_save_path = os.path.join(PLOT_DIR, f"tsne_{RTT_COMBO_NAME}_seed{seed}.png")
        plot_spatial_tsne(model, target_test_loader, COORD_TENSOR, DEVICE, tsne_save_path)
        # ---------------------------------------------------------

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