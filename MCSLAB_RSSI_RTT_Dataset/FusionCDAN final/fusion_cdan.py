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
# 2. 模型架構：Dual Stream CDAN (修改處)
# ==========================================
class DualStreamCDAN(nn.Module):
    def __init__(self, num_aps=4, num_classes=49, hidden_dim=64, rtt_input_dim=1):
        super(DualStreamCDAN, self).__init__()
        self.num_classes = num_classes

        # --- 分支 B: RTT 特徵提取器 (動態輸入維度) ---
        self.rtt_extractor = nn.Sequential(
            nn.Linear(rtt_input_dim, 32), # 修改這裡：根據傳入的維度
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

        self.map_rssi = RandomizedMultiLinearMap(hidden_dim, num_classes, output_dim=512)
        self.disc_rssi = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

        self.map_rtt = RandomizedMultiLinearMap(hidden_dim, num_classes, output_dim=512)
        self.disc_rtt = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        
        self.grl = GradientReversalLayer()

    def forward(self, rssi, rtt, alpha=1.0):
        f_rssi = self.rssi_extractor(rssi)
        f_rtt = self.rtt_extractor(rtt)
        
        f_cat = torch.cat((f_rssi, f_rtt), dim=1)
        class_logits = self.class_classifier(f_cat)
        softmax_output = F.softmax(class_logits, dim=1)

        # Branch A
        h_rssi = self.map_rssi(f_rssi, softmax_output)
        h_rev_rssi = self.grl(h_rssi, alpha)
        d_logits_rssi = self.disc_rssi(h_rev_rssi)

        # Branch B
        h_rtt = self.map_rtt(f_rtt, softmax_output)
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
    rssi_cols = ['RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4'] 
    rtt_cols = rtt_cols_to_use 
    
    # 針對無效值處理
    for col in rssi_cols: df[col] = df[col].replace(-100, np.nan)
    for col in rtt_cols: df[col] = df[col].replace([0, -1], np.nan)

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

def validate_process(model, source_val_loader, target_val_loader, device):
    model.eval()
    total_cls_loss = 0.0
    total_rssi_dom_loss = 0.0
    total_rtt_dom_loss = 0.0
    num_batches = 0
    criterion_dom = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for (s_rssi, s_rtt, s_label), (t_rssi, t_rtt, _) in zip(source_val_loader, target_val_loader):
            s_rssi, s_rtt, s_label = s_rssi.to(device), s_rtt.to(device), s_label.to(device)
            t_rssi, t_rtt = t_rssi.to(device), t_rtt.to(device)
            
            class_out_s, d_logits_rssi_s, d_logits_rtt_s, _ = model(s_rssi, s_rtt, alpha=0) 
            _, d_logits_rssi_t, d_logits_rtt_t, _ = model(t_rssi, t_rtt, alpha=0)
            
            loss_cls = F.cross_entropy(class_out_s, s_label, reduction='sum')
            d_label_s = torch.ones(s_rssi.size(0), 1).to(device)
            d_label_t = torch.zeros(t_rssi.size(0), 1).to(device)
            
            loss_dom_rssi = criterion_dom(d_logits_rssi_s, d_label_s) + criterion_dom(d_logits_rssi_t, d_label_t)
            loss_dom_rtt = criterion_dom(d_logits_rtt_s, d_label_s) + criterion_dom(d_logits_rtt_t, d_label_t)
            
            total_cls_loss += loss_cls.item()
            total_rssi_dom_loss += loss_dom_rssi.item()
            total_rtt_dom_loss += loss_dom_rtt.item()
            num_batches += 1

    if num_batches == 0: return 0, 0, 0
    avg_cls = total_cls_loss / (num_batches * source_val_loader.batch_size)
    avg_rssi_dls = total_rssi_dom_loss / num_batches 
    avg_rtt_dls = total_rtt_dom_loss / num_batches 
    return avg_cls, avg_rssi_dls, avg_rtt_dls

def evaluate_test(model, data_loader, coord_tensor, device, return_all_errors=False):
    model.eval()
    correct = 0; total = 0; total_dist = 0.0; all_dists = []
    
    with torch.no_grad():
        for rssi, rtt, labels in data_loader:
            rssi, rtt, labels = rssi.to(device), rtt.to(device), labels.to(device)
            class_out, _, _, _ = model(rssi, rtt, alpha=0)
            preds = torch.argmax(class_out, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            dist = torch.norm(coord_tensor[preds] - coord_tensor[labels], p=2, dim=1)
            total_dist += dist.sum().item()
            
            if return_all_errors: 
                all_dists.extend(dist.cpu().numpy())
                
    if total == 0: return 0, 0, []
    return 100.*correct/total, total_dist/total, np.array(all_dists)

# ==========================================
# 3. 主程式
# ==========================================
def main():
    # 儲存結果用的 list
    results = []
    
    # seed_candidate = [42, 6767, 123456]
    seed_candidate = [42, 67, 1024] #2/4
    # seed_candidate = [67, 423, 123456] #3/17
    # seed_candidate = [10, 42, 99]
    
    for seed in seed_candidate:
        set_seed(seed)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 根據參數拼接路徑
        # 請確保資料夾路徑正確
        SOURCE_CSV = os.path.join(args.base_path, '2026_1_1/all/All_Data_With_RSSI_Diff.csv')
        TARGET_CSV = os.path.join(args.base_path, '2026_2_4/All_Data_With_RSSI_Diff.csv')
        # TARGET_CSV = os.path.join(args.base_path, '2026_3_17/All_Data_With_RSSI_Diff.csv')
        # TARGET_CSV = os.path.join(args.base_path, '2026_4_1/All_Data_With_RSSI_Diff.csv')

        # === 替換 main() 內的資料載入段落 ===
        s_rssi_raw, s_rtt_raw, s_labels_raw = load_raw_data(SOURCE_CSV, rtt_cols_to_use=RTT_COLS)
        t_rssi_raw, t_rtt_raw, t_labels_raw = load_raw_data(TARGET_CSV, rtt_cols_to_use=RTT_COLS)
        
        # 1. 取得切割索引
        source_split_counts = [80, 20, 20] 
        target_split_counts = [40, 20, 20]
        s_tr_idx, s_val_idx, s_test_idx = get_stratified_indices(s_labels_raw, source_split_counts)
        t_tr_idx, t_val_idx, t_test_idx = get_stratified_indices(t_labels_raw, target_split_counts)

        # ---------------------------------------------------------
        # 新增：2. 執行安全的類別平均值填補 (Source 與 Target 分開做)
        # ---------------------------------------------------------
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
        # ---------------------------------------------------------
        
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

        # 傳入 RTT_INPUT_DIM
        model = DualStreamCDAN(num_aps=4, num_classes=len(class_names), rtt_input_dim=RTT_INPUT_DIM).to(DEVICE)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
        domain_criterion = nn.BCEWithLogitsLoss(reduction='none')

        num_epochs = 400
        best_epoch = -1
        best_score = float('-inf')
        
        WARMUP_EPOCHS = 10
        CLS_THRESHOLD = 0.5 
        W_SCORE_CLS = 0.3      
        W_SCORE_DOM_1 = 0.1       
        W_SCORE_DOM_2 = 0.3
        
        # 臨時模型檔名
        temp_model_name = f"temp_model_{RTT_COMBO_NAME}_seed{seed}.pth"

        print(f"Start Training Seed {seed}...")
        print(f"\nStart CDAN+E Training... (Entropy Conditioning Enabled)")
        print(f"{'Epoch':<6} | {'Total Loss':<10} | {'Tr Cls':<10} | {'Tr Dom RSSI':<12} | {'Tr Dom RTT':<12} | {'Val Cls':<10} | {'Val RSSI Dls':<12} | {'Val RTT Dls':<12} | {'Test MDE':<8}")
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
            
            for (s_rssi_b, s_rtt_b, s_lbl_b), (t_rssi_b, t_rtt_b, _) in zip(source_loader, target_train_loader):
                s_rssi_b, s_rtt_b, s_lbl_b = s_rssi_b.to(DEVICE), s_rtt_b.to(DEVICE), s_lbl_b.to(DEVICE)
                t_rssi_b, t_rtt_b = t_rssi_b.to(DEVICE), t_rtt_b.to(DEVICE)
                
                cls_out_s, d_logits_rssi_s, d_logits_rtt_s, softmax_s = model(s_rssi_b, s_rtt_b, alpha=alpha)
                _, d_logits_rssi_t, d_logits_rtt_t, softmax_t = model(t_rssi_b, t_rtt_b, alpha=alpha)
                
                loss_cls = F.cross_entropy(cls_out_s, s_lbl_b)
                
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
            val_cls, val_rssi_dls, val_rtt_dls = validate_process(model, source_val_loader, target_val_loader, DEVICE)
            
            save_mark = ""
            current_dom_total = (val_rssi_dls + val_rtt_dls - 2.4)
            current_dom_diff = abs(val_rssi_dls - val_rtt_dls)
            current_score = (W_SCORE_DOM_1 * current_dom_total) - (W_SCORE_CLS * val_cls) - (W_SCORE_DOM_2 * current_dom_diff)

            if (epoch + 1) > WARMUP_EPOCHS and val_cls < CLS_THRESHOLD:
                if current_score > best_score:
                    best_score = current_score
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), temp_model_name)
                    save_mark = f"(ADV {current_score:.2f})"
            
            if (epoch + 1) % 1 == 0:
                t_acc, t_mde, _ = evaluate_test(model, target_test_loader, COORD_TENSOR, DEVICE)
                print(f"{epoch+1:<6} | {total_loss_sum/num_batches:<10.4f} | {train_cls_sum/num_batches:<10.4f} | {train_dom_sum_rssi/num_batches:<12.4f} | {train_dom_sum_rtt/num_batches:<12.4f} | {val_cls:<10.4f} | {val_rssi_dls:<12.4f} | {val_rtt_dls:<12.4f} | {t_mde:<8.4f} | {save_mark:<10}")

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
        writer.writerow([RTT_COMBO_NAME, f"{avg_s_acc:.4f}", f"{avg_s_acc_std:.4f}", f"{avg_s_mde:.4f}", f"{avg_s_mde_std:.4f}", f"{avg_t_acc:.4f}", f"{avg_t_acc_std:.4f}", f"{avg_t_mde:.4f}", f"{avg_t_mde_std:.4f}", str(seed_candidate)])
        
    print(f"Finished Combo {RTT_COMBO_NAME}. Results saved.")

if __name__ == '__main__':
    main()