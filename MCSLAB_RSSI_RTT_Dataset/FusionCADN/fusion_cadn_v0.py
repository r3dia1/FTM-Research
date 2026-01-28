# ===================== Version Info =============================
# 修改: Fusion CDAN+E (Conditional Domain Adversarial Network + Entropy)
# 基於論文: https://arxiv.org/pdf/1705.10667
# 核心改變:
# 1. Randomized Multilinear Map (特徵與預測的交互)
# 2. Entropy Conditioning (根據預測不確定性加權 Domain Loss)
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
import matplotlib.pyplot as plt

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
# 2. 核心組件：隨機多線性映射 (Randomized Multilinear Map)
# 論文 [cite: 133-137]
# ==========================================
class RandomizedMultiLinearMap(nn.Module):
    def __init__(self, feature_dim, num_classes, output_dim=1024):
        super(RandomizedMultiLinearMap, self).__init__()
        self.output_dim = output_dim
        # 隨機矩陣 R_f (固定不訓練)
        self.register_buffer('Rf', torch.randn(feature_dim, output_dim))
        # 隨機矩陣 R_g (固定不訓練)
        self.register_buffer('Rg', torch.randn(num_classes, output_dim))
        self.output_dim = output_dim

    def forward(self, f, g):
        # f: [batch, feature_dim], g: [batch, num_classes]
        # 論文公式 (6): 1/sqrt(d) * (Rf * f) element-wise-product (Rg * g)
        
        # 線性映射
        Rf_f = torch.mm(f, self.Rf)
        Rg_g = torch.mm(g, self.Rg)
        
        # 元素積 (Hadamard product) 並歸一化
        h = (Rf_f * Rg_g) / (self.output_dim ** 0.5)
        return h

# ==========================================
# 3. 模型架構：Dual Stream CDAN
# ==========================================
class DualStreamCDAN(nn.Module):
    def __init__(self, num_aps=4, num_classes=49, hidden_dim=64):
        super(DualStreamCDAN, self).__init__()
        self.num_classes = num_classes

        # --- 分支 B: RTT 特徵提取器 ---
        self.rtt_extractor = nn.Sequential(
            nn.Linear(2, 32),
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

        # --- 融合後的特徵維度 ---
        self.feature_dim = hidden_dim * 2

        # --- 標籤分類器 (Task Classifier) ---
        # 注意：這裡不包含 Softmax，因為 CrossEntropyLoss 會做
        self.class_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

        # --- CDAN 組件 ---
        # 1. 條件映射層 (將 Feature 與 Softmax Output 映射到高維空間)
        # self.map = RandomizedMultiLinearMap(self.feature_dim, num_classes, output_dim=1024)
        
        # 2. 條件域判別器 (Conditional Domain Discriminator)
        # # 輸入是 Map 的輸出 (1024維)
        # self.domain_classifier = nn.Sequential(
        #     nn.Linear(1024, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(True),
        #     nn.Dropout(0.5),
        #     nn.Linear(1024, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(True),
        #     nn.Dropout(0.5),
        #     nn.Linear(1024, 1) # 輸出 1 個 logit (用 BCEWithLogitsLoss)
        # )

        self.map_rssi = RandomizedMultiLinearMap(hidden_dim, num_classes, output_dim=512) # 維度可減半
        self.disc_rssi = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

        # Branch B: RTT 的條件映射與判別器
        self.map_rtt = RandomizedMultiLinearMap(hidden_dim, num_classes, output_dim=512)
        self.disc_rtt = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        
        self.grl = GradientReversalLayer()
    
    # def forward(self, rssi, rtt, alpha=1.0):
    #     # 1. 提取特徵
    #     f_rssi = self.rssi_extractor(rssi)
    #     f_rtt = self.rtt_extractor(rtt)
        
    #     # 2. 特徵融合 f
    #     f_cat = torch.cat((f_rssi, f_rtt), dim=1) # [Batch, feature_dim]

    #     # 3. 標籤預測 (Logits)
    #     class_logits = self.class_classifier(f_cat)
        
    #     # 4. 取得 Softmax 預測機率 g (用於 Conditioning)
    #     softmax_output = F.softmax(class_logits, dim=1) # [Batch, num_classes]

    #     # 5. CDAN Conditioning
    #     # 透過 GRL 反轉梯度 (放在 Map 之前或之後皆可，通常放在 Map 輸出的特徵上)
    #     # 這裡我們將 GRL 放在 Conditioning 之後，進入 Discriminator 之前
        
    #     # 隨機多線性映射 h = f (x) g
    #     h = self.map(f_cat, softmax_output) 
        
    #     # GRL
    #     h_rev = self.grl(h, alpha)
        
    #     # 域預測
    #     domain_logits = self.domain_classifier(h_rev)

    #     return class_logits, domain_logits, softmax_output

    def forward(self, rssi, rtt, alpha=1.0):
        # 1. 提取特徵
        f_rssi = self.rssi_extractor(rssi)
        f_rtt = self.rtt_extractor(rtt)
        
        # 2. 融合用於分類 (保持不變，因為分類還是要看兩者)
        f_cat = torch.cat((f_rssi, f_rtt), dim=1)
        class_logits = self.class_classifier(f_cat)
        softmax_output = F.softmax(class_logits, dim=1)

        # 3. 雙分支 CDAN Conditioning
        # Branch A
        h_rssi = self.map_rssi(f_rssi, softmax_output) # RSSI 特徵 x 預測
        h_rev_rssi = self.grl(h_rssi, alpha)
        d_logits_rssi = self.disc_rssi(h_rev_rssi)

        # Branch B
        h_rtt = self.map_rtt(f_rtt, softmax_output)    # RTT 特徵 x 預測
        h_rev_rtt = self.grl(h_rtt, alpha)
        d_logits_rtt = self.disc_rtt(h_rev_rtt)

        return class_logits, d_logits_rssi, d_logits_rtt, softmax_output

# ==========================================
# 輔助函式：計算 Entropy
# ==========================================
def calc_entropy(softmax_output):
    # H(g) = - sum( p * log(p) )
    # 加上 1e-5 避免 log(0)
    epsilon = 1e-5
    entropy = -torch.sum(softmax_output * torch.log(softmax_output + epsilon), dim=1)
    return entropy

# ==========================================
# 資料處理全域變數 (保持不變)
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
    print(f"Random Seed set to: {seed}")

def load_wifi_data(csv_path, is_source=True, samples_per_label=None):
    global is_scaler_fitted
    df = pd.read_csv(csv_path)
    # 使用與您原始代碼相同的特徵
    rssi_cols = ['Diff_RSSI_1_2', 'Diff_RSSI_1_3', 'Diff_RSSI_1_4', 'Diff_RSSI_2_3', 'Diff_RSSI_2_4', 'Diff_RSSI_3_4']
    # rtt_cols = ['Dist_mm_1', 'Dist_mm_2', 'Dist_mm_3', 'Dist_mm_4']
    # rtt_cols = ['Dist_mm_2', 'Dist_mm_3', 'Dist_mm_4']
    rtt_cols = ['Dist_mm_2', 'Dist_mm_3']
    # rtt_cols = ['Dist_mm_2']
    
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

# 坐標部分省略 (保持與原始代碼一致)
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

# ==========================================
# [Validation] CDAN+E 的驗證
# ==========================================
# def validate_process(model, source_val_loader, target_val_loader, device):
#     model.eval()
#     total_cls_loss = 0.0
#     total_dom_loss = 0.0
#     num_batches = 0
    
#     # 這裡只簡單計算 Loss，不使用 Entropy Weight (因為驗證集是用來觀察收斂的)
#     criterion_dom = nn.BCEWithLogitsLoss()
    
#     with torch.no_grad():
#         for (s_rssi, s_rtt, s_label), (t_rssi, t_rtt, _) in zip(source_val_loader, target_val_loader):
#             s_rssi, s_rtt, s_label = s_rssi.to(device), s_rtt.to(device), s_label.to(device)
#             t_rssi, t_rtt = t_rssi.to(device), t_rtt.to(device)
            
#             # Forward
#             class_out_s, d_logits_s, _ = model(s_rssi, s_rtt, alpha=0) 
#             _, d_logits_t, _ = model(t_rssi, t_rtt, alpha=0)
            
#             # 1. Class Loss
#             loss_cls = F.cross_entropy(class_out_s, s_label, reduction='sum')
            
#             # 2. Domain Loss (單一 Discriminator)
#             d_label_s = torch.ones(s_rssi.size(0), 1).to(device)  # Source = 1
#             d_label_t = torch.zeros(t_rssi.size(0), 1).to(device) # Target = 0
            
#             loss_dom = criterion_dom(d_logits_s, d_label_s) + criterion_dom(d_logits_t, d_label_t)
            
#             total_cls_loss += loss_cls.item()
#             total_dom_loss += loss_dom.item()
#             num_batches += 1

#     if num_batches == 0: return 0, 0
#     avg_cls = total_cls_loss / (num_batches * source_val_loader.batch_size)
#     avg_dom = total_dom_loss / num_batches # Domain loss is typically average over batch already
#     return avg_cls, avg_dom

# def evaluate_test(model, data_loader, coord_tensor, device, return_all_errors=False):
#     model.eval()
#     correct = 0; total = 0; total_dist = 0.0; all_dists = []
#     with torch.no_grad():
#         for rssi, rtt, labels in data_loader:
#             rssi, rtt, labels = rssi.to(device), rtt.to(device), labels.to(device)
#             class_out, _, _ = model(rssi, rtt, alpha=0)
#             preds = torch.argmax(class_out, dim=1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)
#             dist = torch.norm(coord_tensor[preds] - coord_tensor[labels], p=2, dim=1)
#             total_dist += dist.sum().item()
#             if return_all_errors: all_dists.extend(dist.cpu().numpy())
#     if total == 0: return 0, 0, []
#     return 100.*correct/total, total_dist/total, np.array(all_dists)
def validate_process(model, source_val_loader, target_val_loader, device):
    model.eval()
    total_cls_loss = 0.0
    total_dom_loss = 0.0
    num_batches = 0
    
    # 定義 Loss (這裡不使用 Entropy Weight，單純觀察收斂)
    criterion_dom = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for (s_rssi, s_rtt, s_label), (t_rssi, t_rtt, _) in zip(source_val_loader, target_val_loader):
            s_rssi, s_rtt, s_label = s_rssi.to(device), s_rtt.to(device), s_label.to(device)
            t_rssi, t_rtt = t_rssi.to(device), t_rtt.to(device)
            
            # Forward: 接收 4 個回傳值
            # class_out, d_rssi, d_rtt, softmax
            class_out_s, d_logits_rssi_s, d_logits_rtt_s, _ = model(s_rssi, s_rtt, alpha=0) 
            _, d_logits_rssi_t, d_logits_rtt_t, _ = model(t_rssi, t_rtt, alpha=0)
            
            # 1. Class Loss (只算 Source)
            loss_cls = F.cross_entropy(class_out_s, s_label, reduction='sum')
            
            # 2. Domain Loss (雙分支加總)
            # Source Label = 1, Target Label = 0
            d_label_s = torch.ones(s_rssi.size(0), 1).to(device)
            d_label_t = torch.zeros(t_rssi.size(0), 1).to(device)
            
            # 計算 RSSI 分支的 Domain Loss
            loss_dom_rssi = criterion_dom(d_logits_rssi_s, d_label_s) + \
                            criterion_dom(d_logits_rssi_t, d_label_t)
            
            # 計算 RTT 分支的 Domain Loss
            loss_dom_rtt = criterion_dom(d_logits_rtt_s, d_label_s) + \
                           criterion_dom(d_logits_rtt_t, d_label_t)
            
            # 加總 (您也可以在這裡乘上權重，例如 0.5，但在驗證階段直接看總和即可)
            loss_dom = loss_dom_rssi + loss_dom_rtt
            
            total_cls_loss += loss_cls.item()
            total_dom_loss += loss_dom.item()
            num_batches += 1

    if num_batches == 0: return 0, 0
    
    # 平均 Loss
    avg_cls = total_cls_loss / (num_batches * source_val_loader.batch_size)
    # Domain Loss 通常在 BCE 內部已經對 batch 做平均了，所以這裡除以 num_batches 即可
    avg_dom = total_dom_loss / num_batches 
    
    return avg_cls, avg_dom

def evaluate_test(model, data_loader, coord_tensor, device, return_all_errors=False):
    model.eval()
    correct = 0; total = 0; total_dist = 0.0; all_dists = []
    
    with torch.no_grad():
        for rssi, rtt, labels in data_loader:
            rssi, rtt, labels = rssi.to(device), rtt.to(device), labels.to(device)
            
            # Forward: 修改解包數量
            # 雙分支模型會回傳 4 個值，我們只需要第一個 (class_out)
            # 寫法 1: 使用 4 個變數接
            class_out, _, _, _ = model(rssi, rtt, alpha=0)
            
            # 寫法 2 (更簡潔): 只取第一個
            # class_out = model(rssi, rtt, alpha=0)[0]
            
            preds = torch.argmax(class_out, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # 計算歐式距離誤差
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
    seed_candidate = [42, 6767, 123456]
    for seed in seed_candidate:
        set_seed(seed)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        NUM_APS = 4
        # 路徑請自行修改
        SOURCE_CSV = '../2026_1_1/all/All_Data_With_RSSI_Diff.csv'  
        TARGET_CSV = '../2026_1_14/All_Data_With_RSSI_Diff.csv'

        print(f"Using device: {DEVICE}")

        SAMPLES_PER_LABEL = 120
        s_rssi, s_rtt, s_labels = load_wifi_data(SOURCE_CSV, is_source=True, samples_per_label=SAMPLES_PER_LABEL)
        t_rssi, t_rtt, t_labels = load_wifi_data(TARGET_CSV, is_source=False, samples_per_label=SAMPLES_PER_LABEL)

        full_source = TensorDataset(s_rssi, s_rtt, s_labels)
        full_target = TensorDataset(t_rssi, t_rtt, t_labels)
        
        # Split (維持原本邏輯)
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

        # 建立 CDAN 模型
        model = DualStreamCDAN(num_aps=NUM_APS, num_classes=len(class_names)).to(DEVICE)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
        
        # Loss Function for Domain (reduction='none' 為了乘上 entropy weight)
        domain_criterion = nn.BCEWithLogitsLoss(reduction='none')

        num_epochs = 400
        best_epoch = -1
        best_score = float('-inf')
        
        # 參數
        WARMUP_EPOCHS = 10
        CLS_THRESHOLD = 0.5 
        
        print(f"\nStart CDAN+E Training... (Entropy Conditioning Enabled)")
        print(f"{'Epoch':<6} | {'Total Loss':<10} | {'Tr Cls':<10} | {'Tr Dom RSSI':<10} | {'Tr Dom RTT':<10} | {'Val Cls':<10} | {'Val Dom':<10} | {'Test MDE':<8}")
        print("-" * 100)

        for epoch in range(num_epochs):
            model.train()
            total_loss_sum = 0.0
            train_cls_sum = 0.0
            train_dom_sum_rssi = 0.0
            train_dom_sum_rtt = 0.0
            num_batches = 0

            p = float(epoch) / num_epochs
            # alpha = 2. / (1. + np.exp(-10 * p)) - 1 # CDAN 常用的 schedule
            alpha = 2. / (1. + np.exp(-7.5 * p)) - 1
            # alpha = min(alpha, 0.3)
            
            for (s_rssi, s_rtt, s_lbl), (t_rssi, t_rtt, _) in zip(source_loader, target_train_loader):
                s_rssi, s_rtt, s_lbl = s_rssi.to(DEVICE), s_rtt.to(DEVICE), s_lbl.to(DEVICE)
                t_rssi, t_rtt = t_rssi.to(DEVICE), t_rtt.to(DEVICE)
                
                # 1. Forward Pass
                cls_out_s, d_logits_rssi_s, d_logits_rtt_s, softmax_s = model(s_rssi, s_rtt, alpha=alpha)
                _, d_logits_rssi_t, d_logits_rtt_t, softmax_t = model(t_rssi, t_rtt, alpha=alpha)
                
                # 2. 計算 Class Loss
                loss_cls = F.cross_entropy(cls_out_s, s_lbl)
                
                # 3. 計算 Entropy Weight 
                # w = 1 + exp(-H(g))
                entropy_s = calc_entropy(softmax_s)
                entropy_t = calc_entropy(softmax_t)
                weight_s = 1.0 + torch.exp(-entropy_s)
                weight_t = 1.0 + torch.exp(-entropy_t)
                
                # Normalize weights to avoid scaling loss too much (optional but recommended)
                weight_s = weight_s / torch.mean(weight_s)
                weight_t = weight_t / torch.mean(weight_t)

                # 4. 計算 Domain Loss (CDAN+E)
                # Source Label = 1, Target Label = 0
                d_lbl_s = torch.ones(s_rssi.size(0), 1).to(DEVICE)
                d_lbl_t = torch.zeros(t_rssi.size(0), 1).to(DEVICE)
                
                loss_dom_rssi_s = domain_criterion(d_logits_rssi_s, d_lbl_s)
                loss_dom_rssi_t = domain_criterion(d_logits_rssi_t, d_lbl_t)
                loss_dom_rtt_s = domain_criterion(d_logits_rssi_s, d_lbl_s)
                loss_dom_rtt_t = domain_criterion(d_logits_rssi_t, d_lbl_t)
                
                # Apply Entropy Conditioning
                loss_dom_rssi = torch.mean(weight_s.view(-1, 1) * loss_dom_rssi_s) + \
                           torch.mean(weight_t.view(-1, 1) * loss_dom_rssi_t)
                loss_dom_rtt = torch.mean(weight_s.view(-1, 1) * loss_dom_rtt_s) + \
                           torch.mean(weight_t.view(-1, 1) * loss_dom_rtt_t)
                
                # Total Loss
                # loss = loss_cls + loss_dom
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
            val_cls, val_dom = validate_process(model, source_val_loader, target_val_loader, DEVICE)
            
            # Score Logic (類似您的邏輯，但這裡 Dom Loss 越高代表 Confusion 越好)
            # 但注意：CDAN 的 discriminator 是要 minimize error，Generator 是 maximize error
            # 我們這裡 val_dom 是 Discriminator 的 Loss，理論上 Discriminator 越爛 (Loss 越高, close to 0.693 for BCE) 代表對齊越好
            current_score = -val_cls + (val_dom if val_dom < 1.0 else 0) # 簡易 Score

            # Checkpoint
            if (epoch + 1) > WARMUP_EPOCHS and val_cls < CLS_THRESHOLD:
                if current_score > best_score:
                    best_score = current_score
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), f"best_model_cdan_seed{seed}.pth")

            # Monitoring
            if (epoch + 1) % 1 == 0:
                t_acc, t_mde, _ = evaluate_test(model, target_test_loader, COORD_TENSOR, DEVICE)
                print(f"{epoch+1:<6} | {total_loss_sum/num_batches:<10.4f} | {train_cls_sum/num_batches:<10.4f} | {train_dom_sum_rssi/num_batches:<10.4f} | {train_dom_sum_rtt/num_batches:<10.4f} | {val_cls:<10.4f} | {val_dom:<10.4f} | {t_mde:<8.4f}")

        # Final Evaluation
        if best_epoch != -1:
            model.load_state_dict(torch.load(f"best_model_cdan_seed{seed}.pth"))
            print(f"Loaded Best Model from Epoch {best_epoch}")

        t_acc, t_mde, _ = evaluate_test(model, target_test_loader, COORD_TENSOR, DEVICE)
        s_acc, s_mde, _ = evaluate_test(model, source_test_loader, COORD_TENSOR, DEVICE)
        print(f"Seed {seed} Final Result -> Source MDE: {s_mde:.4f}m | Target MDE: {t_mde:.4f}m")

if __name__ == '__main__':
    main()