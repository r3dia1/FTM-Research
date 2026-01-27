# ===================== Version Info =============================
# Fusion DANN 大修正版本: 
# 更新正確驗證方式，以及用同時更新三個loss的規則來儲存最佳模型
# 有訓練收斂、CDF圖
# 缺少自動化迴圈
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
# 2. 模型架構：雙分支 DANN (RSSI + RTT)
# ==========================================
class DualStreamDANN(nn.Module):
    def __init__(self, num_aps=4, num_classes=49, hidden_dim=64):
        super(DualStreamDANN, self).__init__()

        # --- 分支 B: RTT 特徵提取器 ---
        self.rtt_extractor = nn.Sequential(
            nn.Linear(num_aps, 32),
            # nn.Linear(1, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True)
        )

        # --- 分支 A: RSSI 特徵提取器 ---
        self.rssi_extractor = nn.Sequential(
            # nn.Linear(num_aps, 32),
            nn.Linear(6, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True)
        )

        # --- 融合後的標籤分類器 (Task Classifier) ---
        self.class_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

        # --- 域分類器 A: RSSI ---
        self.domain_classifier_rssi = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 2) # Source=0, Target=1
        )

        # --- 域分類器 B: RTT ---
        self.domain_classifier_rtt = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 2)
        )
        
        self.grl = GradientReversalLayer()
    
    def forward(self, rssi, rtt, alpha=1.0, mode='fusion'):
        # 1. 提取特徵
        f_rssi = self.rssi_extractor(rssi)
        f_rtt = self.rtt_extractor(rtt)

        # ==========================================
        # [嚴謹測試核心] 強制遮蔽不需要的特徵分支
        # ==========================================
        if mode == 'rssi':
            f_rtt = torch.zeros_like(f_rtt).detach()
        elif mode == 'rtt':
            f_rssi = torch.zeros_like(f_rssi).detach()
            
        # 2. 標籤預測 (Concatenate)
        f_cat = torch.cat((f_rssi, f_rtt), dim=1)
        class_output = self.class_classifier(f_cat)

        # 3. 域預測 (加入 GRL)
        r_rssi = self.grl(f_rssi, alpha)
        domain_output_rssi = self.domain_classifier_rssi(r_rssi)

        r_rtt = self.grl(f_rtt, alpha)
        domain_output_rtt = self.domain_classifier_rtt(r_rtt)

        return class_output, domain_output_rssi, domain_output_rtt

# ==========================================
# 資料處理全域變數
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
    # rssi_cols = ['RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4']
    rssi_cols = ['Diff_RSSI_1_2', 'Diff_RSSI_1_3', 'Diff_RSSI_1_4', 'Diff_RSSI_2_3', 'Diff_RSSI_2_4', 'Diff_RSSI_3_4']
    rtt_cols = ['Dist_mm_1', 'Dist_mm_2', 'Dist_mm_3', 'Dist_mm_4']
    # rtt_cols = ['Dist_mm_1']

    for col in rssi_cols:
        df[col] = df[col].replace(-100, np.nan)
    for col in rtt_cols:
        df[col] = df[col].replace([0, -1], np.nan)

    if not is_source: 
        missing_count = df[rssi_cols + rtt_cols].isnull().sum().sum()
        total_cells = len(df) * (len(rssi_cols) + len(rtt_cols))
        missing_ratio = (missing_count / total_cells) * 100
        print(f"[Check] Target NaN count: {missing_count} ({missing_ratio:.2f}%)")

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
        print(f"[{'Source' if is_source else 'Target'}] Resampled to {samples_per_label} per label. Total: {len(df)}")

    rssi_data = df[rssi_cols].values.astype(np.float32)
    rtt_data = df[rtt_cols].values.astype(np.float32)
    raw_labels = df['Label'].values

    if is_source:
        rssi_data = rssi_scaler.fit_transform(rssi_data)
        rtt_data = rtt_scaler.fit_transform(rtt_data)
        labels = label_encoder.fit_transform(raw_labels)
        is_scaler_fitted = True
    else:
        if not is_scaler_fitted:
            raise ValueError("Error: Scaler not fitted. Load Source data first!")
        rssi_data = rssi_scaler.transform(rssi_data)
        rtt_data = rtt_scaler.transform(rtt_data)
        try:
            labels = label_encoder.transform(raw_labels)
        except:
            print("Warning: Unseen labels in Target domain.")
            labels = np.zeros(len(df))

    return torch.tensor(rssi_data), torch.tensor(rtt_data), torch.tensor(labels, dtype=torch.long)

def stratified_split(dataset, labels, split_counts):
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
        
    num_classes = len(np.unique(labels))
    train_indices = []
    val_indices = []
    test_indices = []
    
    unique_labels = np.unique(labels)
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        np.random.shuffle(label_indices)
        
        n_train, n_val, n_test = split_counts
        
        train_idx = label_indices[:n_train]
        val_idx = label_indices[n_train : n_train + n_val]
        test_idx = label_indices[n_train + n_val : n_train + n_val + n_test]
        
        train_indices.extend(train_idx)
        val_indices.extend(val_idx)
        test_indices.extend(test_idx)
        
    return (Subset(dataset, train_indices), 
            Subset(dataset, val_indices), 
            Subset(dataset, test_indices))

# ==========================================
# 座標映射設定
# ==========================================
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
        if cls_name in LABEL_TO_COORDS:
            coords_list.append(LABEL_TO_COORDS[cls_name])
        else:
            coords_list.append((0, 0))
    return torch.tensor(coords_list, dtype=torch.float32).to(device)

# ==========================================
# [新增] 驗證專用函式 (Validation Process)
# ==========================================
def validate_process(model, source_val_loader, target_val_loader, device):
    """
    計算驗證集的指標，用於模型挑選。
    1. Class Loss: 僅使用 Source Val
    2. Domain Loss: 使用 Source Val + Target Val (Target 不看 Label)
    """
    model.eval()
    
    # 累積變數
    total_cls_loss = 0.0
    total_dom_rssi = 0.0
    total_dom_rtt = 0.0
    num_batches = 0
    
    # 使用 zip 同時遍歷 Source Val 和 Target Val
    # 這裡假設兩者 batch 數差不多，或以短的為準
    with torch.no_grad():
        for (s_rssi, s_rtt, s_label), (t_rssi, t_rtt, _) in zip(source_val_loader, target_val_loader):
            # Target 的 Label 在這裡被丟棄 (_)
            
            s_rssi, s_rtt, s_label = s_rssi.to(device), s_rtt.to(device), s_label.to(device)
            t_rssi, t_rtt = t_rssi.to(device), t_rtt.to(device)
            
            # Forward (驗證時 alpha設為0或1對GRL無影響，因為沒有Backward，但為了Domain Classifier有輸出，alpha參數不影響數值)
            # 但要注意：我們是要看 Domain Classifier 的 Loss，所以需要它的輸出
            class_out_s, d_rssi_s, d_rtt_s = model(s_rssi, s_rtt, alpha=0) 
            _, d_rssi_t, d_rtt_t = model(t_rssi, t_rtt, alpha=0)
            
            # 1. Class Loss (只算 Source)
            loss_cls = F.cross_entropy(class_out_s, s_label, reduction='sum')
            
            # 2. Domain Loss (Source=0, Target=1)
            # 建立 Domain Label
            d_label_s = torch.zeros(s_rssi.size(0), dtype=torch.long).to(device)
            d_label_t = torch.ones(t_rssi.size(0), dtype=torch.long).to(device)
            
            loss_dom_rssi_s = F.cross_entropy(d_rssi_s, d_label_s, reduction='sum')
            loss_dom_rssi_t = F.cross_entropy(d_rssi_t, d_label_t, reduction='sum')
            
            loss_dom_rtt_s = F.cross_entropy(d_rtt_s, d_label_s, reduction='sum')
            loss_dom_rtt_t = F.cross_entropy(d_rtt_t, d_label_t, reduction='sum')
            
            # 累積
            total_cls_loss += loss_cls.item()
            total_dom_rssi += (loss_dom_rssi_s.item() + loss_dom_rssi_t.item())
            total_dom_rtt += (loss_dom_rtt_s.item() + loss_dom_rtt_t.item())
            
            # 計算樣本總數 (Batch Size * 2 for domain, Batch Size for class)
            # 為了方便平均，我們這裡單純用 Batch 數量來做平均
            num_batches += 1

    if num_batches == 0:
        return 0, 0, 0

    # 回傳平均 Loss
    # 注意：這裡的平均計算方式是 (Total Sum / Num Batches)，也可以除以樣本數，只要標準一致即可
    avg_cls = total_cls_loss / (num_batches * source_val_loader.batch_size)
    avg_dom_rssi = total_dom_rssi / (num_batches * source_val_loader.batch_size * 2) # *2 因為有 source+target
    avg_dom_rtt = total_dom_rtt / (num_batches * source_val_loader.batch_size * 2)

    return avg_cls, avg_dom_rssi, avg_dom_rtt

# 最終測試用的 Evaluate (含 MDE)
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

def plot_training_history(history):
    epochs = range(1, len(history['total_loss']) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['cls_loss'], label='Train Cls Loss')
    plt.plot(epochs, history['dom_loss_rssi'], label='Train Dom RSSI')
    plt.plot(epochs, history['dom_loss_rtt'], label='Train Dom RTT')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_loss'], label='Val Class Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Saved training_history.png")

def plot_cdf(errors_dict, filename='cdf_plot.png'):
    plt.figure(figsize=(8, 6))
    for idx, (label, errors) in enumerate(errors_dict.items()):
        sorted_errors = np.sort(errors)
        yvals = np.arange(len(sorted_errors)) / float(len(sorted_errors) - 1)
        plt.plot(sorted_errors, yvals, label=f'{label} (Mean: {np.mean(errors):.2f}m)', linewidth=2)
    plt.xlabel('Distance Error (m)')
    plt.ylabel('CDF')
    plt.title('CDF of Error')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, max(np.max(list(errors_dict.values())[0]), 5)) 
    plt.savefig(filename)
    print(f"Saved {filename}")

# ==========================================
# 3. 主程式
# ==========================================
def main():
    set_seed(123456)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_APS = 4
    SOURCE_CSV = '../2026_1_1/all/All_Data_With_RSSI_Diff.csv'  
    TARGET_CSV = '../2026_1_14/All_Data_With_RSSI_Diff.csv'

    print(f"Using device: {DEVICE}")

    SAMPLES_PER_LABEL = 120
    s_rssi, s_rtt, s_labels = load_wifi_data(SOURCE_CSV, is_source=True, samples_per_label=SAMPLES_PER_LABEL)
    t_rssi, t_rtt, t_labels = load_wifi_data(TARGET_CSV, is_source=False, samples_per_label=SAMPLES_PER_LABEL)

    full_source = TensorDataset(s_rssi, s_rtt, s_labels)
    full_target = TensorDataset(t_rssi, t_rtt, t_labels)

    # 為了計算 "Validation Class Loss"，Source 必須切出 Validation Set
    # Source: Train=80, Val=20, Test=20
    source_split_counts = [80, 20, 20] 
    # Target: Train=80 (for adaptation), Val=20 (for dom loss check), Test=20 (final report)
    target_split_counts = [80, 20, 20]
    
    s_train, s_val, s_test = stratified_split(full_source, s_labels, source_split_counts)
    t_train, t_val, t_test = stratified_split(full_target, t_labels, target_split_counts)

    BATCH_SIZE = 32
    # Loaders
    source_loader = DataLoader(s_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    target_train_loader = DataLoader(t_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # Validation Loaders (Shuffle=False)
    source_val_loader = DataLoader(s_val, batch_size=BATCH_SIZE, shuffle=False, drop_last=True) # drop_last 確保 batch 對齊
    target_val_loader = DataLoader(t_val, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    
    # Test Loaders
    source_test_loader = DataLoader(s_test, batch_size=BATCH_SIZE, shuffle=False)
    target_test_loader = DataLoader(t_test, batch_size=BATCH_SIZE, shuffle=False)

    class_names = label_encoder.classes_
    COORD_TENSOR = create_coord_tensor(class_names, DEVICE)

    model = DualStreamDANN(num_aps=NUM_APS, num_classes=len(class_names)).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    
    W_CLS = 1       
    W_DOM_RSSI = 1    
    W_DOM_RTT = 1    
    num_epochs = 400

    # 紀錄
    history = {'total_loss': [], 'train_cls': [], 'val_cls': [], 'val_d_rssi': [], 'val_d_rtt': []}
    
    # 最佳紀錄初始化
    min_val_cls_rec = float('inf')      # Validation Class Loss (越小越好)
    max_val_dom_rssi_rec = float('-inf') # Validation RSSI Dom Loss (越大越好)
    max_val_dom_rtt_rec = float('-inf')  # Validation RTT Dom Loss (越大越好)
    best_epoch = -1

    WARMUP_EPOCHS = 10
    CLS_THRESHOLD = 0.5 # Validation Class Loss 門檻
    
    print(f"\nStart Training... (Validate on Source Val & Target Val)")
    print(f"{'Epoch':<6} | {'Total Loss':<10} | {'Train Cls':<10} | {'Train D_RSSI':<10} | {'Train D_RTT':<10} | {'Val Cls':<10} | {'Val D_RSSI':<10} | {'Val D_RTT':<10} | {'Save'}")
    print("-" * 65)

    for epoch in range(num_epochs):
        # --- Training ---
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
            
            # Forward
            cls_out, d_rssi_s, d_rtt_s = model(s_rssi, s_rtt, alpha=alpha)
            _, d_rssi_t, d_rtt_t = model(t_rssi, t_rtt, alpha=alpha)
            
            # Loss
            l_cls = F.cross_entropy(cls_out, s_lbl)
            d_lbl_s = torch.zeros(s_rssi.size(0), dtype=torch.long).to(DEVICE)
            d_lbl_t = torch.ones(t_rssi.size(0), dtype=torch.long).to(DEVICE)
            
            l_d_rssi = F.cross_entropy(d_rssi_s, d_lbl_s) + F.cross_entropy(d_rssi_t, d_lbl_t)
            l_d_rtt = F.cross_entropy(d_rtt_s, d_lbl_s) + F.cross_entropy(d_rtt_t, d_lbl_t)
            
            loss = (W_CLS * l_cls) + (W_DOM_RSSI * l_d_rssi) + (W_DOM_RTT * l_d_rtt)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
            
        history['train_cls'].append(train_cls_sum / len(source_loader))

        # ==========================================
        # [修改核心] Validation 階段
        # ==========================================
        # 使用 source_val_loader 計算 Class Loss
        # 使用 source_val + target_val 計算 Domain Loss
        val_cls, val_d_rssi, val_d_rtt = validate_process(model, source_val_loader, target_val_loader, DEVICE)
        
        history['val_cls'].append(val_cls)
        history['val_d_rssi'].append(val_d_rssi)
        history['val_d_rtt'].append(val_d_rtt)
        
        save_mark = ""
        
        # --- 三關卡篩選 (基於 Validation Metrics) ---
        
        # 1. Time Check
        if (epoch + 1) > WARMUP_EPOCHS:
            
            # 2. Quality Check (Class Loss on Source Validation)
            if val_cls < CLS_THRESHOLD:
                
                # 3. Strict Optimization Check
                # 同時滿足：Val Class Loss 創新低 AND Val Domain Loss (RSSI & RTT) 創新高
                is_better = (
                    val_cls < min_val_cls_rec and
                    val_d_rssi > max_val_dom_rssi_rec and
                    val_d_rtt > max_val_dom_rtt_rec
                )
                
                if is_better:
                    min_val_cls_rec = val_cls
                    max_val_dom_rssi_rec = val_d_rssi
                    max_val_dom_rtt_rec = val_d_rtt
                    best_epoch = epoch + 1
                    
                    torch.save(model.state_dict(), "best_model_adversarial.pth")
                    save_mark = "(ADV)"

        print(f"{epoch+1:<6} | {avg_total:<10.4f} | {avg_cls:<10.4f} | {avg_d_rssi:<10.4f} | {avg_d_rtt:<10.4f} | {val_cls:<10.4f} | {val_d_rssi:<10.4f} | {val_d_rtt:<10.4f} | {save_mark:<8}")

    print("-" * 65)
    if best_epoch != -1:
        print(f"Best Model Saved at Epoch {best_epoch}")
        print(f"Metrics -> Val Cls: {min_val_cls_rec:.4f}, Val D_RSSI: {max_val_dom_rssi_rec:.4f}, Val D_RTT: {max_val_dom_rtt_rec:.4f}")
    else:
        print("No model met the criteria.")
        
    # Plotting (省略，可依需要繪製 val_cls, val_d_rssi, val_d_rtt)

    # Final Test
    if best_epoch != -1:
        model.load_state_dict(torch.load("best_model_adversarial.pth"))
        
    # 使用 Test Set 進行最終報告
    t_acc, t_mde, t_err = evaluate_test(model, target_test_loader, COORD_TENSOR, DEVICE, return_all_errors=True)
    s_acc, s_mde, s_err = evaluate_test(model, source_test_loader, COORD_TENSOR, DEVICE, return_all_errors=True)
    
    print(f"\n[Final Report] Source MDE: {s_mde:.4f}m | Target MDE: {t_mde:.4f}m")
    
    # 畫 CDF
    plt.figure()
    plt.plot(np.sort(t_err), np.linspace(0, 1, len(t_err)), label='Target Test')
    plt.plot(np.sort(s_err), np.linspace(0, 1, len(s_err)), label='Source Test')
    plt.legend(); plt.grid(); plt.savefig('final_cdf.png')

if __name__ == '__main__':
    main()