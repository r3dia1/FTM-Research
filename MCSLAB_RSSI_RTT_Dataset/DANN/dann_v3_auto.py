# ===================== Version Info =============================
# 根據 version 2 的版本做自動化迴圈，測試平均 s/t acc/mde
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
# 2. 模型架構：單分支 DANN (Single Stream)
# ==========================================
class SingleStreamDANN(nn.Module):
    def __init__(self, input_dim=4, num_classes=5, hidden_dim=64):
        super(SingleStreamDANN, self).__init__()

        # --- 單一特徵提取器 ---
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True)
        )

        # --- 標籤分類器 (Task Classifier) ---
        self.class_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

        # --- 單一域分類器 ---
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 2) # Source=0, Target=1
        )
        
        self.grl = GradientReversalLayer()
    
    def forward(self, x, alpha=1.0):
        # 1. 提取特徵
        features = self.feature_extractor(x)

        # 2. 標籤預測
        class_output = self.class_classifier(features)

        # 3. 域預測 (加入 GRL)
        r_features = self.grl(features, alpha)
        domain_output = self.domain_classifier(r_features)

        return class_output, domain_output

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
    rssi_cols = ['RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4']
    # rssi_cols = ['Diff_RSSI_1_2', 'Diff_RSSI_1_3', 'Diff_RSSI_1_4', 'Diff_RSSI_2_3', 'Diff_RSSI_2_4', 'Diff_RSSI_3_4']
    # rssi_cols = ['RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4', 'Diff_RSSI_1_2', 'Diff_RSSI_1_3', 'Diff_RSSI_1_4', 'Diff_RSSI_2_3', 'Diff_RSSI_2_4', 'Diff_RSSI_3_4']
    # rtt_cols = ['Dist_mm_1', 'Dist_mm_2', 'Dist_mm_3', 'Dist_mm_4']
    rtt_cols = ['Dist_mm_2', 'Dist_mm_3', 'Dist_mm_4']
    # rtt_cols = ['Dist_mm_3', 'Dist_mm_4']
    # rtt_cols = ['Dist_mm_4']
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
    total_dom_loss = 0.0
    num_batches = 0
    
    # 使用 zip 同時遍歷 Source Val 和 Target Val
    # 這裡假設兩者 batch 數差不多，或以短的為準
    with torch.no_grad():
        for (s_data, s_label), (t_data,_) in zip(source_val_loader, target_val_loader):
            
            s_data, s_label = s_data.to(device), s_label.to(device)
            t_data = t_data.to(device)
            
            # Forward (驗證時 alpha設為0或1對GRL無影響，因為沒有Backward，但為了Domain Classifier有輸出，alpha參數不影響數值)
            # 但要注意：我們是要看 Domain Classifier 的 Loss，所以需要它的輸出
            class_out_s, d_s = model(s_data, alpha=0) 
            _, d_t = model(t_data, alpha=0)
            
            # 1. Class Loss (只算 Source)
            loss_cls = F.cross_entropy(class_out_s, s_label, reduction='sum')
            
            # 2. Domain Loss (Source=0, Target=1)
            # 建立 Domain Label
            d_label_s = torch.zeros(s_data.size(0), dtype=torch.long).to(device)
            d_label_t = torch.ones(t_data.size(0), dtype=torch.long).to(device)
            
            loss_dom_s = F.cross_entropy(d_s, d_label_s, reduction='sum')
            loss_dom_t = F.cross_entropy(d_t, d_label_t, reduction='sum')
            
            # 累積
            total_cls_loss += loss_cls.item()
            total_dom_loss += (loss_dom_s.item() + loss_dom_t.item())
            
            # 計算樣本總數 (Batch Size * 2 for domain, Batch Size for class)
            # 為了方便平均，我們這裡單純用 Batch 數量來做平均
            num_batches += 1

    if num_batches == 0:
        return 0, 0, 0

    # 回傳平均 Loss
    # 注意：這裡的平均計算方式是 (Total Sum / Num Batches)，也可以除以樣本數，只要標準一致即可
    avg_cls = total_cls_loss / (num_batches * source_val_loader.batch_size)
    avg_dom_loss = total_dom_loss / (num_batches * source_val_loader.batch_size * 2) # *2 因為有 source+target

    return avg_cls, avg_dom_loss

# 最終測試用的 Evaluate (含 MDE)
def evaluate_test(model, data_loader, coord_tensor, device, return_all_errors=False):
    model.eval()
    correct = 0; total = 0; total_dist = 0.0; all_dists = []
    with torch.no_grad():
        for x, labels in data_loader:
            x, labels = x.to(device), labels.to(device)
            class_out, _ = model(x, alpha=0)
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
    plt.plot(epochs, history['train_cls'], label='Train Cls Loss')
    plt.plot(epochs, history['train_dls'], label='Train Dom Loss')
    # plt.plot(epochs, history['dom_loss_rssi'], label='Train Dom RSSI')
    # plt.plot(epochs, history['dom_loss_rtt'], label='Train Dom RTT')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_cls'], label='Val Class Loss', color='red')
    plt.plot(epochs, history['val_dls'], label='Val Domain Loss', color='pink')
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
    for idx, (label, errors) in enumerate(errors_dict):
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
    USE_MODE = 'fusion'

    seed_candidate = [42, 6767, 123456]
    source_acc = []
    source_mde = []
    target_acc = []
    target_mde = []

    for seed in seed_candidate:
        set_seed(seed)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if USE_MODE == 'fusion':
            INPUT_DIM = 7
        elif USE_MODE == 'rssi':
            INPUT_DIM = 4 
        else:
            INPUT_DIM = 1

        SOURCE_CSV = '../2026_1_1/all/All_Data_With_RSSI_Diff.csv'  
        TARGET_CSV = '../2026_1_14/All_Data_With_RSSI_Diff.csv'

        print(f"Using device: {DEVICE}")

        SAMPLES_PER_LABEL = 120
        s_rssi, s_rtt, s_labels = load_wifi_data(SOURCE_CSV, is_source=True, samples_per_label=SAMPLES_PER_LABEL)
        t_rssi, t_rtt, t_labels = load_wifi_data(TARGET_CSV, is_source=False, samples_per_label=SAMPLES_PER_LABEL)

        # 選擇 Feature
        if USE_MODE == 'rtt':
            s_data = s_rtt
            t_data = t_rtt
        elif USE_MODE == 'rssi':
            s_data = s_rssi
            t_data = t_rssi
        else:
            s_data = torch.cat((s_rssi, s_rtt), dim=1)
            t_data = torch.cat((t_rssi, t_rtt), dim=1)

        full_source = TensorDataset(s_data, s_labels)
        full_target = TensorDataset(t_data, t_labels)

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

        model = SingleStreamDANN(input_dim=INPUT_DIM, num_classes=len(class_names)).to(DEVICE)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
        
        W_CLS = 1       
        W_DOM = 1    
        num_epochs = 400

        # 紀錄
        history = {'total_loss': [], 'train_cls': [], 'train_dls': [], 'val_cls': [], 'val_dls': []}
        
        # 最佳紀錄初始化
        min_val_cls_rec = float('inf')      # Validation Class Loss (越小越好)
        max_val_dom_rssi_rec = float('-inf') # Validation RSSI Dom Loss (越大越好)
        max_val_dom_rtt_rec = float('-inf')  # Validation RTT Dom Loss (越大越好)
        best_epoch = -1
        best_adv_score = float('-inf')

        WARMUP_EPOCHS = 10
        CLS_THRESHOLD = 0.5 # Validation Class Loss 門檻
        W_SCORE_CLS = 0.3        # 分數權重：分類損失 (扣分項)
        W_SCORE_DOM = 0.1        # 分數權重：域損失 (加分項，設小一點以保守起見)
        
        print(f"\nStart Training... (Validate on Source Val & Target Val)")
        print(f"{'Epoch':<6} | {'Total Loss':<10} | {'Train Cls':<10} | {'Train D_Loss':<14} | {'Val Cls':<10} | {'Val D_Loss':<10} | {'Save':<12} | {'Test MDE':<10} | {'Score':<7}")
        print("-" * 65)

        for epoch in range(num_epochs):
            # --- Training ---
            model.train()
            total_loss_sum = 0.0
            train_cls_sum = 0
            dom_sum = 0.0
            num_batches = 0

            p = float(epoch) / num_epochs
            alpha = 2. / (1. + np.exp(-5 * p)) - 1
            alpha = min(alpha, 0.3)
            
            for (s_data_b, s_label_b), (t_data_b,_) in zip(source_loader, target_train_loader):
                
                s_data_b, s_label_b = s_data_b.to(DEVICE), s_label_b.to(DEVICE)
                t_data_b = t_data_b.to(DEVICE)
                
                # --- Forward ---
                class_out, d_out_s = model(s_data_b, alpha=alpha)
                _, d_out_t = model(t_data_b, alpha=alpha)
                
                # --- Loss Calculation ---
                loss_class = F.cross_entropy(class_out, s_label_b)
                
                d_label_s = torch.zeros(s_data_b.size(0), dtype=torch.long).to(DEVICE)
                d_label_t = torch.ones(t_data_b.size(0), dtype=torch.long).to(DEVICE)
                
                loss_d = F.cross_entropy(d_out_s, d_label_s) + F.cross_entropy(d_out_t, d_label_t)
                loss = (W_CLS * loss_class) + (W_DOM * loss_d)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss_sum += loss.item()
                train_cls_sum += loss_class.item()
                dom_sum += loss_d.item()
                num_batches += 1

            # 計算 Training Avg Loss
            avg_total = total_loss_sum / num_batches
            avg_cls = train_cls_sum / num_batches
            avg_d_loss = dom_sum / num_batches

            history['total_loss'].append(total_loss_sum / len(source_loader))
            history['train_cls'].append(train_cls_sum / len(source_loader))
            history['train_dls'].append(dom_sum / len(source_loader))

            # ==========================================
            # [修改核心] Validation 階段
            # ==========================================
            # 使用 source_val_loader 計算 Class Loss
            # 使用 source_val + target_val 計算 Domain Loss
            val_cls, val_d_loss = validate_process(model, source_val_loader, target_val_loader, DEVICE)
            
            history['val_cls'].append(val_cls)
            history['val_dls'].append(val_d_loss)
            
            save_mark = ""
            
            # -------------------------------------------------------
            # 計算 Fitness Score (方向一：線性加權)
            # Score = (Dom_Loss) * 0.1 - (Cls_Loss) * 0.3
            # 6 RSSI_DIff + 4 RTT:  m / m / m
            # 4 RSSI + 4 RTT:       m / m / m
            # 6 RSSI_DIFF + 1 RTT:  m / m / m (dnn 1.1166m)
            # 4 RSSI + 1 RTT:       m / m / m (dnn 1.2909m)
            # -------------------------------------------------------
            current_score = (W_SCORE_DOM * val_d_loss) - (W_SCORE_CLS * val_cls)

            # --- 三關卡篩選 (基於 Validation Metrics) ---
            
            # 1. Time Check
            if (epoch + 1) > WARMUP_EPOCHS:            
                # 2. Quality Check (Class Loss on Source Validation)
                if val_cls < CLS_THRESHOLD:        
                    if current_score > best_adv_score:
                        best_adv_score = current_score
                        best_epoch = epoch + 1

                        torch.save(model.state_dict(), "best_model_adversarial.pth")
                        save_mark = f"(ADV {current_score:.2f})"

                    # 3. Strict Optimization Check
                    # 同時滿足：Val Class Loss 創新低 AND Val Domain Loss (RSSI & RTT) 創新高
                    # is_better = (
                    #     val_cls < min_val_cls_rec and
                    #     val_d_rssi > max_val_dom_rssi_rec and
                    #     val_d_rtt > max_val_dom_rtt_rec
                    # )
                    
                    # if is_better:
                    #     min_val_cls_rec = val_cls
                    #     max_val_dom_rssi_rec = val_d_rssi
                    #     max_val_dom_rtt_rec = val_d_rtt
                    #     best_epoch = epoch + 1
                        
                    #     torch.save(model.state_dict(), "best_model_adversarial.pth")
                    #     save_mark = "(ADV)"


            
            t_acc, t_mde, t_err = evaluate_test(model, target_test_loader, COORD_TENSOR, DEVICE, return_all_errors=True)

            print(f"{epoch+1:<6} | {avg_total:<10.4f} | {avg_cls:<10.4f} | {avg_d_loss:<14.4f} | {val_cls:<10.4f} | {val_d_loss:<10.4f} | {save_mark:<12} | {t_mde:<10.4f} | {current_score:<7.3f}")


        print("-" * 65)
        if best_epoch != -1:
            print(f"Best Model Saved at Epoch {best_epoch}")
            # print(f"Metrics -> Val Cls: {min_val_cls_rec:.4f}, Val D_RSSI: {max_val_dom_rssi_rec:.4f}, Val D_RTT: {max_val_dom_rtt_rec:.4f}")
        else:
            print("No model met the criteria.")
            
        # Plotting (省略，可依需要繪製 val_cls, val_d_rssi, val_d_rtt)
        plot_training_history(history)

        # Final Test
        if best_epoch != -1:
            model.load_state_dict(torch.load("best_model_adversarial.pth"))
            
        # 使用 Test Set 進行最終報告
        t_acc, t_mde, t_err = evaluate_test(model, target_test_loader, COORD_TENSOR, DEVICE, return_all_errors=True)
        s_acc, s_mde, s_err = evaluate_test(model, source_test_loader, COORD_TENSOR, DEVICE, return_all_errors=True)

        source_acc.append(s_acc)
        source_mde.append(s_mde)
        target_acc.append(t_acc)
        target_mde.append(t_mde)
        
        print(f"\n[Final Report] Source Acc: {s_acc:.4f}% | Source MDE: {s_mde:.4f}m | Target Acc: {t_acc:.4f}% | Target MDE: {t_mde:.4f}m")
        
        # 畫 CDF
        # plot_cdf(t_err)
        plt.figure()
        plt.plot(np.sort(t_err), np.linspace(0, 1, len(t_err)), label='Target Test')
        plt.plot(np.sort(s_err), np.linspace(0, 1, len(s_err)), label='Source Test')
        plt.legend(); plt.grid(); plt.savefig('final_cdf.png')
    
    avg_source_acc = np.mean(source_acc)
    avg_source_mde = np.mean(source_mde)
    avg_target_acc = np.mean(target_acc)
    avg_target_mde = np.mean(target_mde)
    print(f"source acc: {avg_source_acc:.4f}, source mde: {avg_source_mde:.4f}, target acc: {avg_target_acc:.4f}, target mde: {avg_target_mde:.4f}")

if __name__ == '__main__':
    main()