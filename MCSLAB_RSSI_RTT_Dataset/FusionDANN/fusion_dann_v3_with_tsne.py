# ===================== Version Info =============================
# 設計演算法來找最佳模型，詳細看 line 517 comment
# 根據 version 2 的版本做自動化迴圈，測試平均 s/t acc/mde
# [新增功能] 加入 t-SNE 視覺化特徵分佈
# ================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
from torch.autograd import Function
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.manifold import TSNE  # [新增 1] t-SNE 套件
import torch.nn.functional as F
import os
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm        # [新增 1] 繪圖顏色套件

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score
import numpy as np
import torch

def quantify_adaptation_quality(model, source_loader, target_loader, device, limit=1000):
    """
    數值化評估 Domain Adaptation 的品質，不看圖。
    """
    model.eval()
    
    # --- 1. 提取特徵 (Features) ---
    s_features = []
    s_labels = []
    t_features = []
    t_labels = []

    with torch.no_grad():
        # Source Data
        count = 0
        for rssi, rtt, label in source_loader:
            rssi, rtt = rssi.to(device), rtt.to(device)
            # 提取特徵 (concatenate RSSI & RTT features)
            f_rssi = model.rssi_extractor(rssi)
            f_rtt = model.rtt_extractor(rtt)
            f_cat = torch.cat((f_rssi, f_rtt), dim=1)
            
            s_features.append(f_cat.cpu().numpy())
            s_labels.append(label.cpu().numpy())
            count += len(label)
            if count >= limit: break
            
        # Target Data
        count = 0
        for rssi, rtt, label in target_loader:
            rssi, rtt = rssi.to(device), rtt.to(device)
            
            f_rssi = model.rssi_extractor(rssi)
            f_rtt = model.rtt_extractor(rtt)
            f_cat = torch.cat((f_rssi, f_rtt), dim=1)
            
            t_features.append(f_cat.cpu().numpy())
            t_labels.append(label.cpu().numpy())
            count += len(label)
            if count >= limit: break

    s_features = np.concatenate(s_features, axis=0)
    t_features = np.concatenate(t_features, axis=0)
    s_labels = np.concatenate(s_labels, axis=0)
    t_labels = np.concatenate(t_labels, axis=0)

    print("\n" + "="*40)
    print(" [Quantitative Analysis] Feature Alignment Metrics")
    print("="*40)

    # --- 指標 1: Domain Discriminability (Proxy A-Distance 概念) ---
    # 訓練一個簡單的 Logistic Regression 來分辨 Source vs Target
    # 標籤：Source=0, Target=1
    X_dom = np.vstack((s_features, t_features))
    y_dom = np.concatenate([np.zeros(len(s_features)), np.ones(len(t_features))])
    
    clf = LogisticRegression(max_iter=1000, solver='lbfgs')
    clf.fit(X_dom, y_dom)
    dom_acc = accuracy_score(y_dom, clf.predict(X_dom))
    
    print(f"1. Domain Confusion Score (Target: 50%):")
    print(f"   -> Result: {dom_acc*100:.2f}%")
    if 45 < dom_acc * 100 < 55:
        print("   -> 評語: 🌟 完美！特徵已經完全混合，無法分辨來源。")
    elif dom_acc * 100 > 90:
        print("   -> 評語: ⚠️ 糟糕。Source 和 Target 還是分得很開。")
    else:
        print("   -> 評語: ✅ 還不錯，有一定程度的混合。")

    # --- 指標 2: Class-wise Feature Distance (類別特徵中心距) ---
    # 計算每一個類別，其 Source 中心點與 Target 中心點的距離
    unique_classes = np.unique(s_labels)
    distances = []
    
    print(f"\n2. Class-wise Feature Drift (Target: 0.0):")
    for cls in unique_classes:
        # 取出該類別的所有特徵
        s_f_cls = s_features[s_labels == cls]
        t_f_cls = t_features[t_labels == cls]
        
        if len(s_f_cls) > 0 and len(t_f_cls) > 0:
            # 計算中心點 (Centroid)
            s_center = np.mean(s_f_cls, axis=0)
            t_center = np.mean(t_f_cls, axis=0)
            # 計算歐式距離
            dist = np.linalg.norm(s_center - t_center)
            distances.append(dist)
    
    avg_dist = np.mean(distances)
    print(f"   -> Average Distance: {avg_dist:.4f}")
    print("   -> 評語: 數值越小代表同一類別在不同場域的特徵越接近。")
    
    # 列出對齊最差的 3 個類別 (可以用來 debug 哪個位置飄最遠)
    # 這裡假設 distances 和 unique_classes 順序對應
    sorted_idx = np.argsort(distances)[::-1] # 降序
    print("   -> [Debug] Worst aligned classes (Top 3):")
    for i in range(min(3, len(distances))):
        idx = sorted_idx[i]
        print(f"      Class {unique_classes[idx]}: Dist = {distances[idx]:.4f}")

    # --- 指標 3: Silhouette Score (類別區分度) ---
    # 檢查混合後，類別是否還分得清楚 (避免 Negative Transfer 把不同類別混在一起)
    # 我們只看 Target Domain 的分類品質
    try:
        # 為了速度，隨機抽樣部分數據計算
        if len(t_features) > 2000:
            indices = np.random.choice(len(t_features), 2000, replace=False)
            t_sil_score = silhouette_score(t_features[indices], t_labels[indices])
        else:
            t_sil_score = silhouette_score(t_features, t_labels)
            
        print(f"\n3. Target Class Separability (Silhouette Score, Max: 1.0):")
        print(f"   -> Result: {t_sil_score:.4f}")
        if t_sil_score > 0.5:
             print("   -> 評語: 🌟 類別分界非常清晰。")
        elif t_sil_score < 0.1:
             print("   -> 評語: ⚠️ 類別模糊不清，可能混在一起了。")
    except:
        print("\n3. Silhouette Score: (Skip due to single class or data issue)")

    print("="*40 + "\n")

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
            # nn.Linear(num_aps, 32),
            nn.Linear(1, 32),
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
    # rtt_cols = ['Dist_mm_2', 'Dist_mm_3', 'Dist_mm_4']
    rtt_cols = ['Dist_mm_1']
    
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
            
            loss_dom_rssi_s = F.cross_entropy(d_rssi_s, d_label_s, reduction='sum')
            loss_dom_rssi_t = F.cross_entropy(d_rssi_t, d_label_t, reduction='sum')
            
            loss_dom_rtt_s = F.cross_entropy(d_rtt_s, d_label_s, reduction='sum')
            loss_dom_rtt_t = F.cross_entropy(d_rtt_t, d_label_t, reduction='sum')
            
            total_cls_loss += loss_cls.item()
            total_dom_rssi += (loss_dom_rssi_s.item() + loss_dom_rssi_t.item())
            total_dom_rtt += (loss_dom_rtt_s.item() + loss_dom_rtt_t.item())
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

def plot_training_history(history, seed):
    epochs = range(1, len(history['total_loss']) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_cls'], label='Train Cls Loss')
    plt.plot(epochs, history['train_d_rssi'], label='Train Dom Loss')
    plt.plot(epochs, history['train_d_rtt'], label='Train Dom Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_cls'], label='Val Class Loss', color='red')
    plt.plot(epochs, history['val_d_rssi'], label='Val RSSI Domain Loss', color='pink')
    plt.plot(epochs, history['val_d_rtt'], label='Val RTT Domain Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'training_history_seed{seed}.png')
    print("Saved training_history.png")

# ==========================================
# [新增 2] t-SNE 視覺化函式
# ==========================================
def visualize_tsne(model, source_loader, target_loader, device, seed, limit=1000):
    """
    提取特徵層 (f_rssi + f_rtt) 並繪製 t-SNE。
    顏色 = Class Label (Ground Truth), 形狀 = Domain (Source/Target)
    """
    print("Generating t-SNE visualization...")
    model.eval()
    features = []
    domain_labels = [] # 0=Source, 1=Target
    class_labels = []

    # 1. Source Data
    with torch.no_grad():
        count = 0
        for rssi, rtt, label in source_loader:
            rssi, rtt = rssi.to(device), rtt.to(device)
            
            # 這裡我們手動提取特徵，不經過 classifier
            f_rssi = model.rssi_extractor(rssi)
            f_rtt = model.rtt_extractor(rtt)
            f_cat = torch.cat((f_rssi, f_rtt), dim=1) # [Batch, Feature_Dim]
            
            features.append(f_cat.cpu().numpy())
            domain_labels.append(np.zeros(len(label)))
            class_labels.append(label.cpu().numpy())
            
            count += len(label)
            if count >= limit: break

    # 2. Target Data
    with torch.no_grad():
        count = 0
        for rssi, rtt, label in target_loader:
            rssi, rtt = rssi.to(device), rtt.to(device)
            
            f_rssi = model.rssi_extractor(rssi)
            f_rtt = model.rtt_extractor(rtt)
            f_cat = torch.cat((f_rssi, f_rtt), dim=1)
            
            features.append(f_cat.cpu().numpy())
            domain_labels.append(np.ones(len(label)))
            class_labels.append(label.cpu().numpy())
            
            count += len(label)
            if count >= limit: break

    # 3. Concatenate
    features = np.concatenate(features, axis=0)
    domain_labels = np.concatenate(domain_labels, axis=0)
    class_labels = np.concatenate(class_labels, axis=0)

    # 4. Run t-SNE
    print(f"Running t-SNE on {len(features)} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    emb_2d = tsne.fit_transform(features)

    # 5. Plotting
    plt.figure(figsize=(10, 8))
    
    # 設置顏色表 (支援多類別)
    unique_classes = np.unique(class_labels)
    colors = cm.rainbow(np.linspace(0, 1, len(unique_classes)))
    
    # 畫 Source (圓圈 'o')
    for cls, color in zip(unique_classes, colors):
        idx = (domain_labels == 0) & (class_labels == cls)
        if np.sum(idx) > 0:
            plt.scatter(emb_2d[idx, 0], emb_2d[idx, 1], 
                        c=[color], marker='o', label=f'Src-{cls}' if len(unique_classes)<10 else None, 
                        alpha=0.6, s=20)
    
    # 畫 Target (叉叉 'x')
    for cls, color in zip(unique_classes, colors):
        idx = (domain_labels == 1) & (class_labels == cls)
        if np.sum(idx) > 0:
            plt.scatter(emb_2d[idx, 0], emb_2d[idx, 1], 
                        c=[color], marker='x', label=f'Tgt-{cls}' if len(unique_classes)<10 else None, 
                        alpha=0.6, s=30)
            
    plt.title(f't-SNE Feature Visualization (Seed {seed})\nCircle=Source, Cross=Target, Color=Class')
    
    # 只有類別少的時候才顯示 Legend，不然會蓋住圖
    if len(unique_classes) <= 10:
        plt.legend(loc='best', fontsize='small')
        
    save_path = f'tsne_dann_seed_{seed}.png'
    plt.savefig(save_path)
    plt.close()
    print(f"t-SNE plot saved to {save_path}")

# ==========================================
# 3. 主程式
# ==========================================
def main():
    seed_candidate = [42, 6767, 123456]
    source_acc = []
    source_mde = []
    target_acc = []
    target_mde = []

    for seed in seed_candidate:
        set_seed(seed)
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

        model = DualStreamDANN(num_aps=NUM_APS, num_classes=len(class_names)).to(DEVICE)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
        
        W_CLS = 1       
        W_DOM_RSSI = 1    
        W_DOM_RTT = 1    
        num_epochs = 400

        history = {'total_loss': [], 'train_cls': [], 'train_d_rssi': [], 'train_d_rtt': [], 'val_cls': [], 'val_d_rssi': [], 'val_d_rtt': []}
        
        min_val_cls_rec = float('inf')      
        max_val_dom_rssi_rec = float('-inf') 
        max_val_dom_rtt_rec = float('-inf')  
        best_epoch = -1
        best_adv_score = float('-inf')

        WARMUP_EPOCHS = 10
        CLS_THRESHOLD = 0.5 
        W_SCORE_CLS = 0.3        
        W_SCORE_DOM_1 = 0.1        
        W_SCORE_DOM_2 = 0.3
        
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
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss_sum += loss.item()
                train_cls_sum += l_cls.item()
                dom_rssi_sum += l_d_rssi.item()
                dom_rtt_sum += l_d_rtt.item()
                num_batches += 1

            avg_total = total_loss_sum / num_batches
            avg_cls = train_cls_sum / num_batches
            avg_d_rssi = dom_rssi_sum / num_batches
            avg_d_rtt = dom_rtt_sum / num_batches
            
            history['total_loss'].append(total_loss_sum / len(source_loader))
            history['train_cls'].append(train_cls_sum / len(source_loader))
            history['train_d_rssi'].append(dom_rssi_sum / len(source_loader))
            history['train_d_rtt'].append(dom_rtt_sum / len(source_loader))

            val_cls, val_d_rssi, val_d_rtt = validate_process(model, source_val_loader, target_val_loader, DEVICE)
            
            history['val_cls'].append(val_cls)
            history['val_d_rssi'].append(val_d_rssi)
            history['val_d_rtt'].append(val_d_rtt)
            
            save_mark = ""
            current_dom_total = (val_d_rssi + val_d_rtt - 1.2)
            current_dom_diff = abs(val_d_rssi - val_d_rtt)
            current_score = (W_SCORE_DOM_1 * current_dom_total) - (W_SCORE_CLS * val_cls) - (W_SCORE_DOM_2 * current_dom_diff)

            if (epoch + 1) > WARMUP_EPOCHS:            
                if val_cls < CLS_THRESHOLD:        
                    if current_score > best_adv_score:
                        best_adv_score = current_score
                        best_epoch = epoch + 1
                        torch.save(model.state_dict(), f"best_model_adversarial_seed{seed}.pth")
                        save_mark = f"(ADV {current_score:.2f})"
            
            t_acc, t_mde, t_err = evaluate_test(model, target_test_loader, COORD_TENSOR, DEVICE, return_all_errors=True)

            print(f"{epoch+1:<6} | {avg_total:<10.4f} | {avg_cls:<10.4f} | {avg_d_rssi:<12.4f} | {avg_d_rtt:<11.4f} | {val_cls:<10.4f} | {val_d_rssi:<10.4f} | {val_d_rtt:<10.4f} | {save_mark:<11} | {t_mde:<8.4f} | {current_score:<7.3f}")

        print("-" * 135)
        if best_epoch != -1:
            print(f"Best Model Saved at Epoch {best_epoch}")
            model.load_state_dict(torch.load(f"best_model_adversarial_seed{seed}.pth"))
            
            # [新增 3] 呼叫 t-SNE 視覺化
            # 使用 Test set 來觀察未見過資料的分佈
            # visualize_tsne(model, source_test_loader, target_test_loader, DEVICE, seed, limit=800)
            quantify_adaptation_quality(model, source_test_loader, target_test_loader, DEVICE, limit=2000)
            
        else:
            print("No model met the criteria.")
            
        plot_training_history(history, seed)

        t_acc, t_mde, t_err = evaluate_test(model, target_test_loader, COORD_TENSOR, DEVICE, return_all_errors=True)
        s_acc, s_mde, s_err = evaluate_test(model, source_test_loader, COORD_TENSOR, DEVICE, return_all_errors=True)

        source_acc.append(s_acc)
        source_mde.append(s_mde)
        target_acc.append(t_acc)
        target_mde.append(t_mde)
        
        print(f"\n[Final Report] Source Acc: {s_acc:.4f}% | Source MDE: {s_mde:.4f}m | Target Acc: {t_acc:.4f}% | Target MDE: {t_mde:.4f}m")
        
        plt.figure()
        plt.plot(np.sort(t_err), np.linspace(0, 1, len(t_err)), label='Target Test')
        plt.plot(np.sort(s_err), np.linspace(0, 1, len(s_err)), label='Source Test')
        plt.legend(); plt.grid(); plt.savefig(f'final_cdf_seed{seed}.png')

    avg_source_acc = np.mean(source_acc)
    avg_source_mde = np.mean(source_mde)
    avg_target_acc = np.mean(target_acc)
    avg_target_mde = np.mean(target_mde)
    print(f"source acc: {avg_source_acc:.4f}, source mde: {avg_source_mde:.4f}, target acc: {avg_target_acc:.4f}, target mde: {avg_target_mde:.4f}")

if __name__ == '__main__':
    main()