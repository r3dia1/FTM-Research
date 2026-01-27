# ===================== Version Info =============================
# 選擇儲存最佳模型的方式是透過 target 上帝視角偷看(不好)
# 沒有畫圖
# ================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.autograd import Function
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch.nn.functional as F
import os
import random

# ==========================================
# 1. 核心組件：梯度反轉層 (GRL) - (保持不變)
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
        # 注意：雙分支時輸入是 hidden_dim * 2，單分支這裡改為 hidden_dim
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
# 資料處理全域變數 & Seed 設定 (保持不變)
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
    
    # 根據你的設定選擇要讀取的欄位
    rssi_cols = ['RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4']
    # rssi_cols = ['Diff_RSSI_1_2', 'Diff_RSSI_1_3', 'Diff_RSSI_1_4', 'Diff_RSSI_2_3', 'Diff_RSSI_2_4', 'Diff_RSSI_3_4']
    rtt_cols = ['Dist_mm_1', 'Dist_mm_2', 'Dist_mm_3', 'Dist_mm_4']
    
    # 填補缺失值
    for col in rssi_cols:
        df[col] = df[col].replace(-100, np.nan)
    for col in rtt_cols:
        df[col] = df[col].replace([0, -1], np.nan)

    def fill_with_mean(x):
        return x.fillna(x.mean())

    cols_to_fix = rssi_cols + rtt_cols
    df[cols_to_fix] = df.groupby('Label')[cols_to_fix].transform(fill_with_mean)
    df[rssi_cols] = df[rssi_cols].fillna(-100)
    df[rtt_cols] = df[rtt_cols].fillna(-1)

    if samples_per_label is not None:
        df = df.groupby('Label').apply(
            lambda x: x.sample(n=samples_per_label, replace=True)
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
            labels = np.zeros(len(df))

    return torch.tensor(rssi_data), torch.tensor(rtt_data), torch.tensor(labels, dtype=torch.long)

# ==========================================
# 座標映射與評估函式 (修改為單輸入)
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

def evaluate(model, data_loader, coord_tensor, device):
    model.eval()
    correct = 0
    total_samples = 0
    total_dist_error = 0.0

    with torch.no_grad():
        for x_b, labels_b in data_loader: # 只取一個 input
            x_b, labels_b = x_b.to(device), labels_b.to(device)
            
            # 測試時 alpha=0
            class_out, _ = model(x_b, alpha=0.0)
            
            preds = torch.argmax(class_out, dim=1)
            correct += (preds == labels_b).sum().item()
            total_samples += labels_b.size(0)
            
            pred_coords = coord_tensor[preds]
            true_coords = coord_tensor[labels_b]
            distances = torch.norm(pred_coords - true_coords, p=2, dim=1)
            total_dist_error += distances.sum().item()

    avg_acc = 100. * correct / total_samples
    avg_mde = total_dist_error / total_samples
    return avg_acc, avg_mde

# ==========================================
# 3. 主程式
# ==========================================
def main():
    # ----------------------------------------
    # [設定] 選擇要跑的模式: 'rtt' 或 'rssi'
    # ----------------------------------------
    USE_MODE = 'fusion'  # <--- 在這裡切換模式
    
    # seed: 42, 6767, 123456
    set_seed(42)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 根據輸入決定 Input Dimension
    # 如果是 RSSI Diff 記得改成 6，如果是原始 RSSI 則是 4
    if USE_MODE == 'rtt':
        print('----- Using RTT only -----')
        INPUT_DIM = 4 
    elif USE_MODE == 'rssi':
        # 假設 RSSI 也是 4，如果是 Diff 模式請改成 6
        INPUT_DIM = 6 
        if INPUT_DIM == 4:
            print('----- Using RSSI only -----')
        else:
            print('----- Using RSSI Diff only -----')
    else:
        INPUT_DIM = 8
        print('----- Using fusion data -----')

    SOURCE_CSV = '../2026_1_1/all/All_Data_With_RSSI_Diff.csv'  
    # TARGET_CSV = '../2026_1_14/All_Data_With_RSSI_Diff.csv'
    TARGET_CSV = '../2026_1_2/All_Data_With_RSSI_Diff.csv'

    print(f"Using device: {DEVICE}")
    print(f"Running Single Stream DANN with mode: {USE_MODE}")

    # --- 1. 讀取數據 ---
    SAMPLES_PER_CLASS = 100 
    s_rssi, s_rtt, s_labels = load_wifi_data(SOURCE_CSV, is_source=True, samples_per_label=SAMPLES_PER_CLASS)
    t_rssi, t_rtt, t_labels = load_wifi_data(TARGET_CSV, is_source=False, samples_per_label=SAMPLES_PER_CLASS)

    # 選擇要用的數據
    if USE_MODE == 'rtt':
        s_data = s_rtt
        t_data = t_rtt
    elif USE_MODE == 'rssi':
        s_data = s_rssi
        t_data = t_rssi
    else:
        s_data = torch.cat((s_rssi, s_rtt), dim=1)
        t_data = torch.cat((t_rssi, t_rtt), dim=1)

    # 1. Target Train (Unlabeled)
    target_train_dataset = TensorDataset(t_data)
    
    # 2. Target Val/Test (Labeled)
    full_target_labeled_dataset = TensorDataset(t_data, t_labels)
    
    total_size = len(full_target_labeled_dataset)
    val_size = int(0.5 * total_size) 
    test_size = total_size - val_size
    
    target_val_dataset, target_test_dataset = random_split(
        full_target_labeled_dataset, [val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Target Data Split -> Val: {len(target_val_dataset)}, Test: {len(target_test_dataset)}")

    NUM_CLASSES = len(label_encoder.classes_)
    BATCH_SIZE = 32
    
    # Source (Input, Label)
    source_loader = DataLoader(TensorDataset(s_data, s_labels), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # Target Train (Input only)
    target_train_loader = DataLoader(target_train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # Target Val & Test (Input, Label)
    target_val_loader = DataLoader(target_val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    target_test_loader = DataLoader(target_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    class_names = label_encoder.classes_
    COORD_TENSOR = create_coord_tensor(class_names, DEVICE)

    # --- 2. 初始化單分支模型 ---
    model = SingleStreamDANN(input_dim=INPUT_DIM, num_classes=NUM_CLASSES).to(DEVICE)
    
    # 保持參數一致
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    
    # 權重設定 (只有一個 Domain loss 了)
    W_CLS = 1       
    W_DOM = 1       # 對應原本的 Domain Weight (建議保持 1 或根據實驗調整)

    # ==========================================
    # 開始訓練
    # ==========================================
    num_epochs = 400
    best_mde = float('inf')
    best_epoch = 0
    
    print(f"\nStart Training Single Stream ({USE_MODE})...")
    print("-" * 80)
    print(f"{'Epoch':<6} | {'Dom Acc':<10} | {'Tot Loss':<10} | {'Cls Loss':<10} | {'Dom Loss':<8} || {'Val MDE':<8}")
    print("-" * 80)

    for epoch in range(num_epochs):
        model.train()
        
        total_domain_acc = 0.0
        total_loss_sum = 0.0
        cls_loss_sum = 0.0
        dom_loss_sum = 0.0
        num_batches = 0
        
        p = float(epoch) / num_epochs
        alpha = 2. / (1. + np.exp(-5 * p)) - 1
        alpha = min(alpha, 0.3)
        
        # zip Source & Target
        for (s_data_b, s_label_b), (t_data_b,) in zip(source_loader, target_train_loader):
            
            s_data_b, s_label_b = s_data_b.to(DEVICE), s_label_b.to(DEVICE)
            t_data_b = t_data_b.to(DEVICE)
            
            # --- Forward ---
            class_out, d_out_s = model(s_data_b, alpha=alpha)
            _, d_out_t = model(t_data_b, alpha=alpha)
            
            # --- Loss Calculation ---
            # 1. Classification Loss
            loss_class = F.cross_entropy(class_out, s_label_b)
            
            # 2. Domain Loss
            d_label_s = torch.zeros(s_data_b.size(0), dtype=torch.long).to(DEVICE)
            d_label_t = torch.ones(t_data_b.size(0), dtype=torch.long).to(DEVICE)

            preds_s = torch.argmax(d_out_s, dim=1)
            preds_t = torch.argmax(d_out_t, dim=1)
            acc_s = (preds_s == 0).float().mean()
            acc_t = (preds_t == 1).float().mean()
            domain_acc = (acc_s + acc_t) / 2
            
            loss_d = F.cross_entropy(d_out_s, d_label_s) + F.cross_entropy(d_out_t, d_label_t)
            
            # 3. Total Loss
            loss = (W_CLS * loss_class) + (W_DOM * loss_d)
            
            # --- Backward ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_domain_acc += domain_acc.item()
            total_loss_sum += loss.item()
            cls_loss_sum += loss_class.item()
            dom_loss_sum += loss_d.item()
            num_batches += 1

        avg_domain_acc = total_domain_acc / num_batches
        avg_total = total_loss_sum / num_batches
        avg_cls = cls_loss_sum / num_batches
        avg_dom = dom_loss_sum / num_batches
        
        # --- Validation ---
        val_acc, val_mde = evaluate(model, target_val_loader, COORD_TENSOR, DEVICE)
        
        save_mark = ""
        if val_mde < best_mde:
            best_mde = val_mde
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "best_model_single.pth")
            save_mark = "*"
        
        print(f"{epoch+1:<6} | {avg_domain_acc:<10.4f} | {avg_total:<10.4f} | {avg_cls:<10.4f} | {avg_dom:<8.4f} || {val_mde:<8.4f} {save_mark}")

    print("-" * 80)
    print(f"Training Finished. Best Epoch: {best_epoch} with MDE: {best_mde:.4f}")
    
    # --- Final Test ---
    model.load_state_dict(torch.load("best_model_single.pth"))
    final_acc, final_mde = evaluate(model, target_test_loader, COORD_TENSOR, DEVICE)

    print("\n" + "="*50)
    print(f" FINAL REPORT ({USE_MODE} Single Stream)")
    print("="*50)
    print(f" Accuracy                : {final_acc:.2f}%")
    print(f" Mean Distance Error (MDE) : {final_mde:.4f} meters")
    print("="*50)

if __name__ == '__main__':
    main()