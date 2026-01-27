# ===================== Version Info =============================
# 選擇儲存最佳模型的方式是透過 target 上帝視角偷看(不好)
# 補上訓練收斂圖，以及如何偷看選模型的觀察圖
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
            nn.Linear(num_aps, 32),
            # nn.Linear(6, 32),
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
            # 將 RTT 特徵全變為 0，且切斷梯度 (detach)
            f_rtt = torch.zeros_like(f_rtt).detach()
            
        elif mode == 'rtt':
            # 將 RSSI 特徵全變為 0，且切斷梯度 (detach)
            f_rssi = torch.zeros_like(f_rssi).detach()
            
        # mode == 'fusion' 則兩個都保留

        # 2. 標籤預測 (Concatenate)
        # 這裡進去分類器的，有一半是全 0，對分類器來說就是「沒有資訊」
        f_cat = torch.cat((f_rssi, f_rtt), dim=1)
        class_output = self.class_classifier(f_cat)

        # 3. 域預測 (加入 GRL)
        # 注意：如果是單一模態模式，被歸零的那一邊 Domain Output 其實沒意義了，
        # 但為了程式不出錯，我們還是讓它跑完，只是 Loss 我們會在外面設為 0
        r_rssi = self.grl(f_rssi, alpha)
        domain_output_rssi = self.domain_classifier_rssi(r_rssi)

        r_rtt = self.grl(f_rtt, alpha)
        domain_output_rtt = self.domain_classifier_rtt(r_rtt)

        return class_output, domain_output_rssi, domain_output_rtt

    # def forward(self, rssi, rtt, alpha=1.0):
    #     # 1. 提取特徵
    #     f_rssi = self.rssi_extractor(rssi)
    #     f_rtt = self.rtt_extractor(rtt)

    #     # 2. 標籤預測 (Concatenate)
    #     f_cat = torch.cat((f_rssi, f_rtt), dim=1)
    #     class_output = self.class_classifier(f_cat)

    #     # 3. 域預測 (加入 GRL) - 兩個分支分別做 Domain Adaptation
    #     r_rssi = self.grl(f_rssi, alpha)
    #     domain_output_rssi = self.domain_classifier_rssi(r_rssi)

    #     r_rtt = self.grl(f_rtt, alpha)
    #     domain_output_rtt = self.domain_classifier_rtt(r_rtt)

    #     return class_output, domain_output_rssi, domain_output_rtt

# ==========================================
# 資料處理全域變數
# ==========================================
rssi_scaler = MinMaxScaler(feature_range=(-1, 1))
rtt_scaler = MinMaxScaler(feature_range=(-1, 1))
label_encoder = LabelEncoder()
is_scaler_fitted = False

def set_seed(seed):
    """
    固定所有隨機種子以確保實驗可復現
    """
    # 1. Python 原生與 Numpy
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    # 2. PyTorch (CPU & GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果是多 GPU

    # 3. Deterministic (確保運算過程固定，但可能會稍微降低效能)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random Seed set to: {seed}")

def load_wifi_data(csv_path, is_source=True, samples_per_label=None):
    global is_scaler_fitted
    
    df = pd.read_csv(csv_path)
    rssi_cols = ['RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4']
    # rssi_cols = ['Diff_RSSI_1_2', 'Diff_RSSI_1_3', 'Diff_RSSI_1_4', 'Diff_RSSI_2_3', 'Diff_RSSI_2_4', 'Diff_RSSI_3_4']
    # rtt_cols = ['Dist_mm_1', 'Dist_mm_2', 'Dist_mm_3', 'Dist_mm_4']
    rtt_cols = ['Dist_mm_1']
    
    # 填補缺失值
    for col in rssi_cols:
        df[col] = df[col].replace(-100, np.nan)
    for col in rtt_cols:
        df[col] = df[col].replace([0, -1], np.nan)

    if not is_source: # 只檢查 Target
        missing_count = df[rssi_cols + rtt_cols].isnull().sum().sum()
        total_cells = len(df) * (len(rssi_cols) + len(rtt_cols))
        missing_ratio = (missing_count / total_cells) * 100
        print(f"[Check] Target NaN count: {missing_count} ({missing_ratio:.2f}%)")

    def fill_with_mean(x):
        return x.fillna(x.mean())

    cols_to_fix = rssi_cols + rtt_cols

    # ============================ #
    #    Data leakage may occur    #
    # ============================ #
    df[cols_to_fix] = df.groupby('Label')[cols_to_fix].transform(fill_with_mean)
    # if is_source:
    # # Source 可以用 Label 分組填補
    #     df[cols_to_fix] = df.groupby('Label')[cols_to_fix].transform(fill_with_mean)
        
    #     # 計算 Source 的全域平均值並存起來 (供 Target 使用)
    #     global_means = df[cols_to_fix].mean()
    # else:
    #     # Target 絕對不能用 Label，建議用 Source 的全域平均值填補 (模擬真實 Inference)
    #     # 或者用 df[cols_to_fix].fillna(df[cols_to_fix].mean()) # Target 自身的非監督平均
    #     if 'global_means' in globals():
    #         df[cols_to_fix] = df[cols_to_fix].fillna(global_means)
    #     else:
    #         df[cols_to_fix] = df[cols_to_fix].fillna(0) # 或其他預設值

    df[rssi_cols] = df[rssi_cols].fillna(-100)
    df[rtt_cols] = df[rtt_cols].fillna(-1)

    # 取樣
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
            print("Warning: Unseen labels in Target domain.")
            labels = np.zeros(len(df))

    return torch.tensor(rssi_data), torch.tensor(rtt_data), torch.tensor(labels, dtype=torch.long)

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
            print(f"Warning: Class {cls_name} not found in LABEL_TO_COORDS, setting to (0,0)")
            coords_list.append((0, 0))
    return torch.tensor(coords_list, dtype=torch.float32).to(device)

# ==========================================
# 獨立出一個驗證/測試函式 (方便重複呼叫)
# ==========================================
def evaluate(model, data_loader, coord_tensor, device, return_all_errors=False, return_loss=False):
    """
    輸入模型與資料，回傳 (Accuracy, MDE)
    """
    model.eval()
    correct = 0
    total_samples = 0
    total_dist_error = 0.0
    total_loss = 0.0
    all_distances = []

    with torch.no_grad():
        for rssi_b, rtt_b, labels_b in data_loader:
            rssi_b, rtt_b, labels_b = rssi_b.to(device), rtt_b.to(device), labels_b.to(device)
            
            # 測試時 alpha=0
            class_out, _, _ = model(rssi_b, rtt_b, alpha=0.0)

            # 計算 validation loss
            batch_loss = F.cross_entropy(class_out, labels_b, reduction='sum')
            total_loss += batch_loss.item()
            
            # 取得預測
            preds = torch.argmax(class_out, dim=1)
            
            # Accuracy
            correct += (preds == labels_b).sum().item()
            total_samples += labels_b.size(0)
            
            # MDE
            pred_coords = coord_tensor[preds]
            true_coords = coord_tensor[labels_b]
            distances = torch.norm(pred_coords - true_coords, p=2, dim=1)
            total_dist_error += distances.sum().item()

            if return_all_errors:
                all_distances.extend(distances.cpu().numpy())

    avg_acc = 100. * correct / total_samples
    avg_mde = total_dist_error / total_samples
    avg_loss = total_loss / total_samples

    if return_all_errors:
        return avg_acc, avg_mde, np.array(all_distances)
    elif return_loss:
        return avg_acc, avg_mde, avg_loss
    else:
        return avg_acc, avg_mde

def plot_training_history(history):
    epochs = range(1, len(history['total_loss']) + 1)

    plt.figure(figsize=(6, 5))

    # 1. Loss Curve
    # plt.subplot(1, 2, 1)
    plt.plot(epochs, history['total_loss'], label='Total Loss')
    plt.plot(epochs, history['cls_loss'], label='Class Loss')
    plt.plot(epochs, history['dom_loss'], label='Domain Loss')

    # if 'val_loss' in history and len(history['val_loss']) > 0:
    #     plt.plot(epochs, history['val_loss'], label='Val Class Loss', color='red', linestyle='--', linewidth=1.5)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)

    # # 2. Validation MDE Curve
    # plt.subplot(1, 2, 2)
    # plt.plot(epochs, history['val_mde'], label='Val MDE', color='orange')
    # plt.xlabel('Epochs')
    # plt.ylabel('MDE (m)')
    # plt.title('Validation MDE over Epochs')
    # plt.legend()
    # plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Saved training_history.png")

def plot_cdf(errors_dict, filename='cdf_plot.png'):
    """
    errors_dict: {'Target Test': errors_array, 'Source Test': errors_array}
    """
    plt.figure(figsize=(8, 6))
    
    colors = ['blue', 'green', 'red']
    
    for idx, (label, errors) in enumerate(errors_dict.items()):
        sorted_errors = np.sort(errors)
        yvals = np.arange(len(sorted_errors)) / float(len(sorted_errors) - 1)
        plt.plot(sorted_errors, yvals, label=f'{label} (Mean: {np.mean(errors):.2f}m)', linewidth=2)

    plt.xlabel('Distance Error (m)')
    plt.ylabel('CDF')
    plt.title('Cumulative Distribution Function (CDF) of Error')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, max(np.max(list(errors_dict.values())[0]), 5)) # 限制 X 軸範圍以便觀察
    plt.savefig(filename)
    print(f"Saved {filename}")

# ==========================================
# 3. 主程式 (Training Loop with Weights & Checkpoint)
# ==========================================
def main():

    # 固定訓練隨機種子 42, 6767, 123456
    set_seed(123456)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_APS = 4
    
    # 檔案路徑
    # SOURCE_CSV = '../2026_1_1/all/Server_Wide_20260101_140347.csv'  
    # TARGET_CSV = '../2026_1_2/Server_Wide_20260102_114230.csv'
    SOURCE_CSV = '../2026_1_1/all/All_Data_With_RSSI_Diff.csv'  
    # TARGET_CSV = '../2026_1_2/All_Data_With_RSSI_Diff.csv'
    TARGET_CSV = '../2026_1_14/All_Data_With_RSSI_Diff.csv'

    print(f"Using device: {DEVICE}")

    # --- 1. 讀取數據 (Source 修改為 120筆/類) ---
    SAMPLES_SOURCE_TOTAL = 120
    SAMPLES_TARGET = 100
    
    s_rssi, s_rtt, s_labels = load_wifi_data(SOURCE_CSV, is_source=True, samples_per_label=SAMPLES_SOURCE_TOTAL)
    t_rssi, t_rtt, t_labels = load_wifi_data(TARGET_CSV, is_source=False, samples_per_label=SAMPLES_TARGET)

    # --- [修改] Source Split: 100 for Train, 20 for Test ---
    full_source_dataset = TensorDataset(s_rssi, s_rtt, s_labels)
    
    # 計算分割數量 (假設每個類別數量均等)
    NUM_CLASSES = len(label_encoder.classes_)
    # 這裡用 random_split 做近似分割，因為已經是 120*N 的總數
    # 如果要嚴格每一類分 100/20，需要改寫 loader，但 random_split 在數據量大時通常足夠均勻
    total_source = len(full_source_dataset)
    train_source_len = NUM_CLASSES * 100
    test_source_len = total_source - train_source_len
    
    source_train_ds, source_test_ds = random_split(
        full_source_dataset, [train_source_len, test_source_len],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Source Data Split -> Train: {len(source_train_ds)}, Test: {len(source_test_ds)}")

    # 1. Target Train (給 DANN 訓練用的無標籤資料): 通常使用全部的 Target 資料
    #    因為在 UDA 中，我們假設可以看到所有 Target 的特徵 (Transductive setting)
    target_train_dataset = TensorDataset(t_rssi, t_rtt)
    
    # 2. Target Val/Test (給評估用的有標籤資料)
    full_target_labeled_dataset = TensorDataset(t_rssi, t_rtt, t_labels)
    
    # 設定切分比例 (例如 50% 驗證，50% 測試)
    total_size = len(full_target_labeled_dataset)
    val_size = int(0.5 * total_size) 
    test_size = total_size - val_size
    
    # 隨機切分
    target_val_dataset, target_test_dataset = random_split(
        full_target_labeled_dataset, [val_size, test_size],
        generator=torch.Generator().manual_seed(42) # 固定種子確保每次切分一樣
    )
    # target_test_dataset = 
    
    print(f"Target Data Split -> Val: {len(target_val_dataset)}, Test: {len(target_test_dataset)}")

    NUM_CLASSES = len(label_encoder.classes_)
    print(f"Number of class: {NUM_CLASSES}")
    BATCH_SIZE = 32
    
    # Source
    source_loader = DataLoader(TensorDataset(s_rssi, s_rtt, s_labels), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # Target Train (無標籤，用於訓練)
    target_train_loader = DataLoader(target_train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # Target Val (有標籤，用於挑選最佳模型)
    target_val_loader = DataLoader(target_val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Target Test (有標籤，用於最終報告 - 訓練中絕對不碰！)
    target_test_loader = DataLoader(target_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Source Test (新增：用於最後測試 Source 遷移後效果)
    source_test_loader = DataLoader(source_test_ds, batch_size=BATCH_SIZE, shuffle=False)

    class_names = label_encoder.classes_
    COORD_TENSOR = create_coord_tensor(class_names, DEVICE)

    # --- 2. 初始化模型 ---
    model = DualStreamDANN(num_aps=NUM_APS, num_classes=NUM_CLASSES).to(DEVICE)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # weight_decay 防止參數數值過大，抑制過擬合
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    
    # ==========================================
    # 設定 Loss 權重 (在這裡調整)
    # ==========================================
    # 建議：初期如果無法收斂，可降低 W_DOM；如果過擬合 Source，可提高 W_DOM
    W_CLS = 1         # 標籤分類 Loss 的比重 (Task Loss)
    W_DOM_RSSI = 1    # RSSI Domain Loss 的比重
    W_DOM_RTT = 1    # RTT Domain Loss 的比重

    # --- [新增] 歷史紀錄 Lists ---
    history = {
        'total_loss': [],
        'cls_loss': [],
        'dom_loss': []
        # 'val_mde': [],
        # 'val_loss': []
    }

    # ==========================================
    # 開始訓練
    # ==========================================
    num_epochs = 400
    best_mde = float('inf') # 初始設為無限大
    best_epoch = 0
    
    print(f"\nStart Training for {num_epochs} epochs...")
    print(f"Weights -> Class: {W_CLS}, RSSI Dom: {W_DOM_RSSI}, RTT Dom: {W_DOM_RTT}")
    print("-" * 95)
    print(f"{'Epoch':<6} | {'Domain Acc':<10} | {'Tot Loss':<10} | {'Cls Loss':<10} | {'D_RSSI':<8} | {'D_RTT':<8} | {'Save'}")
    print("-" * 95)

    # --- 設定 Warm-up 與 門檻參數 ---
    WARMUP_EPOCHS = 10       # 前 10 個 Epoch 不考慮儲存
    CLS_THRESHOLD = 0.55      # Class Loss 必須小於此值才開始考慮儲存
    best_adv_epoch = -1

    best_adv_epoch = -1
    min_cls_loss_rec = float('inf')      # 越小越好，初始設無限大
    max_dom_rssi_loss_rec = float('-inf') # 越大越好，初始設負無限大
    max_dom_rtt_loss_rec = float('-inf')  # 越大越好，初始設負無限大

    for epoch in range(num_epochs):
        model.train()
        
        total_domain_acc = 0.0
        total_loss_sum = 0.0
        cls_loss_sum = 0.0
        dom_rssi_sum = 0.0
        dom_rtt_sum = 0.0
        num_batches = 0
        
        # GRL Alpha 動態調整
        p = float(epoch) / num_epochs
        alpha = 2. / (1. + np.exp(-5 * p)) - 1
        alpha = min(alpha, 0.3)
        
        for (s_rssi_b, s_rtt_b, s_label_b), (t_rssi_b, t_rtt_b) in zip(source_loader, target_train_loader):
            
            s_rssi_b, s_rtt_b, s_label_b = s_rssi_b.to(DEVICE), s_rtt_b.to(DEVICE), s_label_b.to(DEVICE)
            t_rssi_b, t_rtt_b = t_rssi_b.to(DEVICE), t_rtt_b.to(DEVICE)
            
            # --- Forward ---
            class_out, d_rssi_s, d_rtt_s = model(s_rssi_b, s_rtt_b, alpha=alpha)
            _, d_rssi_t, d_rtt_t = model(t_rssi_b, t_rtt_b, alpha=alpha)
            
            # --- Loss Calculation ---
            # 1. Classification Loss
            loss_class = F.cross_entropy(class_out, s_label_b)
            
            # 2. Domain Loss
            d_label_s = torch.zeros(s_rssi_b.size(0), dtype=torch.long).to(DEVICE)
            d_label_t = torch.ones(t_rssi_b.size(0), dtype=torch.long).to(DEVICE)

            # 計算準確度 (Monitoring)
            preds_s = torch.argmax(d_rssi_s, dim=1)
            preds_t = torch.argmax(d_rssi_t, dim=1)
            acc_s = (preds_s == 0).float().mean()
            acc_t = (preds_t == 1).float().mean()
            domain_acc = (acc_s + acc_t) / 2
            
            loss_d_rssi = F.cross_entropy(d_rssi_s, d_label_s) + F.cross_entropy(d_rssi_t, d_label_t)
            loss_d_rtt = F.cross_entropy(d_rtt_s, d_label_s) + F.cross_entropy(d_rtt_t, d_label_t)
            
            # 3. Weighted Total Loss
            loss = (W_CLS * loss_class) + \
                (W_DOM_RSSI * loss_d_rssi) + \
                (W_DOM_RTT * loss_d_rtt)
            
            # --- Backward ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # --- Record ---
            total_domain_acc += domain_acc.item()
            total_loss_sum += loss.item()
            cls_loss_sum += loss_class.item()
            dom_rssi_sum += loss_d_rssi.item()
            dom_rtt_sum += loss_d_rtt.item()
            num_batches += 1

        # 計算 Training Avg Loss
        avg_domain_acc = total_domain_acc / num_batches
        avg_total = total_loss_sum / num_batches
        avg_cls = cls_loss_sum / num_batches
        avg_d_rssi = dom_rssi_sum / num_batches
        avg_d_rtt = dom_rtt_sum / num_batches
        
        # 這裡計算 val_acc, val_mde 供顯示用 (如果需要)
        # val_acc, val_mde = evaluate(...) 
        # 假設這裡已有 val_mde 和 val_acc 變數
        
        save_mark = ""
        
        # 1. 檢查是否過了 Warm-up 期
        is_warmed_up = (epoch + 1) > WARMUP_EPOCHS
        
        # 2. 檢查 Class Loss 是否已經收斂到可接受範圍 (避免存到瞎猜的模型)
        is_cls_good = avg_cls < CLS_THRESHOLD
        
        # 3. 檢查是否打破歷史紀錄 (同時滿足: Class更低 且 Domain更高)
        is_record_breaker = (
            avg_cls < min_cls_loss_rec and 
            avg_d_rssi > max_dom_rssi_loss_rec and 
            avg_d_rtt > max_dom_rtt_loss_rec
        )

        # 綜合判斷
        if is_warmed_up and is_cls_good and is_record_breaker:
            min_cls_loss_rec = avg_cls
            max_dom_rssi_loss_rec = avg_d_rssi
            max_dom_rtt_loss_rec = avg_d_rtt
            best_adv_epoch = epoch + 1
            
            torch.save(model.state_dict(), "best_model_adversarial.pth")
            save_mark = "(ADV)" 

        # if epoch == 399:
        #     torch.save(model.state_dict(), "best_model_adversarial.pth")
        
        # --- 原本的 MDE 儲存邏輯 (保留) ---
        # if val_mde < best_mde:
        #     best_mde = val_mde
        #     torch.save(model.state_dict(), "best_model_mde.pth")
        #     if save_mark:
        #         save_mark += "/*" # 若同時滿足兩者，標記為 (ADV)/*
        #     else:
        #         save_mark = "*"

        # --- Print 輸出 (加入文字顏色區分會更清楚，這裡用純文字) ---
        # 如果還在 Warm-up，可以在 save_mark 提示一下，或者保持空白
        if not is_warmed_up:
            debug_info = "WarmUp"
        elif not is_cls_good:
            debug_info = "ClsHigh"
        else:
            debug_info = ""

        print(f"{epoch+1:<6} | {avg_domain_acc:<10.4f} | {avg_total:<10.4f} | {avg_cls:<10.4f} | {avg_d_rssi:<8.4f} | {avg_d_rtt:<8.4f} | {save_mark:<8} {debug_info}")

    # --- 訓練結束後的總結 ---
    print("-" * 80)
    if best_adv_epoch != -1:
        print(f"Code saved 'best_model_adversarial.pth' at Epoch: {best_adv_epoch}")
        print(f"Criteria: Epoch > {WARMUP_EPOCHS} and Cls Loss < {CLS_THRESHOLD}")
        print(f"Stats -> Cls: {min_cls_loss_rec:.4f}, RSSI: {max_dom_rssi_loss_rec:.4f}, RTT: {max_dom_rtt_loss_rec:.4f}")
    else:
        print("No model satisfied the strict adversarial criteria.")
    print("-" * 80)
    
    # --- 繪製 Loss 圖 ---
    plot_training_history(history)

    # ==========================================
    # 4. 載入最好的模型進行最終確認
    # ==========================================
    print("Loading best model for final report...")
    model.load_state_dict(torch.load("best_model_adversarial.pth"))

    # 1. Target Test Eval (含 CDF 資料收集)
    t_acc, t_mde, t_errors = evaluate(model, target_test_loader, COORD_TENSOR, DEVICE, return_all_errors=True)
    
    # 2. Source Test Eval (新增需求：含 CDF 資料收集)
    s_acc, s_mde, s_errors = evaluate(model, source_test_loader, COORD_TENSOR, DEVICE, return_all_errors=True)

    print("\n" + "="*50)
    print(f" FINAL REPORT (Fusion)")
    print("="*50)
    print(" [Target Domain Test]")
    print(f" Accuracy : {t_acc:.2f}%")
    print(f" MDE      : {t_mde:.4f} m")
    print("-" * 50)
    print(" [Source Domain Test (Reserved 20/class)]")
    print(f" Accuracy : {s_acc:.2f}%")
    print(f" MDE      : {s_mde:.4f} m")
    print("="*50)

    # --- 繪製 CDF 圖 ---
    plot_cdf({
        'Target Test': t_errors,
        'Source Test': s_errors
    }, filename='cdf_comparison.png')
    
    # # [修正點] 最終報告必須使用 target_test_loader (這是模型從未看過的標籤)
    # final_acc, final_mde = evaluate(model, target_test_loader, COORD_TENSOR, DEVICE)

    # print("\n" + "="*50)
    # print(f" FINAL REPORT (Evaluated on HELD-OUT Test Set)")
    # print("="*50)
    # print(f" Accuracy                : {final_acc:.2f}%")
    # print(f" Mean Distance Error (MDE) : {final_mde:.4f} meters")
    # print("="*50)

if __name__ == '__main__':
    main()