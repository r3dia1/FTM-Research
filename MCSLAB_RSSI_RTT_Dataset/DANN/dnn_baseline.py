import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch.nn.functional as F
import os
import random

# ==========================================
# 1. 模型架構：標準 DNN (無遷移學習)
# ==========================================
class BaselineDNN(nn.Module):
    def __init__(self, input_dim=4, num_classes=5, hidden_dim=64):
        super(BaselineDNN, self).__init__()

        # --- 特徵提取器 ---
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 32),
            # nn.Linear(5, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True)
        )

        # --- 標籤分類器 ---
        self.class_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        class_output = self.class_classifier(features)
        return class_output

# ==========================================
# 資料處理與設定
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
    # rtt_cols = ['Dist_mm_1', 'Dist_mm_2', 'Dist_mm_4']
    rtt_cols = ['Dist_mm_3', 'Dist_mm_4']
    # rtt_cols = ['Dist_mm_1']
    
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
            raise ValueError("Error: Scaler not fitted.")
        rssi_data = rssi_scaler.transform(rssi_data)
        rtt_data = rtt_scaler.transform(rtt_data)
        try:
            labels = label_encoder.transform(raw_labels)
        except:
            labels = np.zeros(len(df))

    return torch.tensor(rssi_data), torch.tensor(rtt_data), torch.tensor(labels, dtype=torch.long)

# ==========================================
# 座標與評估函式
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
        for x_b, labels_b in data_loader:
            x_b, labels_b = x_b.to(device), labels_b.to(device)
            class_out = model(x_b)
            
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
    USE_MODE = 'rssi' 
    seed_candidate = [42, 6767, 123456]
    source_acc = []
    source_mde = []
    target_acc = []
    target_mde = []

    for seed in seed_candidate:
        set_seed(seed)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(DEVICE)
        
        # INPUT_DIM = 4
        # INPUT_DIM = 6
        INPUT_DIM = 4

        SOURCE_CSV = '../2026_1_1/all/All_Data_With_RSSI_Diff.csv'  
        # TARGET_CSV = '../2026_1_2/All_Data_With_RSSI_Diff.csv'
        TARGET_CSV = '/home/mcslab/yutung/MCSLAB_RSSI_RTT_Dataset/2026_1_14/All_Data_With_RSSI_Diff.csv'

        print(f"Using device: {DEVICE}")
        print(f"Running Baseline DNN (Strict Evaluation Mode) with mode: {USE_MODE}")

        # --- 1. 讀取數據 ---
        SAMPLES_PER_CLASS = 100 
        s_rssi, s_rtt, s_labels = load_wifi_data(SOURCE_CSV, is_source=True, samples_per_label=SAMPLES_PER_CLASS)
        t_rssi, t_rtt, t_labels = load_wifi_data(TARGET_CSV, is_source=False, samples_per_label=SAMPLES_PER_CLASS)

        if USE_MODE == 'rtt':
            s_data, t_data = s_rtt, t_rtt
        elif USE_MODE == 'fusion':
            s_data = torch.cat((s_rssi, s_rtt), dim=1)
            t_data = torch.cat((t_rssi, t_rtt), dim=1)
        else:
            s_data, t_data = s_rssi, t_rssi

        # ==========================================
        # 資料切分 (嚴格隔離)
        # ==========================================
        # 1. Source Data (用於 Train & Validation)
        full_source_dataset = TensorDataset(s_data, s_labels)
        train_size = int(0.8 * len(full_source_dataset)) # 80% 用於訓練
        val_size = len(full_source_dataset) - train_size # 20% 用於驗證
        
        source_train_ds, source_val_ds = random_split(
            full_source_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # 2. Target Data (全部封存為 Test，訓練時完全不碰)
        target_test_dataset = TensorDataset(t_data, t_labels)
        
        print(f"Data Split:")
        print(f" - Source Train : {len(source_train_ds)}")
        print(f" - Source Val   : {len(source_val_ds)} (Model Selection Criteria)")
        print(f" - Target Test  : {len(target_test_dataset)} (Hidden until final report)")

        NUM_CLASSES = len(label_encoder.classes_)
        BATCH_SIZE = 32
        
        train_loader = DataLoader(source_train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader   = DataLoader(source_val_ds, batch_size=BATCH_SIZE, shuffle=False)
        # test_loader 先準備好，但訓練迴圈中不使用
        test_loader  = DataLoader(target_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        class_names = label_encoder.classes_
        COORD_TENSOR = create_coord_tensor(class_names, DEVICE)

        # --- 2. 初始化模型 ---
        model = BaselineDNN(input_dim=INPUT_DIM, num_classes=NUM_CLASSES).to(DEVICE)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        # ==========================================
        # 開始訓練
        # ==========================================
        num_epochs = 400
        best_val_acc = 0.0 
        best_val_mde = 999
        best_epoch = 0
        
        print(f"\nStart Training...")
        print("-" * 65)
        # 注意：這裡不顯示 Target Test 的結果
        print(f"{'Epoch':<6} | {'Train Loss':<10} || {'S-Val Acc':<10} | {'S-Val MDE':<10} | {'Save'}")
        print("-" * 65)

        for epoch in range(num_epochs):
            model.train()
            train_loss_sum = 0.0
            num_batches = 0
            
            for x_b, label_b in train_loader:
                x_b, label_b = x_b.to(DEVICE), label_b.to(DEVICE)
                
                output = model(x_b)
                loss = criterion(output, label_b)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss_sum += loss.item()
                num_batches += 1
                
            avg_train_loss = train_loss_sum / num_batches
            
            # --- Validation (只看 Source Validation) ---
            # 這是我們唯一可以用來判斷模型好壞的依據
            val_acc, val_mde = evaluate(model, val_loader, COORD_TENSOR, DEVICE)
            
            save_mark = ""
            # 儲存策略：當 Source Val Accuracy 創新高時儲存
            if val_acc > best_val_acc: 
                best_val_acc = val_acc
                best_val_mde = val_mde
                best_epoch = epoch + 1
                torch.save(model.state_dict(), "baseline_dnn_strict.pth")
                save_mark = "*"
            # if val_mde < best_val_mde: 
            #     best_val_acc = val_acc
            #     best_val_mde = val_mde
            #     best_epoch = epoch + 1
            #     torch.save(model.state_dict(), "baseline_dnn_strict.pth")
            #     save_mark = "*"
            
            print(f"{epoch+1:<6} | {avg_train_loss:<10.4f} || {val_acc:<10.2f} | {val_mde:<10.4f} | {save_mark}")

        print("-" * 65)

        source_acc.append(best_val_acc)
        source_mde.append(best_val_mde)
        print(f"Training Finished. Best Source Val Acc: {best_val_acc:.2f}%, Val MDE: {best_val_mde:.2f} at Epoch {best_epoch}")
        
        # ==========================================
        # 4. 最終測試 (揭曉時刻)
        # ==========================================
        print("\nLoading best model (selected purely by Source Validation)...")
        model.load_state_dict(torch.load("baseline_dnn_strict.pth"))
        
        # 這是 Target Test Set 第一次被輸入模型
        final_acc, final_mde = evaluate(model, test_loader, COORD_TENSOR, DEVICE)
        target_acc.append(final_acc)
        target_mde.append(final_mde)

        print("\n" + "="*50)
        print(f" FINAL REPORT (Blind Test Result)")
        print("="*50)
        print(f" Test Set                : Target Domain Data")
        print(f" Accuracy                : {final_acc:.2f}%")
        print(f" Mean Distance Error (MDE) : {final_mde:.4f} meters")
        print("="*50)
    
    avg_source_acc = np.mean(source_acc)
    avg_source_mde = np.mean(source_mde)
    avg_target_acc = np.mean(target_acc)
    avg_target_mde = np.mean(target_mde)
    print(f"source acc: {avg_source_acc:.4f}, source mde: {avg_source_mde:.4f}, target acc: {avg_target_acc:.4f}, target mde: {avg_target_mde:.4f}")

if __name__ == '__main__':
    main()