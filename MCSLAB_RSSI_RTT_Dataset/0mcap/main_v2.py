import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import math
import json
import os
import copy
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# 假設這是您的 data_utils，我們稍後會用自定義的 loader 邏輯來適應動態特徵
# from data_utils import get_data_loaders 

# ==========================================
# 1. 全域設定與特徵選擇
# ==========================================
# 原始 Student 資料
RAW_STUDENT_CSV = '../2026_1_1/all/Server_Wide_20260101_140347.csv'
# 生成的中繼資料 (包含 RSSI + Pseudo RTT)
FUSION_CSV_PATH = 'debug_fusion_input_data.csv'

# Teacher 資料路徑 (用於 Stage 1 生成 RTT)
TEACHER_FILES = {
    1: '../2026_1_1/Server_Wide_20260101_075453_location_AP1.csv',
    2: '../2026_1_1/Server_Wide_20260101_124356_location_AP2.csv',
    3: '../2026_1_1/Server_Wide_20260101_065323_location_AP3.csv',
    4: '../2026_1_1/Server_Wide_20260101_113813_location_AP4.csv'
}

# [關鍵] 定義最後要丟入 DNN 的特徵
SELECTED_FEATURES = [
    'RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4',
    # 'Pseudo_RTT_1', 'Pseudo_RTT_2', 'Pseudo_RTT_3', 'Pseudo_RTT_4'
]

# 訓練參數
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MAX_EPOCHS = 1000
PATIENCE = 15
NUM_RUNS = 10
LOG_INTERVAL = 10
SAMPLES_PER_RP = 50  # 您的設定

# 座標表
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

# ==========================================
# 2. 工具類別 (EarlyStopping, RegDNN)
# ==========================================
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_wts = None
        self.path = path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        self.best_model_wts = copy.deepcopy(model.state_dict())

class DNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(DNN, self).__init__()
        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            # Layer 2
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            # Layer 3
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            # Output Layer (Classification)
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def calculate_distance(label_str_1, label_str_2):
    p1 = LABEL_TO_COORDS.get(label_str_1, (0, 0))
    p2 = LABEL_TO_COORDS.get(label_str_2, (0, 0))
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# ==========================================
# 3. Stage 1: 生成 Pseudo RTT (簡化版)
# ==========================================
# 這裡為了節省篇幅，我假設您已經跑過之前的程式生成了 CSV
# 如果需要，我可以把 train_teacher_and_generate_pseudo 函式放回來
# 這裡我們假設 FUSION_CSV_PATH 已經準備好 (或是第一次執行會自動檢查)

def ensure_fusion_data_exists(le):
    if os.path.exists(FUSION_CSV_PATH):
        print(f"[Info] Found existing fusion data: {FUSION_CSV_PATH}")
        return
    
    print("[Error] Fusion data not found! Please run the 'Stage 1' script first to generate Pseudo RTTs.")
    # 這裡應該呼叫之前寫好的 train_teacher_and_generate_pseudo(df_student, le)
    # 為了保持程式碼結構清晰，請確保先執行過之前的生成步驟
    raise FileNotFoundError("Run Stage 1 generation first.")

# ==========================================
# 4. Stage 2: 您的實驗迴圈 (Run Experiment)
# ==========================================
def get_fusion_data_loaders(csv_path, selected_features, batch_size, random_state, samples_per_label):
    """
    客製化的 DataLoader，專門讀取包含 Pseudo RTT 的 CSV，並選取特定特徵
    """
    df = pd.read_csv(csv_path)
    
    # Label Encoder
    labels = sorted(list(LABEL_TO_COORDS.keys()))
    le = LabelEncoder()
    le.fit(labels)
    
    # Filter valid labels
    df = df[df['Label'].isin(le.classes_)].copy()
    
    # Sampling per Label (Balance Data)
    if samples_per_label is not None:
        df = df.groupby('Label').sample(n=samples_per_label, random_state=random_state, replace=True)
    
    # Feature Selection
    # 
    missing = [c for c in selected_features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
        
    X = df[selected_features].values
    y = le.transform(df['Label'].values)
    
    # Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train
    )
    
    train_ds = SimpleDataset(X_train, y_train)
    val_ds = SimpleDataset(X_val, y_val)
    test_ds = SimpleDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    meta = {
        'input_dim': len(selected_features),
        'num_classes': len(le.classes_),
        'feature_names': selected_features,
        'label_encoder': le
    }
    
    return train_loader, val_loader, test_loader, meta

def run_single_experiment(run_id):
    # === [修改] 使用新的 loader 函式 ===
    print(f"\n--- Run {run_id} Data Loading (Features: {len(SELECTED_FEATURES)}) ---")
    
    data = get_fusion_data_loaders(
        csv_path=FUSION_CSV_PATH,  # 讀取生成好的資料
        selected_features=SELECTED_FEATURES,
        batch_size=BATCH_SIZE,
        random_state=42 + run_id*100,
        samples_per_label=SAMPLES_PER_RP
    )
    
    train_loader, val_loader, test_loader, meta = data

    print(f"--- [Check] Feature Columns: {meta['feature_names']} ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # [修改] 使用您的 RegDNN
    model = DNN(meta['input_dim'], meta['num_classes']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    early_stopping = EarlyStopping(patience=PATIENCE)
    
    print(f"--- Run {run_id} Training Start ---")
    
    # 訓練迴圈 (完全保留您的邏輯)
    for epoch in range(MAX_EPOCHS):
        # === Training ===
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = 100 * train_correct / train_total
            
        # === Validation ===
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = 100 * val_correct / val_total
        
        # Early Stopping Check
        early_stopping(avg_val_loss, model)
        
        if (epoch + 1) % LOG_INTERVAL == 0:
            print(f"Epoch {epoch+1:04d} | Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.2f}% | Val Loss: {avg_val_loss:.4f} Acc: {avg_val_acc:.2f}%")

        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # === Testing & Reporting (保留您的 JSON 輸出格式) ===
    model.load_state_dict(early_stopping.best_model_wts)
    model.eval()
    
    correct = 0
    total = 0
    mde_report_per_label = {}
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pred_strs = meta['label_encoder'].inverse_transform(predicted.cpu().numpy())
            true_strs = meta['label_encoder'].inverse_transform(labels.cpu().numpy())
            
            for p_str, t_str in zip(pred_strs, true_strs):
                dist = calculate_distance(p_str, t_str)
                if t_str not in mde_report_per_label:
                    mde_report_per_label[t_str] = []
                mde_report_per_label[t_str].append(float(dist))

    # JSON Reports
    mde_report_avg = {l: {"mde": np.mean(d), "count": len(d)} for l, d in mde_report_per_label.items()}
    with open(f"Fusion_Result_Run_{run_id}.json", "w") as f:
        json.dump(mde_report_avg, f, indent=4)
        
    mde_detailed = {}
    for l, d in mde_report_per_label.items():
        error_dict = {str(i+1): v for i, v in enumerate(d) if v > 0}
        entry = {"mde": np.mean(d), "count": len(d)}
        if error_dict: entry["error"] = error_dict
        mde_detailed[l] = entry
    with open(f"Fusion_Detailed_Run_{run_id}.json", "w") as f:
        json.dump(mde_detailed, f, indent=4)

    acc = 100 * correct / total
    mean_mde = np.mean([np.mean(d) for d in mde_report_per_label.values()])
    print(f"Run {run_id} Result -> Test Acc: {acc:.2f}%, MDE: {mean_mde:.4f}m")
    
    return acc, mean_mde

def main():
    # 確保 Stage 1 已經跑過，且 CSV 存在
    # (我們使用簡單的 Label Encoder 來檢查檔案，不進行完整訓練)
    labels = sorted(list(LABEL_TO_COORDS.keys()))
    le = LabelEncoder()
    le.fit(labels)
    ensure_fusion_data_exists(le)
    
    acc_list = []
    mde_list = []
    for i in range(NUM_RUNS):
        res = run_single_experiment(i)
        if res:
            acc_list.append(res[0])
            mde_list.append(res[1])
            
    print("\n=== Final Results (Fusion) ===")
    if acc_list:
        print(f"Avg Accuracy: {np.mean(acc_list):.2f}% (+/- {np.std(acc_list):.2f})")
        print(f"Avg MDE:      {np.mean(mde_list):.4f}m (+/- {np.std(mde_list):.4f})")
    else:
        print("No successful runs.")

if __name__ == "__main__":
    main()