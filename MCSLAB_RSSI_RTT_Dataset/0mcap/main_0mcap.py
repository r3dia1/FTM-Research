import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import json
import os
import copy
from data_utils_cvae_0mcap import get_data_loaders  # 確保這引用的是剛修改過的 data_utils.py

# === 1. 路徑與資料設定區 ===
# Full Set (Student Data)
CSV_PATH = '../2026_1_1/all/Server_Wide_20260101_140347.csv'

# Teacher Data (用來訓練 CVAE 的 Single Sets)
# 請確認這些路徑是正確的 (根據你之前的描述設定)
TEACHER_POS3 = '../2026_1_1/Server_Wide_20260101_065323_location_AP3.csv'
TEACHER_POS2 = '../2026_1_1/Server_Wide_20260101_124356_location_AP2.csv'
TEACHER_POS1 = '../2026_1_1/Server_Wide_20260101_075453_location_AP1.csv'
TEACHER_POS4 = '../2026_1_1/Server_Wide_20260101_113813_location_AP4.csv'

# 設定每個 RP 要用多少筆資料 (設為 None 代表用全部 500 筆)
SAMPLES_PER_RP = 50

# === 2. CVAE 配置與特徵選擇 (核心修改) ===
# 定義每個預測目標對應的訓練檔案與來源 RSSI
CVAE_CONFIG = {
    'Dist_Pred_1': {'train_file': TEACHER_POS1, 'rssi_col': 'RSSI_1'},
    'Dist_Pred_2': {'train_file': TEACHER_POS2, 'rssi_col': 'RSSI_2'},
    'Dist_Pred_3': {'train_file': TEACHER_POS3, 'rssi_col': 'RSSI_3'},
    'Dist_Pred_4': {'train_file': TEACHER_POS4, 'rssi_col': 'RSSI_4'}
}

# 定義這一次 DNN 實驗要使用的輸入特徵
SELECTED_FEATURES = [
    # 'Dist_mm_3',   # AP1 可信，直接用
    # 'Std_mm_3',    
    'RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4', # 環境特徵
    'Dist_Pred_3',
    'Dist_Pred_2', # CVAE (Trained on Pos2) 生成的距離
    'Dist_Pred_1', # CVAE (Trained on Pos3) 生成的距離
    'Dist_Pred_4'  # CVAE (Trained on Pos4) 生成的距離
]

# === 3. 訓練超參數 ===
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MAX_EPOCHS = 1000
PATIENCE = 15      
NUM_RUNS = 10      
LOG_INTERVAL = 10  

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

# DNN 模型
class RegDNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(RegDNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def calculate_distance(label_str_1, label_str_2):
    p1 = LABEL_TO_COORDS.get(label_str_1, (0, 0))
    p2 = LABEL_TO_COORDS.get(label_str_2, (0, 0))
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def run_single_experiment(run_id):
    # === 4. 修改：呼叫 get_data_loaders 並傳入 Config ===
    # 注意：新的介面不接受 loop_id，我們用 random_state 來控制隨機性
    current_random_state = 42 + run_id * 100
    
    print(f"\n[Run {run_id}] Loading Data & CVAE Pipelines...")
    
    data = get_data_loaders(
        main_csv_path=CSV_PATH, 
        selected_features=SELECTED_FEATURES, 
        cvae_config=CVAE_CONFIG,             
        batch_size=BATCH_SIZE, 
        random_state=current_random_state,
        samples_per_label=SAMPLES_PER_RP  # <--- 這裡
    )
    
    if data is None: 
        print("Error: Data loading failed.")
        return None
        
    train_loader, val_loader, test_loader, meta = data
    
    # 接下來的訓練流程與原本相同
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegDNN(meta['input_dim'], meta['num_classes']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    early_stopping = EarlyStopping(patience=PATIENCE)
    
    print(f"--- Run {run_id} Training Start (Features: {meta['feature_names']}) ---")
    
    for epoch in range(MAX_EPOCHS):
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
        
        early_stopping(avg_val_loss, model)
        
        if (epoch + 1) % LOG_INTERVAL == 0:
            print(f"Epoch {epoch+1:04d} | "
                  f"Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.2f}% | "
                  f"Val Loss: {avg_val_loss:.4f} Acc: {avg_val_acc:.2f}%")

        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

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
    with open(f"Testing_mde_run_{run_id}.json", "w") as f:
        json.dump(mde_report_avg, f, indent=4)
        
    mde_detailed = {}
    for l, d in mde_report_per_label.items():
        error_dict = {str(i+1): v for i, v in enumerate(d) if v > 0}
        entry = {"mde": np.mean(d), "count": len(d)}
        if error_dict: entry["error"] = error_dict
        mde_detailed[l] = entry
    with open(f"Testing_mde_detailed_run_{run_id}.json", "w") as f:
        json.dump(mde_detailed, f, indent=4)

    acc = 100 * correct / total
    mean_mde = np.mean([np.mean(d) for d in mde_report_per_label.values()])
    print(f"Run {run_id} Result -> Test Acc: {acc:.2f}%, MDE: {mean_mde:.4f}m")
    
    return acc, mean_mde

def main():
    acc_list = []
    mde_list = []
    for i in range(NUM_RUNS):
        res = run_single_experiment(i)
        if res:
            acc_list.append(res[0])
            mde_list.append(res[1])
            
    print("\n=== Final Results ===")
    if acc_list:
        print(f"Avg Accuracy: {np.mean(acc_list):.2f}% (+/- {np.std(acc_list):.2f})")
        print(f"Avg MDE:      {np.mean(mde_list):.4f}m (+/- {np.std(mde_list):.4f})")
    else:
        print("No successful runs.")

if __name__ == "__main__":
    main()