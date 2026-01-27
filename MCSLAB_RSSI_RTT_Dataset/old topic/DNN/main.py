import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import json
import os
import copy
from data_utils_v1 import get_data_loaders

# === 設定區 (Configuration) ===
CSV_PATH = '../2025_12_30/Server_Wide_20251230_084158.csv'

# 特徵設定
FEATURE_CONFIG = ["RSSI", "Dist_mm", "Std_mm"] 
# FEATURE_CONFIG = ["RSSI"] 
# FEATURE_CONFIG = ["Dist_mm"] 
# FEATURE_CONFIG = ["RSSI", "Dist_mm"] 
# FEATURE_CONFIG = ["Std_mm"] 

# 新增 WINDOW_SIZE 設定
# row 106 modified
# WINDOW_SIZE = 10  # 使用過去 10 筆數據 (約 2 秒) 來計算 IQR

BATCH_SIZE = 32
LEARNING_RATE = 0.001
MAX_EPOCHS = 1000        
PATIENCE = 20            
NUM_RUNS = 10             
LOG_INTERVAL = 10        # 設定每幾 Epoch 印出一次 Log

# 座標對照表
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

# === Early Stopping Class ===
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
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        self.best_model_wts = copy.deepcopy(model.state_dict())

# === Model Definition ===
class IndoorDNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(IndoorDNN, self).__init__()
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
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def calculate_distance(label_str_1, label_str_2):
    p1 = LABEL_TO_COORDS.get(label_str_1, (0, 0))
    p2 = LABEL_TO_COORDS.get(label_str_2, (0, 0))
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# === Single Experiment Runner ===
def run_single_experiment(run_id):
    # 1. 取得資料
    data = get_data_loaders(CSV_PATH, FEATURE_CONFIG, BATCH_SIZE, random_state=42 + run_id*100)
    # data = get_data_loaders(CSV_PATH, FEATURE_CONFIG, BATCH_SIZE, 
    #                         window_size=WINDOW_SIZE,
    #                         random_state=42 + run_id*100)
    
    if data is None: return None
    train_loader, val_loader, test_loader, meta = data
    
    input_dim = meta['input_dim']
    num_classes = meta['num_classes']
    label_encoder = meta['label_encoder']

    # 2. 初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IndoorDNN(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 初始化 Early Stopping
    model_save_name = f"model_run_{run_id}.pt"
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=False, path=model_save_name)

    print(f"\n--- Run {run_id} Start (Features: {FEATURE_CONFIG}) ---")

    # 3. 訓練 Loop (含 Validation)
    for epoch in range(MAX_EPOCHS):
        # --- Train Phase ---
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
            
            # 統計 Train Metrics
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = 100 * train_correct / train_total
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # 統計 Val Metrics
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = 100 * val_correct / val_total
        
        # Check Early Stopping
        early_stopping(avg_val_loss, model)
        
        # Print Log
        if (epoch+1) % LOG_INTERVAL == 0:
             print(f"Epoch {epoch+1:04d} | Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.2f}% | Val Loss: {avg_val_loss:.4f} Acc: {avg_val_acc:.2f}%")

        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # 4. 載入最佳模型權重進行測試
    model.load_state_dict(early_stopping.best_model_wts)
    torch.save(model.state_dict(), model_save_name)

    # 5. 測試與生成詳細報表
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
            
            pred_strs = label_encoder.inverse_transform(predicted.cpu().numpy())
            true_strs = label_encoder.inverse_transform(labels.cpu().numpy())
            
            for p_str, t_str in zip(pred_strs, true_strs):
                dist = calculate_distance(p_str, t_str)
                if t_str not in mde_report_per_label:
                    mde_report_per_label[t_str] = []
                mde_report_per_label[t_str].append(float(dist))

    # 計算統計數據
    final_report = {}
    all_distances = []
    
    for label, dists in mde_report_per_label.items():
        mean_d = np.mean(dists)
        all_distances.extend(dists)
        errors_detail = [d for d in dists if d > 0]
        final_report[label] = {
            "mde": round(mean_d, 4),
            "count": len(dists),
            "errors": errors_detail 
        }

    overall_acc = 100 * correct / total
    overall_mde = np.mean(all_distances)
    
    print(f"Run {run_id} Result -> Test Acc: {overall_acc:.2f}%, Test MDE: {overall_mde:.4f}m")
    
    json_filename = f"report_run_{run_id}.json"
    with open(json_filename, "w") as f:
        json.dump(final_report, f, indent=4)
        
    return overall_acc, overall_mde

# === 主程式 ===
def main():
    print(f"開始執行實驗，特徵組合: {FEATURE_CONFIG}")
    acc_list = []
    mde_list = []
    
    for i in range(NUM_RUNS):
        res = run_single_experiment(i)
        if res:
            acc_list.append(res[0])
            mde_list.append(res[1])
            
    print("\n=== 最終平均結果 ===")
    print(f"Feature Used: {FEATURE_CONFIG}")
    print(f"Avg Accuracy: {np.mean(acc_list):.2f}% (+/- {np.std(acc_list):.2f})")
    print(f"Avg MDE:      {np.mean(mde_list):.4f}m (+/- {np.std(mde_list):.4f})")

if __name__ == "__main__":
    main()