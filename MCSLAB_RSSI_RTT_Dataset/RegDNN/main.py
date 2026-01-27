import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import json
import os
import copy
# from data_utils_cvae import get_data_loaders
from data_utils import get_data_loaders

# === Path Variable ===
CSV_PATH = '../2026_1_1/all/Server_Wide_20260101_140347.csv'
# CSV_PATH = '../2026_1_1/merged_wifi_dataset.csv'
# CSV_PATH = './debug_fusion_input_data.csv'
# CSV_PATH = '../2026_1_1/merged_output_ap1_replaced.csv' # Data Fusion Exp 3
# CSV_PATH = '../2026_1_1/merged_output_ALL_APs_replaced.csv'
# CSV_PATH = '../2025_1_3/mcslab_2025_1_3.csv'

BATCH_SIZE = 32
LEARNING_RATE = 0.001
MAX_EPOCHS = 1000
PATIENCE = 15      # Early Stopping 等待次數
NUM_RUNS = 10      # 實驗執行次數
LOG_INTERVAL = 10  # 設定每 10 Epoch 印出一次

# === [修改重點] ===
# 設定每個 RP 要使用多少筆資料 (設為 None 代表用全部)
SAMPLES_PER_RP = 50 

# 座標表 (維持不變)
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
        self.best_model_wts = copy.deepcopy(model.state_dict()) # deep copy 用來建立一份完全獨立的變數 (DNN模型權重)

# DNN 模型
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
            # Output Layer
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# 計算距離平方差 (meter)
def calculate_distance(label_str_1, label_str_2):
    p1 = LABEL_TO_COORDS.get(label_str_1, (0, 0))
    p2 = LABEL_TO_COORDS.get(label_str_2, (0, 0))
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def run_single_experiment(run_id):

    print(f"\n--- Run {run_id} Data Loading (Samples per RP: {SAMPLES_PER_RP}) ---")
    
    data = get_data_loaders(
        csv_path=CSV_PATH, 
        batch_size=BATCH_SIZE, 
        random_state=42 + run_id*100, 
        loop_id=run_id,
        samples_per_label=SAMPLES_PER_RP
    )
    
    if data is None: return None
    train_loader, val_loader, test_loader, meta = data

    # === [新增] 檢查點: 印出最終訓練用的特徵名稱 ===
    print(f"--- [Check] Feature Columns ({len(meta['feature_names'])}) ---")
    print(meta['feature_names'])
    print("--------------------------------------------------")
    # ================================================
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNN(meta['input_dim'], meta['num_classes']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    early_stopping = EarlyStopping(patience=PATIENCE)
    
    print(f"--- Run {run_id} Training Start ---")
    
    # 2. 訓練 Loop (與之前相同)
    for epoch in range(MAX_EPOCHS):
        # === Training Phase ===
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
            
        # === Validation Phase ===
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
        
        # Check Early Stopping
        early_stopping(avg_val_loss, model)
        
        # 每 10 Epoch 印出一次詳細資訊
        if (epoch + 1) % LOG_INTERVAL == 0:
            print(f"Epoch {epoch+1:04d} | "
                  f"Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.2f}% | "
                  f"Val Loss: {avg_val_loss:.4f} Acc: {avg_val_acc:.2f}%")

        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # 3. 測試與報表生成
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