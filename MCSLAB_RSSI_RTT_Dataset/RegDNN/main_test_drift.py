import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import json
import os
import copy
from data_utils_test_drift import get_data_loaders, get_new_data_loader # 引用更新後的函式

# === 設定區 ===
# TRAIN_CSV_PATH = '../2026_1_1/all/Server_Wide_20260101_140347.csv'
# NEW_TEST_CSV_PATH = '../2026_1_2/Server_Wide_20260102_114230.csv'
# NEW_TEST_CSV_PATH = '../2026_1_1/all/Server_Wide_20260101_140347.csv'
# TRAIN_CSV_PATH = '../2026_1_2/Server_Wide_20260102_114230.csv'
TRAIN_CSV_PATH = '../2025_1_3/mcslab_2025_1_3.csv'
NEW_TEST_CSV_PATH = '../2026_1_1/all/Server_Wide_20260101_140347.csv'

BATCH_SIZE = 32
LEARNING_RATE = 0.001
MAX_EPOCHS = 1000
PATIENCE = 15
LOG_INTERVAL = 10
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
        self.best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), self.path)

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

def test_model(model, loader, meta, device, desc="Test"):
    """通用的測試與 MDE 計算函式"""
    model.eval()
    correct = 0
    total = 0
    mde_list = []
    encoder = meta['label_encoder']
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pred_strs = encoder.inverse_transform(predicted.cpu().numpy())
            true_strs = encoder.inverse_transform(labels.cpu().numpy())
            
            for p_str, t_str in zip(pred_strs, true_strs):
                dist = calculate_distance(p_str, t_str)
                mde_list.append(dist)

    acc = 100 * correct / total
    avg_mde = np.mean(mde_list) if mde_list else 0
    print(f"[{desc}] Acc: {acc:.2f}%, MDE: {avg_mde:.4f}m")
    return acc, avg_mde

def run_single_experiment(run_id):
    model_save_path = f"best_model_run_{run_id}.pt"
    print(f"\n--- Run {run_id} Data Loading ---")
    
    # 1. 取得資料 (包含 Train/Val/Test Split)
    data = get_data_loaders(
        csv_path=TRAIN_CSV_PATH, 
        batch_size=BATCH_SIZE, 
        random_state=42 + run_id*100, 
        loop_id=run_id,
        samples_per_label=SAMPLES_PER_RP 
    )
    
    if data is None: return None
    train_loader, val_loader, test_loader, meta = data
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegDNN(meta['input_dim'], meta['num_classes']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    early_stopping = EarlyStopping(patience=PATIENCE, path=model_save_path)
    
    print(f"--- Run {run_id} Training Start ---")
    
    # 2. 訓練 Loop
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
        
        avg_train_acc = 100 * train_correct / train_total

        # Validation
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
            print(f"Epoch {epoch+1:04d} | Train: {avg_train_acc:.2f}% | Val: {avg_val_acc:.2f}%")

        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # 3. 測試原本的 Test Split (Requirement 1)
    print(f"\n--- Run {run_id} Evaluation (Original Split) ---")
    # 載入最佳模型權重
    model.load_state_dict(torch.load(model_save_path))
    test_acc, test_mde = test_model(model, test_loader, meta, device, desc="Original Test Split")

    # 回傳結果與 meta
    return test_acc, test_mde, model_save_path, meta 

def test_on_new_csv(model_path, new_csv_path, meta):
    """
    對全新 CSV 進行獨立測試
    """
    print(f"\n>>> Start Independent Testing on: {new_csv_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 準備資料 (使用 get_new_data_loader 確保轉換一致性)
    # 這裡直接傳入 meta，不需要再傳 samples_per_label，因為測試通常用全量
    new_loader = get_new_data_loader(new_csv_path, meta, BATCH_SIZE)
    
    if new_loader is None:
        print("Error: 無法建立 DataLoader (可能是檔案不存在或 Label 不符)")
        return

    # 2. 重建模型並載入權重
    model = RegDNN(meta['input_dim'], meta['num_classes']).to(device)
    model.load_state_dict(torch.load(model_path))
    
    # 3. 執行測試
    acc, mde = test_model(model, new_loader, meta, device, desc="New Extra Test")
    
    # 輸出 JSON
    report = {"accuracy": acc, "mde": mde, "csv": new_csv_path}
    with open("New_Data_Test_Result.json", "w") as f:
        json.dump(report, f, indent=4)

def main():
    # 1. 訓練階段
    print("=== Phase 1: Training & Original Split Evaluation ===")
    # 這裡示範只跑一次 (Run 0)，你可以跑多次迴圈
    result = run_single_experiment(0)
    
    if result:
        acc, mde, saved_model_path, train_meta = result
        print(f"Run 0 Completed. Saved to {saved_model_path}")

        # 2. 純測試階段 (使用新資料集)
        # Requirement 2: 額外測試
        if os.path.exists(NEW_TEST_CSV_PATH):
            print("\n=== Phase 2: Testing on New Dataset ===")
            test_on_new_csv(saved_model_path, NEW_TEST_CSV_PATH, train_meta)
        else:
            print(f"\n[Warning] New CSV path not found: {NEW_TEST_CSV_PATH}")
    else:
        print("Training Failed.")

if __name__ == "__main__":
    main()