import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

# ==========================================
# 1. 參數與路徑設定
# ==========================================
# 請確認以下檔案路徑是否正確
TEACHER_FILES = {
    1: '../2026_1_1/Server_Wide_20260101_075453_location_AP1.csv',
    2: '../2026_1_1/Server_Wide_20260101_124356_location_AP2.csv',
    3: '../2026_1_1/Server_Wide_20260101_065323_location_AP3.csv',
    4: '../2026_1_1/Server_Wide_20260101_113813_location_AP4.csv'
}
STUDENT_FILE = '../2026_1_1/all/Server_Wide_20260101_140347.csv'

SELECTED_FEATURES = [
    'RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4',
    # 'Pseudo_RTT_1', 'Pseudo_RTT_2', 'Pseudo_RTT_3', 'Pseudo_RTT_4'
]

# 座標映射表 (49 RPs)
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
TEACHER_EPOCHS = 150
FUSION_EPOCHS = 200

# ==========================================
# 2. 模型定義
# ==========================================

# Stage 1: Teacher Multi-Task Model (Pseudo RTT Generator)
class TeacherMTL(nn.Module):
    def __init__(self, num_classes):
        super(TeacherMTL, self).__init__()
        # Input: Single RSSI (1 dim)
        self.shared = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        # Head 1: RTT Regression (保持物理意義)
        self.head_rtt = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # Head 2: RP Classification (輔助任務)
        self.head_cls = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        feat = self.shared(x)
        return self.head_rtt(feat), self.head_cls(feat)

# Stage 2: Fusion Model (Classification Only)
class FusionClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FusionClassifier, self).__init__()
        # Input: 4 RSSI + 4 Pseudo RTT = 8 dims
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes) # Output: Logits
        )

    def forward(self, x):
        return self.net(x)

# Dataset Wrapper
class SimpleDataset(Dataset):
    def __init__(self, X, y_rtt=None, y_cls=None, mode='train'):
        self.X = torch.FloatTensor(X)
        self.mode = mode
        if mode == 'train':
            self.y_rtt = torch.FloatTensor(y_rtt) if y_rtt is not None else None
            self.y_cls = torch.LongTensor(y_cls) if y_cls is not None else None
            
    def __len__(self): return len(self.X)
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            if self.y_rtt is not None:
                return self.X[idx], self.y_rtt[idx], self.y_cls[idx]
            else:
                return self.X[idx], self.y_cls[idx]
        else:
            return self.X[idx] 

# ==========================================
# 3. 核心流程函數 (修正欄位讀取邏輯)
# ==========================================

def get_label_encoder():
    labels = sorted(list(LABEL_TO_COORDS.keys()))
    le = LabelEncoder()
    le.fit(labels)
    return le

def train_teacher_and_generate_pseudo(df_student, le):
    """
    [Fix] 加入 NaN 檢查、Drop Last 與 Gradient Clipping 的修正版本
    """
    print("\n[Stage 1] Training Teacher Models & Generating Pseudo RTTs...")
    pseudo_rtts = {}
    num_classes = len(le.classes_)
    
    # 建立 loss_plots 資料夾
    if not os.path.exists('loss_plots'):
        os.makedirs('loss_plots')
    
    for ap_id, path in TEACHER_FILES.items():
        print(f"  >>> Processing AP {ap_id} Model (using file: {os.path.basename(path)})...")
        
        # --- A. 讀取與清理資料 ---
        df = pd.read_csv(path)
        
        # 1. 過濾 Label
        df = df[df['Label'].isin(le.classes_)]
        
        teacher_rssi_col = 'RSSI_1' 
        teacher_dist_col = 'Dist_mm_1'
        
        # 2. 檢查欄位是否存在
        if teacher_rssi_col not in df.columns:
            print(f"Error: {teacher_rssi_col} not found in {path}.")
            continue

        # 3. [Fix] 強制移除 NaN
        original_len = len(df)
        df = df.dropna(subset=[teacher_rssi_col, teacher_dist_col])
        if len(df) < original_len:
            print(f"      [Warning] Dropped {original_len - len(df)} rows containing NaN values.")
        
        if len(df) == 0:
            print(f"      [Error] No valid data left for AP {ap_id}. Skipping.")
            continue

        X = df[teacher_rssi_col].values.reshape(-1, 1)
        y_rtt = (df[teacher_dist_col] / 1000.0).values.reshape(-1, 1) # mm -> m
        y_cls = le.transform(df['Label'].values)
        
        # [Debug] 印出資料範圍檢查
        print(f"      [Data Check] RSSI Range: {X.min()} ~ {X.max()}, Dist Range: {y_rtt.min()} ~ {y_rtt.max()}")

        # 標準化 RSSI
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # [Fix] drop_last=True 防止 BatchNorm 在最後一個 batch (size=1) 出錯
        ds = SimpleDataset(X, y_rtt, y_cls, mode='train')
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        
        # --- B. 訓練模型 ---
        model = TeacherMTL(num_classes).to(DEVICE)
        # [Fix] 調降 Learning Rate 增加穩定性
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        crit_rtt = nn.MSELoss()
        crit_cls = nn.CrossEntropyLoss()
        
        history = {'rtt': [], 'cls': [], 'total': []}
        
        model.train()
        for epoch in range(TEACHER_EPOCHS):
            epoch_rtt = 0.0
            epoch_cls = 0.0
            epoch_total = 0.0
            batch_count = 0
            
            for bx, by_r, by_c in dl:
                bx, by_r, by_c = bx.to(DEVICE), by_r.to(DEVICE), by_c.to(DEVICE)
                
                optimizer.zero_grad()
                pred_r, pred_c = model(bx)
                
                l_rtt = crit_rtt(pred_r, by_r)
                l_cls = crit_cls(pred_c, by_c)
                
                # 若 loss 已經是 NaN，直接報錯停止
                if torch.isnan(l_rtt) or torch.isnan(l_cls):
                    print(f"      [Error] NaN detected at Epoch {epoch}!")
                    break

                loss = l_rtt + 0.5 * l_cls
                loss.backward()
                
                # [Fix] Gradient Clipping 防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_rtt += l_rtt.item()
                epoch_cls += l_cls.item()
                epoch_total += loss.item()
                batch_count += 1
            
            if batch_count > 0:
                avg_total = epoch_total / batch_count
                avg_rtt = epoch_rtt / batch_count
                avg_cls = epoch_cls / batch_count
            else:
                avg_total = 0
            
            history['rtt'].append(avg_rtt)
            history['cls'].append(avg_cls)
            history['total'].append(avg_total)

            if (epoch + 1) % 20 == 0:
                print(f"      Epoch {epoch+1}/{TEACHER_EPOCHS} - Total: {avg_total:.4f} (RTT: {avg_rtt:.4f}, Cls: {avg_cls:.4f})")

        # --- B-2. 畫圖 ---
        plt.figure(figsize=(10, 5))
        plt.plot(history['rtt'], label='RTT Loss (MSE)', linestyle='--')
        plt.plot(history['cls'], label='Class Loss (CE)', linestyle='--')
        plt.plot(history['total'], label='Total Loss', linewidth=2)
        plt.title(f'AP {ap_id} MTL Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'loss_plots/AP_{ap_id}_loss.png')
        plt.close()
        
        # --- C. 推論 Student 資料 ---
        student_rssi_col = f'RSSI_{ap_id}'
        model.eval()
        
        # 取 Student 資料
        X_stu = df_student[student_rssi_col].values.reshape(-1, 1)
        
        # 檢查 Student RSSI 是否有 NaN，若有填補為 -100 (假設收不到訊號) 或 平均值
        if np.isnan(X_stu).any():
            print(f"      [Warning] Student data contains NaN in {student_rssi_col}. Filling with -100.")
            X_stu = np.nan_to_num(X_stu, nan=-100.0)

        X_stu = scaler.transform(X_stu)
        
        preds = []
        with torch.no_grad():
            t_x = torch.FloatTensor(X_stu).to(DEVICE)
            p_rtt, _ = model(t_x)
            preds = p_rtt.cpu().numpy().flatten()
            
        pseudo_rtts[ap_id] = preds
        
    return pseudo_rtts

def calculate_mde(pred_indices, true_indices, le):
    pred_labels = le.inverse_transform(pred_indices)
    true_labels = le.inverse_transform(true_indices)
    
    distances = []
    for p_lbl, t_lbl in zip(pred_labels, true_labels):
        p_coord = np.array(LABEL_TO_COORDS[p_lbl])
        t_coord = np.array(LABEL_TO_COORDS[t_lbl])
        dist = np.linalg.norm(p_coord - t_coord)
        distances.append(dist)
        
    return np.mean(distances), distances

# ==========================================
# 4. 主程式
# ==========================================

def main():
    le = get_label_encoder()
    print(f"Target Labels: {len(le.classes_)} classes.")
    
    # 讀取 Student 資料
    # (記得加上 skipinitialspace=True 防止欄位名稱有空白)
    df_student = pd.read_csv(STUDENT_FILE, skipinitialspace=True)
    
    # 清理欄位空白 (保險起見)
    df_student.columns = df_student.columns.str.strip()
    
    df_student = df_student[df_student['Label'].isin(le.classes_)].copy()
    print(f"Student Data Loaded: {len(df_student)} samples.")

    # [Stage 1] 生成 Pseudo RTT
    # 注意：即使 SELECTED_FEATURES 不包含 Pseudo RTT，
    # 這裡還是建議先生成，以免你切換特徵時要重新跑這一段耗時的訓練。
    # 或者你可以加個判斷：只有當 SELECTED_FEATURES 包含 'Pseudo_RTT' 時才跑這段。
    
    need_pseudo_rtt = any('Pseudo_RTT' in f for f in SELECTED_FEATURES)
    
    if need_pseudo_rtt:
        pseudo_rtt_dict = train_teacher_and_generate_pseudo(df_student, le)
        for i in range(1, 5):
            df_student[f'Pseudo_RTT_{i}'] = pseudo_rtt_dict[i]
    else:
        print("[Info] Skipping Stage 1 (Pseudo RTT generation) as it's not in SELECTED_FEATURES.")

    # [Stage 2] 訓練 Fusion Model
    print("\n[Stage 2] Training Fusion Classification Model...")
    
    # --- [關鍵修改] 動態選取特徵 ---
    print(f"Using Features: {SELECTED_FEATURES}")
    
    # 檢查是否所有特徵都在 DataFrame 中
    missing_cols = [c for c in SELECTED_FEATURES if c not in df_student.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols}")
    
    X = df_student[SELECTED_FEATURES].values
    y = le.transform(df_student['Label'].values)
    
    # 自動計算輸入維度
    input_dim = len(SELECTED_FEATURES)
    print(f"Model Input Dimension: {input_dim}")
    
    # 標準化 Input (StandardScaler 會自動適應 column 數量)
    scaler_fusion = StandardScaler()
    X = scaler_fusion.fit_transform(X)
    
    # 切分訓練/測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    train_dl = DataLoader(SimpleDataset(X_train, y_cls=y_train, mode='train'), batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(SimpleDataset(X_test, y_cls=y_test, mode='train'), batch_size=BATCH_SIZE, shuffle=False)
    
    # --- [關鍵修改] 傳入動態計算的 input_dim ---
    model = FusionClassifier(input_dim=input_dim, num_classes=len(le.classes_)).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    best_mde = 999.0
    
    for epoch in range(FUSION_EPOCHS):
        model.train()
        for bx, by in train_dl:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for bx, by in test_dl:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                out = model(bx)
                _, pred = torch.max(out, 1)
                
                correct += (pred == by).sum().item()
                total += by.size(0)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(by.cpu().numpy())
        
        acc = 100 * correct / total
        mde, _ = calculate_mde(all_preds, all_labels, le)
        
        if acc > best_acc:
            best_acc = acc
            best_mde = mde
            
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:03d} | Acc: {acc:.2f}% | MDE: {mde:.4f}m")

    print("\n========================================")
    print(f"Final Results:")
    print(f"  > Best Test Accuracy : {best_acc:.2f}%")
    print(f"  > Best Mean Dist Error: {best_mde:.4f} m")
    print("========================================")

if __name__ == "__main__":
    main()