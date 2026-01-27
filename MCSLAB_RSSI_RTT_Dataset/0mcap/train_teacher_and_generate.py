import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# === 設定 ===
TEACHER_FILES = {
    1: '../2026_1_1/Server_Wide_20260101_075453_location_AP1.csv',
    2: '../2026_1_1/Server_Wide_20260101_124356_location_AP2.csv',
    3: '../2026_1_1/Server_Wide_20260101_065323_location_AP3.csv',
    4: '../2026_1_1/Server_Wide_20260101_113813_location_AP4.csv'
}
STUDENT_FILE = '../2026_1_1/all/Server_Wide_20260101_140347.csv'
OUTPUT_FILE = 'Student_Data_with_PseudoRTT_Class.csv'

# 為了確保分類標籤一致，我們先定義好所有可能的 Label (根據你的 49 個 RP)
# 這裡用簡單的生成邏輯，請確保跟你的真實 Label 一致
ALL_LABELS = []
for x in range(1, 12):
    for y in range(1, 12):
        lbl = f"{x}-{y}"
        # 這裡只是示範產生 Label 列表，請根據實際存在的 Label 過濾
        # 為了保險，我們待會直接從 Teacher 檔案讀取所有出現過的 Label
        pass

# === 模型定義 ===
class TeacherMTL(nn.Module):
    def __init__(self, num_classes):
        super(TeacherMTL, self).__init__()
        # Input: 1 dim (Single RSSI)
        self.shared = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        # Task 1: Pseudo RTT (Regression) - 保持數值精度
        self.head_rtt = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # Task 2: Location (Classification) - 你的新要求
        self.head_cls = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes) # Output: Logits for each RP
        )

    def forward(self, x):
        feat = self.shared(x)
        rtt_pred = self.head_rtt(feat)
        cls_pred = self.head_cls(feat)
        return rtt_pred, cls_pred

def train_and_generate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 0. 預處理 Label Encoder
    # 掃描所有 Teacher 檔案以獲取完整的 Label 集合
    all_labels_set = set()
    for path in TEACHER_FILES.values():
        df = pd.read_csv(path)
        all_labels_set.update(df['Label'].astype(str).unique())
    
    # 建立 Label Encoder (例如: "1-1" -> 0, "1-2" -> 1)
    le = LabelEncoder()
    le.fit(list(all_labels_set))
    num_classes = len(le.classes_)
    print(f"Total Unique RPs (Classes): {num_classes}")
    print(f"Classes: {le.classes_[:5]} ...")

    # 1. 讀取 Student 資料
    df_student = pd.read_csv(STUDENT_FILE)
    pseudo_rtts = {1: [], 2: [], 3: [], 4: []}

    # 2. 訓練 4 個 Teacher 模型
    for ap_id, csv_path in TEACHER_FILES.items():
        print(f"\n>>> Processing AP {ap_id} Model...")
        
        # Load Data
        df_teacher = pd.read_csv(csv_path)
        df_teacher = df_teacher[df_teacher['Label'].isin(le.classes_)] # Filter valid labels
        
        # Prepare Inputs/Targets
        X_train = df_teacher[f'RSSI_{ap_id}'].values.reshape(-1, 1)
        y_rtt_train = (df_teacher['Dist_mm'] / 1000.0).values.reshape(-1, 1)
        y_cls_train = le.transform(df_teacher['Label'].values) # 轉成 0~48 的整數

        # Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        # To Tensor
        t_X = torch.FloatTensor(X_train).to(device)
        t_y_rtt = torch.FloatTensor(y_rtt_train).to(device)
        t_y_cls = torch.LongTensor(y_cls_train).to(device) # Class 必須是 LongTensor

        # Model Setup
        model = TeacherMTL(num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.002)
        
        # Loss: MSE (for RTT) + CrossEntropy (for Classification)
        criterion_rtt = nn.MSELoss()
        criterion_cls = nn.CrossEntropyLoss()

        # Training
        model.train()
        for epoch in range(300):
            optimizer.zero_grad()
            p_rtt, p_cls = model(t_X)
            
            loss_rtt = criterion_rtt(p_rtt, t_y_rtt)
            loss_cls = criterion_cls(p_cls, t_y_cls)
            
            # Weighted Sum: 0.5 * RTT + 0.5 * Classification
            # Classification Loss 通常比較大，權重可以自行調整
            loss = loss_rtt + loss_cls 
            
            loss.backward()
            optimizer.step()
        
        print(f"    AP {ap_id} Final Loss - RTT: {loss_rtt.item():.4f}, Cls: {loss_cls.item():.4f}")

        # --- Inference on Student ---
        model.eval()
        X_student = df_student[f'RSSI_{ap_id}'].values.reshape(-1, 1)
        X_student = scaler.transform(X_student) # Use Teacher's scaler
        
        with torch.no_grad():
            t_X_student = torch.FloatTensor(X_student).to(device)
            pred_rtt, _ = model(t_X_student)
            pseudo_rtts[ap_id] = pred_rtt.cpu().numpy().flatten()

    # 3. Save
    # 將生成的 Pseudo RTT 寫入 DataFrame
    for ap_id in [1, 2, 3, 4]:
        df_student[f'Pseudo_RTT_{ap_id}'] = pseudo_rtts[ap_id]
    
    # 也把 Label 編碼存進去，方便 Part 2 使用
    # 注意：Student 資料集裡面的 Label 也要符合 Teacher 的 Label 集合
    df_student = df_student[df_student['Label'].isin(le.classes_)]
    df_student['Label_Idx'] = le.transform(df_student['Label'].values)
    
    df_student.to_csv(OUTPUT_FILE, index=False)
    
    # 儲存 Label Encoder 的對照表，最後轉回來要用
    np.save('label_classes.npy', le.classes_)
    print(f"Done! Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    train_and_generate()