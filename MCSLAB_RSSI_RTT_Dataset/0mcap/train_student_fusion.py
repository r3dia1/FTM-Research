import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# === 設定 ===
DATA_FILE = 'Student_Data_with_PseudoRTT_Class.csv'
BATCH_SIZE = 32
NUM_EPOCHS = 200

# === 模型定義 (Classification) ===
class FusionClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FusionClassifier, self).__init__()
        # Input: 4 RSSI + 4 Pseudo RTT = 8 features
        self.net = nn.Sequential(
            nn.Linear(8, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes) # Output: logits for each RP
        )

    def forward(self, x):
        return self.net(x)

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) # Classification 需要 LongTensor
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def train_fusion():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    df = pd.read_csv(DATA_FILE)
    classes = np.load('label_classes.npy', allow_pickle=True)
    num_classes = len(classes)
    
    # Features
    feature_cols = ['RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4', 
                    'Pseudo_RTT_1', 'Pseudo_RTT_2', 'Pseudo_RTT_3', 'Pseudo_RTT_4']
    X = df[feature_cols].values
    y = df['Label_Idx'].values # 這是 Part 1 已經轉好的 0~48
    
    # Standardize Inputs
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    train_loader = DataLoader(SimpleDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(SimpleDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Model
    model = FusionClassifier(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Start Training Fusion Model on {device}...")
    
    # 3. Training Loop
    best_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            outputs = model(bx)
            loss = criterion(outputs, by)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                outputs = model(bx)
                _, predicted = torch.max(outputs.data, 1)
                total += by.size(0)
                correct += (predicted == by).sum().item()
        
        acc = 100 * correct / total
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1} | Test Acc: {acc:.2f}%")
            
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_fusion_cls.pth')
            
    print(f"Best Accuracy: {best_acc:.2f}%")
    
    # 4. Final Inference Example (Optional)
    # 展示一下預測出來的結果是什麼 Label
    model.load_state_dict(torch.load('best_fusion_cls.pth'))
    model.eval()
    bx, by = next(iter(test_loader))
    bx = bx.to(device)
    out = model(bx)
    _, pred = torch.max(out, 1)
    
    print("\n--- Sample Predictions ---")
    print(f"Predicted Indices: {pred[:5].cpu().numpy()}")
    print(f"Predicted Labels : {classes[pred[:5].cpu().numpy()]}")
    print(f"True Labels      : {classes[by[:5].numpy()]}")

if __name__ == "__main__":
    train_fusion()