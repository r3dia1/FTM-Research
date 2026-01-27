import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
import os

# === 1. 資料合成函數 (已修正：Teacher NaN 處理 & Sampling 優化) ===
def merge_student_teacher(student_path, teacher_configs, rssi_cols, drop_head=5, samples_per_label=None, random_state=42):
    
    print(f"Loading Student Data: {student_path}")
    try:
        df_student = pd.read_csv(student_path)
    except FileNotFoundError:
        print("Error: Student file not found.")
        return None

    # [Step 1] 移除 Student RSSI 有缺失值的 Rows
    original_len = len(df_student)
    df_student = df_student.dropna(subset=rssi_cols)
    print(f" -> Student RSSI NaN dropped: {original_len} -> {len(df_student)}")

    if 'Timestamp' not in df_student.columns:
        df_student['Timestamp'] = range(len(df_student))

    # 資料排序與 Index 生成
    df_student = df_student.sort_values(['Label', 'Timestamp'])
    df_student['Group_Idx'] = df_student.groupby('Label').cumcount()
    
    # 去除前 5 筆
    df_s_clean = df_student[df_student['Group_Idx'] >= drop_head].copy()
    df_s_clean['Group_Idx'] = df_s_clean['Group_Idx'] - drop_head

    # 開始合併 Teacher 資料
    merged_df = df_s_clean.copy()

    for col_name, config in teacher_configs.items():
        t_path = config['file']
        if not os.path.exists(t_path):
            print(f"Warning: Teacher file not found {t_path}, skipping.")
            continue
        
        df_t = pd.read_csv(t_path)
        dist_col = [c for c in df_t.columns if 'Dist_mm' in c][0] 
        
        # [Step 2] 確保 Teacher 的距離資料也沒有 NaN
        df_t = df_t.dropna(subset=[dist_col])

        if 'Timestamp' not in df_t.columns: 
            df_t['Timestamp'] = range(len(df_t))
        
        # Teacher 同樣處理
        df_t = df_t.sort_values(['Label', 'Timestamp'])
        df_t['Group_Idx'] = df_t.groupby('Label').cumcount()
        
        df_t_clean = df_t[df_t['Group_Idx'] >= drop_head].copy()
        df_t_clean['Group_Idx'] = df_t_clean['Group_Idx'] - drop_head
        
        # 準備合併
        df_t_subset = df_t_clean[['Label', 'Group_Idx', dist_col]].copy()
        df_t_subset = df_t_subset.rename(columns={dist_col: col_name})

        # Merge
        merged_df = pd.merge(merged_df, df_t_subset, on=['Label', 'Group_Idx'], how='inner')

    # [Step 3] 最終檢查：確保合併後的 RTT 欄位沒有 NaN (解決 ValueError)
    # Inner join 理論上不該有 NaN，但若 index 對不上可能會導致問題，這裡做最後保險
    before_final_drop = len(merged_df)
    merged_df = merged_df.dropna()
    if len(merged_df) < before_final_drop:
        print(f" -> Final Merge Clean: Dropped {before_final_drop - len(merged_df)} rows containing NaNs.")

    # [Step 4] 限制每個 Label 的資料量 (Sampling) - 修正 FutureWarning
    if samples_per_label is not None:
        print(f" -> Sampling {samples_per_label} rows per label...")
        
        # 使用 List Comprehension 取代 groupby.apply，效能更好且無警告
        sampled_groups = []
        for _, group in merged_df.groupby('Label'):
            n = min(len(group), samples_per_label)
            sampled_groups.append(group.sample(n=n, random_state=random_state))
        
        if sampled_groups:
            merged_df = pd.concat(sampled_groups).reset_index(drop=True)
        else:
            merged_df = pd.DataFrame(columns=merged_df.columns)

    print(f"Merged Final Data Shape: {merged_df.shape}")
    return merged_df

# === 2. k-NN 檢索模組 (不變) ===
class KNN_Retrieval_Imputer:
    def __init__(self, k=5):
        self.k = k
        self.scaler = StandardScaler()

    def fit(self, X_train_rssi, Y_train_rtt, feature_cols, target_cols):
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        X_scaled = self.scaler.fit_transform(X_train_rssi)
        self.model = KNeighborsRegressor(n_neighbors=self.k, metric='euclidean')
        
        # 這裡若還有 NaN 會報錯，但在 merge_student_teacher Step 3 已經擋掉了
        self.model.fit(X_scaled, Y_train_rtt)
        
        print(f" -> k-NN Retrieval System (k={self.k}) built on {len(X_scaled)} training samples.")

    def transform(self, X_rssi):
        X_scaled = self.scaler.transform(X_rssi)
        predicted_rtt = self.model.predict(X_scaled) 
        return predicted_rtt

# === 3. Data Loader Pipeline (不變) ===
def get_data_loaders(student_csv, teacher_configs, feature_config, batch_size=32, random_state=42, k_neighbors=5, samples_per_label=None):
    
    rssi_cols = feature_config['rssi_cols']
    
    # 1. Merge Data
    df = merge_student_teacher(
        student_csv, 
        teacher_configs, 
        rssi_cols=rssi_cols,
        drop_head=5, 
        samples_per_label=samples_per_label,
        random_state=random_state
    )
    
    if df is None or df.empty:
        print("Error: Merged DataFrame is empty.")
        return None
    
    # 2. Encode Labels
    label_encoder = LabelEncoder()
    df['Label_ID'] = label_encoder.fit_transform(df['Label'])
    
    rtt_cols = list(teacher_configs.keys()) 
    
    # 3. Split Data
    X_indices = np.arange(len(df))
    y = df['Label_ID'].values
    
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        X_indices, y, test_size=0.3, random_state=random_state, stratify=y
    )
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )
    
    df_train = df.iloc[idx_train].copy()
    df_val = df.iloc[idx_val].copy()
    df_test = df.iloc[idx_test].copy()
    
    print(f"Split Sizes -> Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    # === 4. k-NN System ===
    knn_imputer = KNN_Retrieval_Imputer(k=k_neighbors)
    knn_imputer.fit(
        df_train[rssi_cols].values, 
        df_train[rtt_cols].values,
        rssi_cols, 
        rtt_cols
    )
    
    # === 5. Replace RTT for Val/Test ===
    print(" -> Replacing Val/Test RTT with k-NN retrieved values...")
    df_val[rtt_cols] = knn_imputer.transform(df_val[rssi_cols].values)
    df_test[rtt_cols] = knn_imputer.transform(df_test[rssi_cols].values)
    
    # === 6. Standardize & Tensor ===
    final_features = rssi_cols + rtt_cols 
    print(f" -> DNN Input Features: {final_features}")
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[final_features].values)
    X_val = scaler.transform(df_val[final_features].values)
    X_test = scaler.transform(df_test[final_features].values)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    loaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    }
    
    meta = {
        'input_dim': len(final_features),
        'num_classes': len(label_encoder.classes_),
        'label_encoder': label_encoder,
        'feature_names': final_features
    }
    
    return loaders, meta