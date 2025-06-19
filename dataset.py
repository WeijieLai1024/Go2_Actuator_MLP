# src/dataset.py
import pandas as pd, torch, numpy as np
from torch.utils.data import Dataset

class Go2ActDataset(Dataset):
    """构造 [eps_hist, dq_hist] → tau 样本"""
    def __init__(self, csv_path: str, hist_len: int = 4, split: str = "train"):
        assert split in ("train", "val")
        df   = pd.read_csv(csv_path)
        eps  = df[[f"q_des{i}"  for i in range(12)]].values - \
               df[[f"q_meas{i}" for i in range(12)]].values
        dq   = df[[f"dq_meas{i}" for i in range(12)]].values
        tau  = df[[f"tau_est{i}" for i in range(12)]].values
        X, Y = [], []
        for t in range(hist_len - 1, len(df)):
            e_hist = eps[t - hist_len + 1 : t + 1][::-1]   # 形状 (H,12)
            d_hist = dq [t - hist_len + 1 : t + 1][::-1]
            X.append(np.hstack([e_hist, d_hist]).astype(np.float32))  # (H,24)
            Y.append(tau[t].astype(np.float32))                       # (12,)
        X = np.stack(X)  # (N,H,24)
        Y = np.stack(Y)
        sep = int(0.9 * len(X))
        if split == "train":
            self.X = torch.from_numpy(X[:sep])
            self.Y = torch.from_numpy(Y[:sep])
        else:
            self.X = torch.from_numpy(X[sep:])
            self.Y = torch.from_numpy(Y[sep:])

    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]        # X: (H,24) , Y: (12,)
