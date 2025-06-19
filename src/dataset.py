import pandas as pd, torch, numpy as np
from torch.utils.data import Dataset

class Go2ActDataset(Dataset):
    """
    构造论文版 Actuator-Net 样本：
        输入  X : (12, H*2)       --  每个关节  [ε(t..t-H+1), dq(t..t-H+1)]
        标签  y : (12,)           --  同时刻  τ_est(t)
    """
    def __init__(self, csv_path: str, hist_len: int = 3, split: str = "train"):
        assert split in ("train", "val")
        df  = pd.read_csv(csv_path)
        eps = df[[f"q_des{i}"  for i in range(12)]].values \
            - df[[f"q_meas{i}" for i in range(12)]].values          # (T,12)
        dq  = df[[f"dq_meas{i}" for i in range(12)]].values         # (T,12)
        tau = df[[f"tau_est{i}" for i in range(12)]].values         # (T,12)

        X, Y = [], []
        for t in range(hist_len - 1, len(df)):                      # 滑窗
            e_hist = eps[t-hist_len+1:t+1][::-1]                    # (H,12)
            d_hist = dq [t-hist_len+1:t+1][::-1]                    # (H,12)

            # ① 对每个关节拼 [ε_hist , dq_hist] → (H*2,)
            per_joint = [np.concatenate([e_hist[:, j], d_hist[:, j]])
                         for j in range(12)]                       # list 12 × (H*2,)
            X.append(np.stack(per_joint, axis=0).astype(np.float32))  # (12, H*2)
            Y.append(tau[t].astype(np.float32))                       # (12,)

        X, Y = np.stack(X), np.stack(Y)                              # (N,12,H*2) , (N,12)
        sep  = int(0.9 * len(X))
        if split == "train":
            self.X, self.Y = torch.from_numpy(X[:sep]), torch.from_numpy(Y[:sep])
        else:
            self.X, self.Y = torch.from_numpy(X[sep:]), torch.from_numpy(Y[sep:])

    def __len__(self):  return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]     # X:(12,H*2) , Y:(12,)
