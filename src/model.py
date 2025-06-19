# src/model.py
import torch.nn as nn, torch

class SmallMLP(nn.Module):
    """单关节 3×32 Softsign"""
    def __init__(self, in_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.Softsign(),
            nn.Linear(32, 32),     nn.Softsign(),
            nn.Linear(32, 32),     nn.Softsign(),
            nn.Linear(32, 1)
        )
    def forward(self, x):          # x: (B,8)
        return self.net(x)

class BigMLP(nn.Module):
    """一次性 96→12"""
    def __init__(self, in_dim: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.Softsign(),
            nn.Linear(256, 256),    nn.Softsign(),
            nn.Linear(256, 12)
        )
    def forward(self, x):          # x: (B,96)
        return self.net(x)
