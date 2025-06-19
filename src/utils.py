# src/utils.py
import torch

def metrics(pred: torch.Tensor, label: torch.Tensor):
    diff = pred - label
    rmse = torch.sqrt(torch.mean(diff ** 2)).item()
    mae  = torch.mean(torch.abs(diff)).item()
    r2   = 1 - torch.sum(diff ** 2) / torch.sum((label - label.mean()) ** 2)
    return rmse, mae, r2.item()
