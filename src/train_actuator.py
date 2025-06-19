# src/train_actuator.py
import argparse, os, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from dataset import Go2ActDataset
from model   import SmallMLP, BigMLP
from utils   import metrics
from tqdm    import tqdm

def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--hist', type=int, default=3)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch', type=int, default=4096)
    p.add_argument('--lr',    type=float, default=1e-3)
    p.add_argument('--fused', action='store_true')
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', default='../models')
    return p.parse_args()

def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    train_ds = Go2ActDataset(args.csv, args.hist, "train")
    val_ds   = Go2ActDataset(args.csv, args.hist, "val")
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    val_X, val_Y = val_ds.X.to(args.device), val_ds.Y.to(args.device)

    if args.fused:
        net  = BigMLP().to(args.device)
        opt  = optim.Adam(net.parameters(), lr=args.lr)
        nets = [net]  # for unified saving logic
    else:
        nets = [SmallMLP(in_dim=args.hist*2).to(args.device) for _ in range(12)]
        opts = [optim.Adam(n.parameters(), lr=args.lr) for n in nets]

    lossf = nn.MSELoss()
    best  = 1e9
    for ep in range(args.epochs):
        for X, Y in tqdm(train_dl, desc=f"Epoch {ep}"):
            X, Y = X.to(args.device), Y.to(args.device)
            if args.fused:
                opt.zero_grad()
                pred = net(X.reshape(len(X), -1))
                loss = lossf(pred, Y)
                loss.backward(); opt.step()
            else:
                X = X.permute(1,0,2)  # 12,B,8
                for j,(n,optj) in enumerate(zip(nets,opts)):
                    optj.zero_grad()
                    out = n(X[j])
                    loss = lossf(out, Y[:,j:j+1])
                    loss.backward(); optj.step()

        # ---- validation ----
        with torch.no_grad():
            if args.fused:
                pred = net(val_X.reshape(len(val_X), -1))
            else:
                vx = val_X.permute(1,0,2)
                out = [n(vx[j]) for j,n in enumerate(nets)]
                pred = torch.stack(out, dim=1).squeeze(-1)
            rmse, mae, r2 = metrics(pred, val_Y)
            print(f"[Val] RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.3f}")
            if rmse < best: best = rmse
            if rmse < 0.7:
                print("Early stop (0.7 N·m reached)"); break

    # ---- save ----
    if args.fused:
        torch.jit.script(net.cpu()).save(f"{args.out}/go2_act_net_fused.pt")
    else:
        for j,n in enumerate(nets):
            torch.jit.script(n.cpu()).save(f"{args.out}/go2_act_net_joint{j}.pt")
    print(f"done, best RMSE {best:.3f}")

if __name__ == "__main__":
    main()
