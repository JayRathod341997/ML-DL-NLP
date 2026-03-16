"""
Solution: 95th percentile threshold on reconstruction error
"""

from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except Exception as e:  # pragma: no cover
    raise SystemExit("Missing dependency: numpy/pandas. Install with `pip install numpy pandas`.") from e

try:
    import torch
    from torch import nn
except Exception as e:  # pragma: no cover
    raise SystemExit("Missing dependency: torch. Install with `pip install torch`.") from e


here = Path(__file__).resolve().parent
csv_path = (here / ".." / "dataset" / "anomaly_2d.csv").resolve()
df = pd.read_csv(csv_path)

df_normal = df[df["label"] == 0]
X = torch.tensor(df_normal[["f1", "f2"]].to_numpy(), dtype=torch.float32)

ae = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(ae.parameters(), lr=0.02)

for _ in range(400):
    recon = ae(X)
    loss = loss_fn(recon, X)
    opt.zero_grad()
    loss.backward()
    opt.step()

all_X = torch.tensor(df[["f1", "f2"]].to_numpy(), dtype=torch.float32)
with torch.no_grad():
    errs = torch.mean((ae(all_X) - all_X) ** 2, dim=1).numpy()

normal_errs = errs[df["label"].to_numpy() == 0]
thr = float(np.quantile(normal_errs, 0.95))
pred = (errs > thr).astype(int)
true = df["label"].to_numpy().astype(int)

tp = int(((pred == 1) & (true == 1)).sum())
fp = int(((pred == 1) & (true == 0)).sum())
fn = int(((pred == 0) & (true == 1)).sum())

precision = tp / (tp + fp + 1e-9)
recall = tp / (tp + fn + 1e-9)

print("Threshold:", round(thr, 4))
print("Precision:", round(precision, 3))
print("Recall   :", round(recall, 3))

