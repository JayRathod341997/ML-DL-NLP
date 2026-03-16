"""
Autoencoder - CSV Example
Reads: ../dataset/anomaly_2d.csv
Trains on label==0 (normal), then prints reconstruction error stats.
"""

from pathlib import Path

try:
    import pandas as pd
except Exception as e:  # pragma: no cover
    raise SystemExit("Missing dependency: pandas. Install with `pip install pandas`.") from e

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

df_out = df.copy()
df_out["recon_error"] = errs

print("Loaded:", csv_path)
print("Mean error (normal):", round(df_out[df_out["label"] == 0]["recon_error"].mean(), 4))
print("Mean error (anomaly):", round(df_out[df_out["label"] == 1]["recon_error"].mean(), 4))

