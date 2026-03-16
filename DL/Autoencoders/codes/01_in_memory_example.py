"""
Autoencoder - In-Memory Example
Train on 2D normal points; use reconstruction error for anomaly scoring.
"""

import random

try:
    import torch
    from torch import nn
except Exception as e:  # pragma: no cover
    raise SystemExit("Missing dependency: torch. Install with `pip install torch`.") from e


random.seed(42)
torch.manual_seed(42)

# Normal points near (0,0)
X_normal = []
for _ in range(400):
    X_normal.append([random.gauss(0, 0.7), random.gauss(0, 0.7)])

X = torch.tensor(X_normal, dtype=torch.float32)

ae = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 2),
)

loss_fn = nn.MSELoss()
opt = torch.optim.Adam(ae.parameters(), lr=0.02)

for _ in range(400):
    recon = ae(X)
    loss = loss_fn(recon, X)
    opt.zero_grad()
    loss.backward()
    opt.step()

with torch.no_grad():
    normal = torch.tensor([[0.2, -0.1]], dtype=torch.float32)
    anomaly = torch.tensor([[4.0, 4.0]], dtype=torch.float32)
    err_normal = torch.mean((ae(normal) - normal) ** 2).item()
    err_anom = torch.mean((ae(anomaly) - anomaly) ** 2).item()

print("Reconstruction error (normal):", round(err_normal, 4))
print("Reconstruction error (anomaly):", round(err_anom, 4))

