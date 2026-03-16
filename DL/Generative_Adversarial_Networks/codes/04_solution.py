"""
Solution: Slightly different learning rates for D and G
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
csv_path = (here / ".." / "dataset" / "gaussian_mixture_2d.csv").resolve()
df = pd.read_csv(csv_path)
real = torch.tensor(df[["x", "y"]].to_numpy(), dtype=torch.float32)

G = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 2))
D = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 1))

loss_fn = nn.BCEWithLogitsLoss()
opt_g = torch.optim.Adam(G.parameters(), lr=0.02)
opt_d = torch.optim.Adam(D.parameters(), lr=0.008)

for _ in range(1200):
    z = torch.randn(len(real), 2)
    fake = G(z).detach()

    loss_d = loss_fn(D(real), torch.ones(len(real), 1)) + loss_fn(
        D(fake), torch.zeros(len(real), 1)
    )
    opt_d.zero_grad()
    loss_d.backward()
    opt_d.step()

    z = torch.randn(len(real), 2)
    fake = G(z)
    loss_g = loss_fn(D(fake), torch.ones(len(real), 1))
    opt_g.zero_grad()
    loss_g.backward()
    opt_g.step()

with torch.no_grad():
    samples = G(torch.randn(200, 2)).numpy()

print("Generated mean x,y:", [round(float(v), 3) for v in samples.mean(axis=0)])
print("Generated std  x,y:", [round(float(v), 3) for v in samples.std(axis=0)])

