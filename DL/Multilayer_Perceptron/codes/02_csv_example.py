"""
MLP - CSV Example (XOR) using PyTorch
Reads: ../dataset/xor.csv
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
csv_path = (here / ".." / "dataset" / "xor.csv").resolve()
df = pd.read_csv(csv_path)

X = torch.tensor(df[["x1", "x2"]].to_numpy(), dtype=torch.float32)
y = torch.tensor(df[["y"]].to_numpy(), dtype=torch.float32)

model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for _ in range(300):
    logits = model(X)
    loss = loss_fn(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

with torch.no_grad():
    probs = torch.sigmoid(model(X)).squeeze(1)
    preds = (probs >= 0.5).int().tolist()
print("Loaded:", csv_path)
print("Preds:", preds)
print("True :", df["y"].astype(int).tolist())

