"""
CNN - CSV Example
Reads: ../dataset/line_images_4x4.csv
CSV format:
  label,p0,p1,...,p15  where p* are pixels in row-major order
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
csv_path = (here / ".." / "dataset" / "line_images_4x4.csv").resolve()
df = pd.read_csv(csv_path)

y = torch.tensor(df["label"].to_numpy(), dtype=torch.long)
pixel_cols = [c for c in df.columns if c.startswith("p")]
X = torch.tensor(df[pixel_cols].to_numpy(), dtype=torch.float32).reshape(-1, 1, 4, 4)

model = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=2),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(8 * 3 * 3, 2),
)

loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for _ in range(120):
    logits = model(X)
    loss = loss_fn(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

with torch.no_grad():
    acc = (model(X).argmax(dim=1) == y).float().mean().item()

print("Loaded:", csv_path)
print("Accuracy:", round(acc, 3))

