"""
Solution: Noisy XOR with a slightly larger MLP
"""

import random

try:
    import torch
    from torch import nn
except Exception as e:  # pragma: no cover
    raise SystemExit("Missing dependency: torch. Install with `pip install torch`.") from e


random.seed(42)
torch.manual_seed(42)

base = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
]

X_list, y_list = [], []
for _ in range(400):
    x, y = random.choice(base)
    x = [v + random.uniform(-0.08, 0.08) for v in x]
    X_list.append(x)
    y_list.append([y])

X = torch.tensor(X_list, dtype=torch.float32)
y = torch.tensor(y_list, dtype=torch.float32)

model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.03)

for _ in range(400):
    logits = model(X)
    loss = loss_fn(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

with torch.no_grad():
    pred = (torch.sigmoid(model(X)) >= 0.5).float()
    acc = (pred.eq(y)).float().mean().item()

print("Final accuracy:", round(acc, 3))

