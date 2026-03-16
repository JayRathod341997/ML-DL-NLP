"""
MLP - In-Memory Example (XOR classification) using PyTorch
"""

import random

try:
    import torch
    from torch import nn
except Exception as e:  # pragma: no cover
    raise SystemExit("Missing dependency: torch. Install with `pip install torch`.") from e


random.seed(42)
torch.manual_seed(42)

# XOR dataset (repeat with tiny noise for a less trivial training loop)
base = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
]

X_list, y_list = [], []
for _ in range(200):
    x, y = random.choice(base)
    x = [v + random.uniform(-0.02, 0.02) for v in x]
    X_list.append(x)
    y_list.append([y])

X = torch.tensor(X_list, dtype=torch.float32)
y = torch.tensor(y_list, dtype=torch.float32)

model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
)

loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for epoch in range(200):
    logits = model(X)
    loss = loss_fn(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

    if (epoch + 1) % 50 == 0:
        with torch.no_grad():
            pred = (torch.sigmoid(logits) >= 0.5).float()
            acc = (pred.eq(y)).float().mean().item()
        print(f"Epoch {epoch+1:3d} | loss={loss.item():.4f} | acc={acc:.3f}")

# Test on clean XOR points
test_X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
with torch.no_grad():
    probs = torch.sigmoid(model(test_X)).squeeze(1)
print("Test probs:", [round(float(p), 3) for p in probs])

