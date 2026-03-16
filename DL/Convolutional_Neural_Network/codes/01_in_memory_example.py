"""
CNN - In-Memory Example
Synthetic 4x4 images: horizontal vs vertical line classification.
"""

import random

try:
    import torch
    from torch import nn
except Exception as e:  # pragma: no cover
    raise SystemExit("Missing dependency: torch. Install with `pip install torch`.") from e


def make_image(label, size=4):
    # label 0: horizontal line, label 1: vertical line
    img = [[0.0 for _ in range(size)] for _ in range(size)]
    if label == 0:
        r = random.randrange(size)
        for c in range(size):
            img[r][c] = 1.0
    else:
        c = random.randrange(size)
        for r in range(size):
            img[r][c] = 1.0
    return img


random.seed(42)
torch.manual_seed(42)

X_list, y_list = [], []
for _ in range(200):
    label = random.randrange(2)
    img = make_image(label)
    X_list.append(img)
    y_list.append(label)

X = torch.tensor(X_list, dtype=torch.float32).unsqueeze(1)  # (N,1,4,4)
y = torch.tensor(y_list, dtype=torch.long)

model = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=2),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(8 * 3 * 3, 2),
)

loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for _ in range(100):
    logits = model(X)
    loss = loss_fn(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

with torch.no_grad():
    pred = model(X).argmax(dim=1)
    acc = (pred == y).float().mean().item()

print("Train accuracy:", round(acc, 3))

