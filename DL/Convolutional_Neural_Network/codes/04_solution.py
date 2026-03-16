"""
Solution: CNN with small pixel noise
"""

import random

try:
    import torch
    from torch import nn
except Exception as e:  # pragma: no cover
    raise SystemExit("Missing dependency: torch. Install with `pip install torch`.") from e


def make_noisy_image(label, size=4, noise_flips=2):
    img = [[0.0 for _ in range(size)] for _ in range(size)]
    if label == 0:
        r = random.randrange(size)
        for c in range(size):
            img[r][c] = 1.0
    else:
        c = random.randrange(size)
        for r in range(size):
            img[r][c] = 1.0

    for _ in range(noise_flips):
        rr = random.randrange(size)
        cc = random.randrange(size)
        img[rr][cc] = 1.0 - img[rr][cc]
    return img


random.seed(42)
torch.manual_seed(42)

X_list, y_list = [], []
for _ in range(300):
    label = random.randrange(2)
    X_list.append(make_noisy_image(label))
    y_list.append(label)

X = torch.tensor(X_list, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(y_list, dtype=torch.long)

model = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=2),
    nn.ReLU(),
    nn.Conv2d(8, 16, kernel_size=2),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(16 * 2 * 2, 2),
)

loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.03)

for _ in range(200):
    logits = model(X)
    loss = loss_fn(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

with torch.no_grad():
    acc = (model(X).argmax(dim=1) == y).float().mean().item()

print("Train accuracy:", round(acc, 3))

