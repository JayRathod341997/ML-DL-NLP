"""
Solution: Longer sequences with an LSTM
y=1 if there exists a run of 5 consecutive 1s.
"""

import random

try:
    import torch
    from torch import nn
except Exception as e:  # pragma: no cover
    raise SystemExit("Missing dependency: torch. Install with `pip install torch`.") from e


def has_run(seq, k=5):
    run = 0
    for v in seq:
        if v == 1:
            run += 1
            if run >= k:
                return True
        else:
            run = 0
    return False


random.seed(42)
torch.manual_seed(42)

seq_len = 20
X_list, y_list = [], []
for _ in range(600):
    seq = [random.randrange(2) for _ in range(seq_len)]
    y_list.append(1 if has_run(seq, k=5) else 0)
    X_list.append(seq)

X = torch.tensor(X_list, dtype=torch.float32).unsqueeze(-1)
y = torch.tensor(y_list, dtype=torch.long)

lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
head = nn.Linear(32, 2)

loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(list(lstm.parameters()) + list(head.parameters()), lr=0.01)

for _ in range(400):
    out, _ = lstm(X)
    logits = head(out[:, -1, :])
    loss = loss_fn(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

with torch.no_grad():
    out, _ = lstm(X)
    acc = (head(out[:, -1, :]).argmax(dim=1) == y).float().mean().item()

print("Train accuracy:", round(acc, 3))

