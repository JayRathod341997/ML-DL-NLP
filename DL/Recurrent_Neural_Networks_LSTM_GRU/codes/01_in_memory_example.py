"""
RNN/LSTM/GRU - In-Memory Example
Binary classification on short sequences:
  y = 1 if sum(x1..x5) >= 3 else 0
"""

import random

try:
    import torch
    from torch import nn
except Exception as e:  # pragma: no cover
    raise SystemExit("Missing dependency: torch. Install with `pip install torch`.") from e


random.seed(42)
torch.manual_seed(42)

seq_len = 5

X_list, y_list = [], []
for _ in range(300):
    seq = [random.randrange(2) for _ in range(seq_len)]
    label = 1 if sum(seq) >= 3 else 0
    X_list.append(seq)
    y_list.append(label)

X = torch.tensor(X_list, dtype=torch.float32).unsqueeze(-1)  # (N,T,1)
y = torch.tensor(y_list, dtype=torch.long)

model_type = "lstm"  # change to: rnn | gru | lstm
hidden = 16

if model_type == "rnn":
    rnn = nn.RNN(input_size=1, hidden_size=hidden, batch_first=True)
elif model_type == "gru":
    rnn = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
else:
    rnn = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)

head = nn.Linear(hidden, 2)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(list(rnn.parameters()) + list(head.parameters()), lr=0.03)

for _ in range(200):
    out, _ = rnn(X)
    last = out[:, -1, :]
    logits = head(last)
    loss = loss_fn(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

with torch.no_grad():
    out, _ = rnn(X)
    pred = head(out[:, -1, :]).argmax(dim=1)
    acc = (pred == y).float().mean().item()

print("Model:", model_type.upper())
print("Train accuracy:", round(acc, 3))

