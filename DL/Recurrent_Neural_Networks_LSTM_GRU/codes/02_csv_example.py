"""
RNN/LSTM/GRU - CSV Example
Reads: ../dataset/sequence_sum.csv
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
csv_path = (here / ".." / "dataset" / "sequence_sum.csv").resolve()
df = pd.read_csv(csv_path)

seq_cols = ["x1", "x2", "x3", "x4", "x5"]
X = torch.tensor(df[seq_cols].to_numpy(), dtype=torch.float32).unsqueeze(-1)
y = torch.tensor(df["y"].to_numpy(), dtype=torch.long)

rnn = nn.GRU(input_size=1, hidden_size=16, batch_first=True)
head = nn.Linear(16, 2)

loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(list(rnn.parameters()) + list(head.parameters()), lr=0.03)

for _ in range(250):
    out, _ = rnn(X)
    logits = head(out[:, -1, :])
    loss = loss_fn(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

with torch.no_grad():
    out, _ = rnn(X)
    acc = (head(out[:, -1, :]).argmax(dim=1) == y).float().mean().item()

print("Loaded:", csv_path)
print("Accuracy:", round(acc, 3))

