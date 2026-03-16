"""
Transformer Encoder - CSV Example
Reads: ../dataset/int_sequences.csv
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
csv_path = (here / ".." / "dataset" / "int_sequences.csv").resolve()
df = pd.read_csv(csv_path)

seq_cols = [c for c in df.columns if c.startswith("x")]
X = torch.tensor(df[seq_cols].to_numpy(), dtype=torch.long)
y = torch.tensor(df["y"].to_numpy(), dtype=torch.long)

vocab = int(X.max().item()) + 1
seq_len = X.shape[1]

d_model = 32
emb = nn.Embedding(vocab, d_model)
pos_emb = nn.Embedding(seq_len, d_model)
pos_ids = torch.arange(seq_len).unsqueeze(0)
encoder = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True),
    num_layers=2,
)
head = nn.Linear(d_model, 2)

opt = torch.optim.Adam(
    list(emb.parameters())
    + list(pos_emb.parameters())
    + list(encoder.parameters())
    + list(head.parameters()),
    lr=0.01,
)
loss_fn = nn.CrossEntropyLoss()

for _ in range(250):
    x = emb(X) + pos_emb(pos_ids)
    logits = head(encoder(x)[:, -1, :])
    loss = loss_fn(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

with torch.no_grad():
    x = emb(X) + pos_emb(pos_ids)
    logits = head(encoder(x)[:, -1, :])
    acc = (logits.argmax(dim=1) == y).float().mean().item()

print("Loaded:", csv_path)
print("Accuracy:", round(acc, 3))
