"""
Transformer Encoder - In-Memory Example
Sequence classification on integer sequences.
Label: y=1 if last two numbers are equal else 0
"""

import random

try:
    import torch
    from torch import nn
except Exception as e:  # pragma: no cover
    raise SystemExit("Missing dependency: torch. Install with `pip install torch`.") from e


random.seed(42)
torch.manual_seed(42)

vocab = 10
seq_len = 8

X_list, y_list = [], []
for _ in range(600):
    seq = [random.randrange(vocab) for _ in range(seq_len)]
    y = 1 if seq[-1] == seq[-2] else 0
    X_list.append(seq)
    y_list.append(y)

X = torch.tensor(X_list, dtype=torch.long)  # (N,T)
y = torch.tensor(y_list, dtype=torch.long)

d_model = 32
emb = nn.Embedding(vocab, d_model)
pos_emb = nn.Embedding(seq_len, d_model)
pos_ids = torch.arange(seq_len).unsqueeze(0)

enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
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
    h = encoder(x)
    logits = head(h[:, -1, :])
    loss = loss_fn(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

with torch.no_grad():
    x = emb(X) + pos_emb(pos_ids)
    logits = head(encoder(x)[:, -1, :])
    acc = (logits.argmax(dim=1) == y).float().mean().item()

print("Train accuracy:", round(acc, 3))
