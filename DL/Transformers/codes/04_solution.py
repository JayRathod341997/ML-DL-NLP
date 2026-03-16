"""
Solution: Pattern [3,1,4] detection using a Transformer encoder
"""

import random

try:
    import torch
    from torch import nn
except Exception as e:  # pragma: no cover
    raise SystemExit("Missing dependency: torch. Install with `pip install torch`.") from e


def has_pattern(seq, pattern=(3, 1, 4)):
    for i in range(len(seq) - len(pattern) + 1):
        if tuple(seq[i : i + len(pattern)]) == pattern:
            return True
    return False


random.seed(42)
torch.manual_seed(42)

vocab = 10
seq_len = 12

X_list, y_list = [], []
for _ in range(1200):
    seq = [random.randrange(vocab) for _ in range(seq_len)]
    y = 1 if has_pattern(seq) else 0
    X_list.append(seq)
    y_list.append(y)

X = torch.tensor(X_list, dtype=torch.long)
y = torch.tensor(y_list, dtype=torch.long)

d_model = 48
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
    lr=0.008,
)
loss_fn = nn.CrossEntropyLoss()

for _ in range(350):
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

print("Train accuracy:", round(acc, 3))
