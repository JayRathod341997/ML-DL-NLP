"""
Toy GAN - In-Memory Example (2D points)
Generates samples near a simple 2D Gaussian.
"""

import random

try:
    import torch
    from torch import nn
except Exception as e:  # pragma: no cover
    raise SystemExit("Missing dependency: torch. Install with `pip install torch`.") from e


random.seed(42)
torch.manual_seed(42)

# Real data: 2D Gaussian around (2,2)
real = torch.randn(256, 2) * 0.5 + torch.tensor([2.0, 2.0])

G = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))
D = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))

loss_fn = nn.BCEWithLogitsLoss()
opt_g = torch.optim.Adam(G.parameters(), lr=0.02)
opt_d = torch.optim.Adam(D.parameters(), lr=0.02)

for _ in range(500):
    # Train discriminator
    z = torch.randn(256, 2)
    fake = G(z).detach()
    d_real = D(real)
    d_fake = D(fake)
    loss_d = loss_fn(d_real, torch.ones_like(d_real)) + loss_fn(
        d_fake, torch.zeros_like(d_fake)
    )
    opt_d.zero_grad()
    loss_d.backward()
    opt_d.step()

    # Train generator
    z = torch.randn(256, 2)
    fake = G(z)
    d_fake = D(fake)
    loss_g = loss_fn(d_fake, torch.ones_like(d_fake))
    opt_g.zero_grad()
    loss_g.backward()
    opt_g.step()

with torch.no_grad():
    samples = G(torch.randn(5, 2))
print("Generated samples (first 5):")
for s in samples.tolist():
    print([round(v, 3) for v in s])

