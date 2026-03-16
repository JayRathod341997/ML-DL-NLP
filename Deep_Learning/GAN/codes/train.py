"""
GAN Training Script
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import Generator, Discriminator


def train(epochs=100, latent_dim=64, img_dim=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Real data
    real_data = torch.randn(1000, img_dim)
    loader = DataLoader(TensorDataset(real_data), batch_size=32, shuffle=True)

    G = Generator(latent_dim, img_dim).to(device)
    D = Discriminator(img_dim).to(device)

    g_opt = torch.optim.Adam(G.parameters(), lr=0.001)
    d_opt = torch.optim.Adam(D.parameters(), lr=0.001)

    criterion = nn.BCELoss()

    for epoch in range(epochs):
        for batch in loader:
            real = batch[0].to(device)
            batch_size = real.size(0)

            # Train Discriminator
            noise = torch.randn(batch_size, latent_dim).to(device)
            fake = G(noise)

            d_loss = criterion(D(real), torch.ones_like(real)) + criterion(
                D(fake.detach()), torch.zeros_like(fake)
            )
            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            # Train Generator
            noise = torch.randn(batch_size, latent_dim).to(device)
            fake = G(noise)
            g_loss = criterion(D(fake), torch.ones_like(fake))
            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

        print(f"Epoch {epoch+1}/{epochs}, D_loss: {d_loss:.4f}, G_loss: {g_loss:.4f}")


if __name__ == "__main__":
    train()
