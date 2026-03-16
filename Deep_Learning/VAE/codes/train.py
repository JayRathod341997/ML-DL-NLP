"""
VAE Training Script
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import VAE
import numpy as np


def vae_loss(x_recon, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


def train(epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.random.randn(1000, 64)
    dataset = TensorDataset(torch.FloatTensor(data))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = VAE(64, 32, 8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        for batch in loader:
            x = batch[0].to(device)
            x_recon, mu, logvar = model(x)
            loss = vae_loss(x_recon, x, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}")

    torch.save(model.state_dict(), "vae.pth")


if __name__ == "__main__":
    train()
