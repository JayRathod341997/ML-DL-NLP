"""
Autoencoder Training Script
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import Autoencoder
import numpy as np


def train(epochs=50, lr=0.001, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate sample data
    data = np.random.randn(1000, 64)
    dataset = TensorDataset(torch.FloatTensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Autoencoder(64, 32, 8).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            x = batch[0].to(device)
            x_recon, _ = model(x)
            loss = criterion(x_recon, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")

    torch.save(model.state_dict(), "autoencoder.pth")
    print("Model saved")


if __name__ == "__main__":
    train()
