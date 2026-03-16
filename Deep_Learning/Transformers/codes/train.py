"""
Transformer Training Script
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import TransformerModel


def train(epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Simple sequence data
    data = torch.randint(0, 100, (1000, 20))
    loader = DataLoader(TensorDataset(data), batch_size=32, shuffle=True)

    model = TransformerModel(100, 128, 4, 2, 256, 100).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for batch in loader:
            x = batch[0].to(device)
            output = model(x[:, :-1])
            loss = criterion(output.reshape(-1, 100), x[:, 1:].reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}")


if __name__ == "__main__":
    train()
