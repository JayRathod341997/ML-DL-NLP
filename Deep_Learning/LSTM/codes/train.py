"""
LSTM Training Script
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import LSTM
from dataset import SequenceDataset


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = SequenceDataset(seq_length=args.seq_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = LSTM(
        input_size=1,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_size=1,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            hidden = model.init_hidden(batch_x.size(0), device)

            output, hidden = model(batch_x, hidden)
            loss = criterion(output[:, -1, :], batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{args.epochs}], Loss: {total_loss/len(train_loader):.6f}"
        )

    torch.save(model.state_dict(), "lstm_model.pth")
    print("Model saved to lstm_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seq-length", type=int, default=20)
    args = parser.parse_args()

    train(args)
