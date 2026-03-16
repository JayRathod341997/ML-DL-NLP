"""
Training Module

Training loop implementation with logging, validation, and checkpointing.
Device-agnostic setup for CPU/GPU training.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import NeuralNetwork, count_parameters, get_device


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
) -> dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Dictionary with training metrics
    """
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return {"loss": avg_loss, "accuracy": accuracy}


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> dict[str, float]:
    """
    Validate the model.

    Args:
        model: Neural network model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on

    Returns:
        Dictionary with validation metrics
    """
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total

    return {"loss": avg_loss, "accuracy": accuracy}


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    config: dict,
) -> dict:
    """
    Full training loop with logging and checkpointing.

    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        config: Training configuration

    Returns:
        Training history dictionary
    """
    device = get_device()
    print(f"Using device: {device}")
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get("learning_rate", 0.001),
        weight_decay=config.get("weight_decay", 0.0),
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min" if val_loader else "max",
        patience=config.get("patience", 5),
        factor=0.5,
    )

    # TensorBoard writer
    writer = SummaryWriter(log_dir=f"runs/{datetime.now().strftime('%Y%m%d-%H%M%S')}")

    # Training loop
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    epochs = config.get("epochs", 50)

    for epoch in range(epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        if val_loader:
            val_metrics = validate(model, val_loader, criterion, device)
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])

            # Learning rate adjustment
            scheduler.step(val_metrics["loss"])

            # Print progress
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.2f}% | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.2f}%"
            )

            # Save best model
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                save_checkpoint(model, optimizer, epoch, val_metrics, "best_model.pt")

        else:
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.2f}%"
            )

        # Log to TensorBoard
        writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
        writer.add_scalar("Accuracy/train", train_metrics["accuracy"], epoch)
        if val_loader:
            writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
            writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, train_metrics, f"checkpoint_epoch_{epoch+1}.pt"
            )

    writer.close()

    # Save final model
    save_checkpoint(model, optimizer, epochs - 1, train_metrics, "final_model.pt")

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")

    return history


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: dict,
    filename: str,
) -> None:
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Current metrics
        filename: Output filename
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, Path("checkpoints") / filename)


def load_checkpoint(model: nn.Module, filename: str, device: str = "cpu") -> dict:
    """
    Load model checkpoint.

    Args:
        model: Model to load weights into
        filename: Checkpoint file path
        device: Device to load to

    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def main() -> None:
    """Main training function with command-line arguments."""
    parser = argparse.ArgumentParser(description="Train Neural Network")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience"
    )
    parser.add_argument("--input-size", type=int, default=784, help="Input size")
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[256, 128],
        help="Hidden layer sizes",
    )
    parser.add_argument("--num-classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")

    args = parser.parse_args()

    # Configuration
    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
    }

    # Create model
    model = NeuralNetwork(
        input_size=args.input_size,
        hidden_sizes=args.hidden_sizes,
        num_classes=args.num_classes,
        dropout_rate=args.dropout,
    )

    print(f"Model parameters: {count_parameters(model):,}")

    # Load MNIST data
    from dataset import MNISTDataModule

    data_module = MNISTDataModule(batch_size=args.batch_size)
    train_loader = data_module.get_train_loader()
    val_loader = data_module.get_val_loader()

    # Train
    history = train(model, train_loader, val_loader, config)

    # Save history
    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
