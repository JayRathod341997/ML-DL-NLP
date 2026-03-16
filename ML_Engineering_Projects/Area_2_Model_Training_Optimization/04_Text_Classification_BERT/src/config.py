from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    # Model
    model_name: str = "distilbert-base-uncased"
    num_labels: int = 4
    max_length: int = 128
    dropout: float = 0.1

    # Training
    num_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_clip: float = 1.0

    # Data
    dataset_name: str = "ag_news"
    train_split: str = "train"
    test_split: str = "test"
    val_size: float = 0.1  # fraction of train to use for validation

    # Output
    output_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    log_steps: int = 100
    eval_steps: int = 500
    save_best: bool = True

    # Hardware
    device: str = "cpu"  # "cuda" if available
    num_workers: int = 4

    # Experiment tracking (optional)
    use_wandb: bool = False
    wandb_project: str = "text-classification"
