from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BatchConfig:
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    batch_size: int = 64
    max_length: int = 128
    num_workers: int = 0  # DataLoader workers (0 = main process)
    device: str = "cpu"
    chunk_size: int = 10_000  # rows to read at a time from file
    output_format: str = "parquet"  # "parquet" or "jsonl"
    text_column: str = "text"
    show_progress: bool = True
