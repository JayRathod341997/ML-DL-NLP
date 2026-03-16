from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd
from tqdm import tqdm


class ChunkedReader:
    """Reads JSONL or CSV files in configurable chunks to control memory usage."""

    def __init__(self, chunk_size: int = 10_000, text_column: str = "text") -> None:
        self.chunk_size = chunk_size
        self.text_column = text_column

    def read_file(self, path: str | Path) -> Iterator[list[dict]]:
        """Yield chunks of rows from a JSONL or CSV file.

        Args:
            path: Path to a JSONL or CSV file.

        Yields:
            List of dicts, each representing a row with at least 'text' key.
        """
        path = Path(path)
        if path.suffix == ".jsonl":
            yield from self._read_jsonl(path)
        elif path.suffix == ".csv":
            yield from self._read_csv(path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

    def read_hf_dataset(self, dataset_name: str, split: str = "test") -> Iterator[list[dict]]:
        """Read from a HuggingFace dataset in chunks.

        Args:
            dataset_name: HuggingFace dataset name (e.g., "imdb").
            split: Dataset split to use.

        Yields:
            List of dicts with 'text' key.
        """
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split=split)
        for i in range(0, len(ds), self.chunk_size):
            chunk = ds[i : i + self.chunk_size]
            texts = chunk.get(self.text_column, chunk.get("text", []))
            yield [{"text": t, "row_index": i + j} for j, t in enumerate(texts)]

    def _read_jsonl(self, path: Path) -> Iterator[list[dict]]:
        chunk = []
        with open(path, encoding="utf-8") as f:
            import json
            for i, line in enumerate(f):
                try:
                    row = json.loads(line)
                    if self.text_column not in row:
                        row["text"] = str(row)
                    row["row_index"] = i
                    chunk.append(row)
                except json.JSONDecodeError:
                    continue
                if len(chunk) >= self.chunk_size:
                    yield chunk
                    chunk = []
        if chunk:
            yield chunk

    def _read_csv(self, path: Path) -> Iterator[list[dict]]:
        for df in pd.read_csv(path, chunksize=self.chunk_size):
            if self.text_column not in df.columns:
                raise ValueError(f"Column '{self.text_column}' not found in CSV")
            yield df.reset_index().rename(columns={"index": "row_index"}).to_dict("records")
