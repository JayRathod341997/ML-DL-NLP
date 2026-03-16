from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


class ResultWriter:
    """Writes prediction results to Parquet or JSONL, and errors to errors.jsonl."""

    def __init__(self, output_path: str | Path, format: str = "parquet") -> None:
        self.output_path = Path(output_path)
        self.format = format
        self._buffer: list[dict] = []
        self._error_path = self.output_path.parent / "errors.jsonl"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def write_batch(self, rows: list[dict]) -> None:
        """Buffer a batch of result rows."""
        self._buffer.extend(rows)

    def write_error(self, row_index: int, text: str, error: str) -> None:
        """Append a failed row to errors.jsonl."""
        with open(self._error_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"row_index": row_index, "text": text[:100], "error": error}) + "\n")

    def flush(self) -> None:
        """Write all buffered results to disk."""
        if not self._buffer:
            return
        df = pd.DataFrame(self._buffer)
        if self.format == "parquet":
            df.to_parquet(self.output_path, index=False)
        elif self.format == "jsonl":
            df.to_json(self.output_path, orient="records", lines=True)
        else:
            raise ValueError(f"Unknown format: {self.format}")
        print(f"Written {len(self._buffer)} rows to {self.output_path}")

    def __len__(self) -> int:
        return len(self._buffer)
