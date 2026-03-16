from __future__ import annotations

import time
from pathlib import Path

from tqdm import tqdm

from .config import BatchConfig
from .data_reader import ChunkedReader
from .model_runner import BatchModelRunner
from .postprocessor import Postprocessor
from .preprocessor import TextPreprocessor
from .writer import ResultWriter


class BatchPipeline:
    """Orchestrates the full batch inference pipeline.

    reader → preprocessor → model_runner → postprocessor → writer
    """

    def __init__(self, config: BatchConfig) -> None:
        self.config = config
        self.reader = ChunkedReader(
            chunk_size=config.chunk_size,
            text_column=config.text_column,
        )
        self.preprocessor = TextPreprocessor()
        self.model_runner = BatchModelRunner(config)
        self.postprocessor = Postprocessor()

    def run_on_file(self, input_path: str | Path, output_path: str | Path) -> dict:
        """Run the full pipeline on a JSONL or CSV file."""
        writer = ResultWriter(output_path, format=self.config.output_format)
        return self._run(self.reader.read_file(input_path), writer)

    def run_on_dataset(
        self,
        dataset_name: str,
        split: str,
        output_path: str | Path,
    ) -> dict:
        """Run the full pipeline on a HuggingFace dataset."""
        writer = ResultWriter(output_path, format=self.config.output_format)
        return self._run(self.reader.read_hf_dataset(dataset_name, split), writer)

    def _run(self, data_iter, writer: ResultWriter) -> dict:
        total_processed, total_errors = 0, 0
        start = time.perf_counter()

        for chunk in data_iter:
            texts = self.preprocessor.process_batch([r.get("text", "") for r in chunk])
            valid_rows, valid_texts = [], []
            for row, text in zip(chunk, texts):
                if text:
                    valid_rows.append(row)
                    valid_texts.append(text)
                else:
                    writer.write_error(row.get("row_index", -1), row.get("text", ""), "empty text")
                    total_errors += 1

            try:
                predictions = self.model_runner.predict_batch(valid_texts)
                results = self.postprocessor.merge(valid_rows, predictions)
                writer.write_batch(results)
                total_processed += len(results)
            except Exception as e:
                for row in valid_rows:
                    writer.write_error(row.get("row_index", -1), row.get("text", ""), str(e))
                total_errors += len(valid_rows)

        writer.flush()
        elapsed = time.perf_counter() - start
        stats = {
            "total_processed": total_processed,
            "total_errors": total_errors,
            "elapsed_seconds": round(elapsed, 2),
            "throughput_per_second": round(total_processed / max(elapsed, 0.001), 1),
        }
        print(f"\nPipeline complete: {stats}")
        return stats
