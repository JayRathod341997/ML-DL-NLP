"""Benchmark throughput vs batch size.

Usage:
    uv run python scripts/benchmark.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import psutil
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

from src.config import BatchConfig
from src.model_runner import BatchModelRunner

console = Console()

BATCH_SIZES = [8, 16, 32, 64, 128]
NUM_SAMPLES = 512
SAMPLE_TEXT = "This is a sample review for benchmarking the inference speed of the model."


def benchmark_batch_size(batch_size: int) -> dict:
    config = BatchConfig(batch_size=batch_size)
    runner = BatchModelRunner(config)
    texts = [SAMPLE_TEXT] * NUM_SAMPLES

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024**2

    start = time.perf_counter()
    runner.predict_batch(texts)
    elapsed = time.perf_counter() - start

    mem_after = process.memory_info().rss / 1024**2
    return {
        "batch_size": batch_size,
        "throughput": round(NUM_SAMPLES / elapsed, 1),
        "elapsed_s": round(elapsed, 2),
        "memory_mb": round(mem_after - mem_before, 1),
    }


def main() -> None:
    console.print(f"[bold]Benchmarking batch sizes on {NUM_SAMPLES} samples...[/bold]\n")
    results = []
    for bs in BATCH_SIZES:
        console.print(f"  batch_size={bs}...")
        r = benchmark_batch_size(bs)
        results.append(r)
        console.print(f"    {r['throughput']:.1f} samples/sec, {r['memory_mb']:.1f}MB peak")

    table = Table(title="Batch Size Benchmark")
    table.add_column("Batch Size")
    table.add_column("Throughput (samples/sec)", style="cyan")
    table.add_column("Memory Delta (MB)")
    for r in results:
        table.add_row(str(r["batch_size"]), str(r["throughput"]), str(r["memory_mb"]))
    console.print(table)

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot([r["batch_size"] for r in results], [r["throughput"] for r in results], "o-")
    plt.xlabel("Batch Size")
    plt.ylabel("Throughput (samples/sec)")
    plt.title("Inference Throughput vs Batch Size")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("throughput_benchmark.png", dpi=150)
    console.print("\n[bold green]Plot saved to throughput_benchmark.png[/bold green]")


if __name__ == "__main__":
    main()
