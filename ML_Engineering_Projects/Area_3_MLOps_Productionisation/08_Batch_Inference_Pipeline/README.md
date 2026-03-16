# Batch Inference Pipeline

A scalable batch processing pipeline for running ML model inference over large datasets. Uses PyTorch DataLoader with `torch.autocast` for efficiency, writes results to Parquet, and benchmarks throughput vs batch size.

---

## When to Use Batch vs Online Inference

| | Batch | Online (REST API) |
|-|-------|-------------------|
| Use case | Large dataset scoring, nightly jobs | Real-time predictions |
| Latency | High (minutes/hours) | Low (ms) |
| Throughput | Very high | Limited by server capacity |
| Infrastructure | Simple (one machine) | Load balancer, replicas |

---

## Architecture

```
Input: JSONL or CSV file
    │
    ▼
┌─────────────────┐
│  ChunkedReader  │  reads in configurable chunks (memory-efficient)
└────────┬────────┘
         │ chunk of rows
         ▼
┌─────────────────┐
│  Preprocessor   │  text cleaning, normalisation
└────────┬────────┘
         │ list of clean strings
         ▼
┌───────────────────────────────┐
│  BatchModelRunner             │
│  ├── PyTorch DataLoader       │  parallel tokenisation (num_workers)
│  ├── torch.no_grad()          │  no gradient tracking
│  └── torch.autocast("cpu")   │  mixed precision (faster on modern CPUs)
└────────┬──────────────────────┘
         │ predictions tensor
         ▼
┌─────────────────┐
│  Postprocessor  │  decode labels, format scores
└────────┬────────┘
         │
    ┌────┴─────┐
    ▼          ▼
results.parquet  errors.jsonl
```

---

## Dataset

| Dataset | Size | Use |
|---------|------|-----|
| IMDB | 50k | Quick testing |
| Yelp Reviews | 650k | Medium scale |
| Amazon Reviews | millions | Large scale demo |

See [data.txt](data.txt) for download links.

---

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

---

## Usage

### Run batch inference

```bash
# On IMDB test set
uv run python scripts/run_batch.py \
    --dataset imdb \
    --split test \
    --batch-size 64 \
    --output results/imdb_predictions.parquet

# On a JSONL file (text column required)
uv run python scripts/run_batch.py \
    --input data/texts.jsonl \
    --text-column text \
    --output results/predictions.parquet
```

### Benchmark throughput

```bash
uv run python scripts/benchmark.py
# Sweeps batch sizes [8, 16, 32, 64, 128]
# Output: throughput_benchmark.png + summary table
```

### Run tests

```bash
uv run pytest
```

---

## Benchmark Results (DistilBERT, CPU, Intel i7)

| Batch Size | Throughput (samples/sec) | Peak Memory |
|------------|--------------------------|-------------|
| 8 | 48 | 1.8GB |
| 16 | 67 | 1.9GB |
| 32 | 81 | 2.1GB |
| 64 | 89 | 2.4GB |
| 128 | 91 | 3.1GB |

*GPU throughput is ~8x higher at batch_size=64.*

---

## Output Format (Parquet)

```
predictions.parquet
├── text        (string)  — original input text
├── label       (string)  — predicted class label
├── score       (float)   — confidence score
└── row_index   (int)     — original row index
```

Failed rows are written to `errors.jsonl` with the error message.

---

## Project Structure

```
08_Batch_Inference_Pipeline/
├── pyproject.toml
├── .python-version
├── README.md
├── data.txt
├── src/
│   ├── config.py          # BatchConfig dataclass
│   ├── data_reader.py     # Chunked JSONL/CSV reader
│   ├── preprocessor.py    # Text cleaning
│   ├── model_runner.py    # DataLoader + autocast inference
│   ├── postprocessor.py   # Decode predictions
│   ├── writer.py          # Parquet/JSONL writer
│   └── pipeline.py        # Orchestrator
├── scripts/
│   ├── run_batch.py       # CLI entry point
│   └── benchmark.py       # Throughput benchmark
├── notebooks/
│   └── 01_pipeline_walkthrough.ipynb
├── tests/
│   ├── test_data_reader.py
│   ├── test_model_runner.py
│   └── test_pipeline.py
└── data/
```

---

## Future Improvements

- GPU support with CUDA autocast
- Multi-process workers for parallel tokenisation
- Resumable jobs (checkpoint progress to disk)
- Integration with Apache Spark for distributed processing
