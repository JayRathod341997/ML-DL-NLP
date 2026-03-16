# Batch Inference Pipeline — Interview Preparation Guide

> Project stack: transformers · torch · pandas · pyarrow · psutil · click
> Difficulty labels: ⭐ Mid-level  ⭐⭐ Senior-level

---

## Quick Reference Card

```
KEY FILES
─────────────────────────────────────────────────────────────
src/data_reader.py     ChunkedReader: JSONL/CSV + HuggingFace datasets in chunks
src/model_runner.py    BatchModelRunner: TextDataset, DataLoader, torch.autocast
src/pipeline.py        BatchPipeline: orchestrator, returns stats dict
src/writer.py          ResultWriter: Parquet/JSONL + errors.jsonl for failed records
scripts/run_batch.py   Click CLI entry point
scripts/benchmark.py   Batch size sweep [8,16,32,64,128], throughput curve, psutil

KEY NUMBERS
─────────────────────────────────────────────────────────────
Recommended batch sizes          8, 16, 32, 64, 128
GPU memory target utilization    80–90% of VRAM
Throughput plateau               typically at batch_size 32–64 for transformers
FP16 memory savings              ~2× vs FP32
FP16 throughput gain             2–8× on Tensor Cores (A100, V100)
DataLoader num_workers           max(1, os.cpu_count() - 1)
Parquet row group size           ~100MB (default snappy compression)
psutil memory check interval     every 100 batches
Error threshold for abort        > 10% record failure rate
Chunk size formula               memory_budget_GB × 1e9 / avg_record_bytes
Warm-up batches (benchmarking)   3–5 batches before measurement starts
```

---

## 1. Core Concepts & Theory

### 1.1 Batch Inference Architecture

**Q1. ⭐ What is batch inference and how does it differ from real-time (online) inference?**

Batch inference processes a large collection of inputs offline, typically reading from files or databases, running inference, and writing results to storage — not serving users directly. Real-time (online) inference responds to individual requests synchronously within a latency SLA (e.g., < 200ms). Batch inference optimizes for throughput (records per second) and cost efficiency; real-time optimizes for latency and availability. Batch can run on cheaper spot/preemptible instances (since restarts are tolerable), can use larger batch sizes (no per-request latency constraint), and can process data in sorted order for better cache utilization. The trade-off is that results are not immediately available to users.

**Q2. ⭐ When should you choose batch inference over real-time inference?**

Choose batch when: (1) data is available in bulk before results are needed (nightly scoring, ETL pipelines), (2) the SLA allows hours of latency (e.g., "recommendations refreshed daily"), (3) cost is a primary constraint (batch on spot instances is 60–90% cheaper), (4) the model is too large or slow for real-time latency budgets even with optimization, (5) inputs are dependent (score user A only after all of user A's interactions for the day are collected). Choose real-time when: results are needed immediately (fraud detection, content moderation at submission time), inputs arrive one at a time, or personalization requires latest user state.

**Q3. ⭐⭐ Describe the BatchPipeline orchestrator design. What should the stats dict contain and why?**

The `BatchPipeline` orchestrates: `ChunkedReader → BatchModelRunner → ResultWriter`, handling failures gracefully. The stats dict captures:
```python
{
  "total_records": 1_000_000,
  "processed_records": 998_543,
  "error_records": 1_457,
  "error_rate": 0.00146,
  "throughput_records_per_sec": 2_341.5,
  "p50_batch_latency_ms": 87.3,
  "p95_batch_latency_ms": 142.1,
  "p99_batch_latency_ms": 891.2,    # outlier batches with long inputs
  "total_wall_time_s": 426.8,
  "gpu_utilization_pct": 84.2,
  "peak_memory_gb": 11.3,
  "output_file_size_gb": 2.14,
  "model_version": "bert-base-v2.1"
}
```
Stats drive: (1) cost attribution (throughput × cloud cost per hour), (2) SLA validation, (3) error rate alerting (abort if > 10%), (4) capacity planning for next batch run, (5) benchmark comparisons across model versions.

  ↳ Follow-up: "How would you implement checkpoint-based resumption in the BatchPipeline?"

  Write a checkpoint file after each successfully processed chunk: `{"last_chunk_id": 47, "records_processed": 47_000, "output_files": [...]}`. On restart, read the checkpoint, skip already-processed chunks (ChunkedReader seeks to offset `last_chunk_id * chunk_size`), and continue from the last checkpoint. Use atomic checkpoint writes (write to temp file, then rename) to prevent corrupted checkpoints from partial writes.

  ↳ Follow-up: "What is idempotency and how do you achieve it in a batch job?"

  An idempotent operation produces the same result whether run once or multiple times. For batch inference: write outputs to partitioned paths with a job_id prefix (`output/job_20240315_abc123/part_001.parquet`). If the job re-runs with the same job_id, it either skips existing partitions (if checkpointing) or overwrites them (producing identical results since the model and inputs are the same). Never append to existing output files — use overwrite semantics with job-scoped output paths.

  ↳ Follow-up: "How do you handle the case where 10% of records are causing inference errors?"

  Implement a circuit-breaker threshold: if error rate exceeds 10%, abort the job and alert — likely indicates a systematic data quality issue (wrong encoding, schema change upstream). For individual record errors: catch exceptions in the batch loop, write to `errors.jsonl` with record_id, error type, and original text, continue processing remaining records. Do NOT fail the entire job for individual record errors unless the error rate threshold is exceeded.

**Q4. ⭐ How does the `ChunkedReader` avoid loading the entire dataset into RAM?**

`ChunkedReader` reads the input file in fixed-size chunks using Python generators, never materializing the full dataset in memory. For CSV: `pandas.read_csv(filepath, chunksize=10_000)` returns a generator of DataFrames. For JSONL: read line-by-line with a buffer accumulator, yielding when buffer reaches `chunk_size` lines. For HuggingFace datasets: use `dataset.select(range(start, end))` or the streaming API (`load_dataset(..., streaming=True)`) with `take()` and `skip()`. Chunk size should be chosen as `min(available_memory_bytes × 0.3 / avg_record_bytes, 50_000)` to use ~30% of RAM per chunk, leaving headroom for model tensors.

### 1.2 PyTorch DataLoader Internals

**Q5. ⭐ Explain the PyTorch DataLoader parameters relevant to batch inference performance.**

`num_workers`: number of worker processes that load data in parallel with model inference. Set to `cpu_count - 1` (leave 1 CPU for main process and GPU orchestration). Each worker pickles batches and sends to main process via shared memory. `pin_memory=True`: allocates tensors in page-locked memory, enabling asynchronous DMA to GPU — eliminates CPU stall during GPU transfer. `prefetch_factor=2` (default): each worker prefetches 2 batches ahead, hiding data loading latency. `persistent_workers=True`: keeps worker processes alive between epochs/iterations (avoids process creation overhead per iteration, ~200ms saved per epoch). `drop_last=False`: keep the final partial batch (important for batch inference where every record matters, unlike training).

**Q6. ⭐⭐ How does `pin_memory=True` improve GPU transfer performance? What are its downsides?**

Page-locked (pinned) memory cannot be swapped to disk by the OS, so the GPU's DMA engine can directly read from it without the CPU staging the transfer through virtual memory. This enables asynchronous host-to-device transfers: the CUDA stream can execute GPU kernels and DMA transfers concurrently. Speedup: 2–4× faster CPU→GPU transfer for large tensors. Downside: pinned memory is a scarce, non-swappable resource. Using too much can cause OOM at the OS level, degrading overall system performance. Do not use `pin_memory=True` if: running on CPU only, using shared GPU (Kubernetes GPU sharing), or available RAM is < 16GB with a large batch size.

**Q7. ⭐ What is the `TextDataset` class and why is it needed for HuggingFace tokenization?**

`TextDataset` is a `torch.utils.data.Dataset` subclass that wraps a list/chunk of text strings and applies tokenization lazily (in `__getitem__`). This allows the DataLoader to apply tokenization in parallel worker processes rather than sequentially before loading. Each `__getitem__` call applies `tokenizer(text, max_length=512, truncation=True, padding='max_length', return_tensors='pt')`, returns the token dict. The `collate_fn` then stacks individual tokenized dicts into a batch. Lazy tokenization avoids: (1) tokenizing the entire dataset before inference (memory-prohibitive), (2) bottlenecking on a single tokenizer thread.

  ↳ Follow-up: "What is the `collate_fn` and when do you need a custom one?"

  The `collate_fn` takes a list of items from `__getitem__` and combines them into a batch. The default stacks tensors of equal size. A custom `collate_fn` is needed when: inputs have variable length (pad to the longest sequence in the batch rather than global max), items contain non-tensor data (strings, metadata dicts), or you want dynamic padding. Dynamic padding (`padding='longest'` per batch) is more efficient than global max-length padding — a batch of short texts uses much less memory and is much faster than a batch padded to 512 tokens.

  ↳ Follow-up: "What is `persistent_workers=True` and when should you disable it?"

  `persistent_workers=True` keeps worker processes alive after a DataLoader iteration is complete, avoiding the `mp.Process` creation overhead (100–400ms per epoch). Disable when: running in a Jupyter notebook (worker processes can cause issues with interactive environments), the dataset changes between iterations (workers cache the dataset state from `__init__`), or the system has tight process count limits.

  ↳ Follow-up: "How does the `pin_memory` + `non_blocking=True` combination work?"

  `pin_memory=True` on the DataLoader allocates pinned host memory. `tensor.cuda(non_blocking=True)` then initiates an asynchronous DMA transfer — the call returns immediately without waiting for the transfer to complete. The GPU kernel that uses the tensor will synchronize automatically when it needs the data. This allows CPU preprocessing and GPU compute to overlap: while GPU is running forward pass on batch N, CPU is transferring batch N+1.

### 1.3 Mixed Precision (torch.autocast / AMP)

**Q8. ⭐ What is `torch.autocast` and how does it improve batch inference performance?**

`torch.autocast` is a context manager that automatically casts certain operations to lower precision (FP16 or BF16) while keeping others in FP32 for numerical stability. Usage:
```python
with torch.autocast(device_type='cuda', dtype=torch.float16):
    outputs = model(**inputs)
```
Benefits: (1) FP16 uses 2× less VRAM than FP32, allowing 2× larger batch sizes, (2) NVIDIA Tensor Cores execute FP16 matrix multiplications 2–8× faster than FP32, (3) memory bandwidth is halved (larger models become bandwidth-bound, not compute-bound). The `autocast` context automatically selects which ops run in FP16 (matmul, conv) vs FP32 (softmax, layer norm, loss computation).

**Q9. ⭐⭐ When should you NOT use FP16 in batch inference? What causes numerical instability?**

Avoid FP16 when: (1) **Softmax with large logits**: FP16 range is ±65504; logits > 65504 produce inf, making softmax output NaN. Common in large vocabulary models with extreme logits. (2) **Log operations**: `log(0)` in FP16 can behave differently — use FP32 for log-probability computations. (3) **Accumulation in attention**: long sequences accumulate many small values; FP16 precision loss compounds over 512+ tokens, especially in models not trained with AMP. (4) **Quantization-unaware models**: models fine-tuned in FP32 without mixed-precision training can show significant accuracy degradation in FP16 inference. Use BF16 instead of FP16 on A100/H100 GPUs — BF16 has the same exponent range as FP32 (no overflow risk) with less precision, making it more stable for inference.

  ↳ Follow-up: "What is the difference between FP16, BF16, and INT8 for inference?"

  FP16 (half precision): 1 sign bit, 5 exponent bits, 10 mantissa bits. Range ±65504, good for Tensor Cores. BF16 (bfloat16): 1 sign bit, 8 exponent bits, 7 mantissa bits. Same range as FP32 (±3.4×10^38), less prone to overflow. Preferred on A100/H100. INT8 (8-bit integer): 2–4× compression, 2–4× throughput improvement, but requires quantization-aware training or post-training quantization calibration. Accuracy impact varies (typically < 0.5% for NLP classification, larger for generation).

  ↳ Follow-up: "How does `torch.autocast` decide which operations to run in FP16 vs FP32?"

  `torch.autocast` uses an op-level allow-list maintained in PyTorch. Ops that benefit from FP16 and are numerically stable (matrix multiplication, convolution, linear): FP16. Ops that are numerically sensitive (softmax, layer norm, batch norm, loss functions, reduction ops): FP32. The lists are defined in `torch.amp.autocast_mode._get_autocast_dtype` and `torch._C._jit_get_operation`. You can override with `torch.amp.autocast_mode.custom_fwd` and `custom_bwd` decorators.

  ↳ Follow-up: "What is gradient scaling and does it apply to inference?"

  Gradient scaling (`torch.cuda.amp.GradScaler`) multiplies the loss by a large scale factor before backward pass to prevent FP16 gradients from underflowing to zero. It applies to TRAINING only, not inference. During inference, there is no backward pass and no GradScaler needed — only `torch.autocast` is required.

**Q10. ⭐ What is `torch.no_grad()` and why is it critical for batch inference?**

`torch.no_grad()` disables gradient tracking for all operations within the context. During inference, gradients are never needed — but by default, PyTorch tracks operations for autograd. `torch.no_grad()` provides: (1) ~50% memory reduction (no activation storage for backward pass), (2) slight speed improvement (no autograd graph construction overhead). Always use `with torch.no_grad():` in the model forward pass during inference. Combine with `torch.autocast` for maximum performance:
```python
with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
    outputs = model(**inputs)
```

### 1.4 Throughput vs Latency Trade-off

**Q11. ⭐ How does batch size affect throughput and latency in batch inference?**

```
Throughput-Latency-Memory Trade-off:
──────────────────────────────────────────────────────────
Batch   Latency    Throughput    GPU Mem    GPU Util
Size    (ms/batch) (rec/sec)    (GB)       (%)
──────────────────────────────────────────────────────────
  8       23         347          2.1        41%     ← underutilized
 16       28         571          3.4        58%
 32       41         780          5.9        74%
 64       72         889          10.8       88%     ← sweet spot
128      OOM         ---          22.4+     OOM!
──────────────────────────────────────────────────────────
```

Larger batches better utilize GPU's parallel cores (higher utilization) up to the saturation point. Beyond that, throughput plateaus while latency increases linearly. For batch inference, choose the largest batch size that fits in GPU memory with 10–20% headroom, since latency per batch is not user-facing. The throughput curve typically shows a "knee" where throughput stops increasing sharply — that is the optimal batch size.

**Q12. ⭐⭐ Explain the benchmark methodology used in `scripts/benchmark.py`. What statistical rigor is required?**

```python
import psutil, time, statistics

def benchmark_batch_size(model, dataloader, batch_size, n_warmup=3, n_trials=10):
    # Warm-up: GPU JIT compilation, cache warming
    for batch in itertools.islice(dataloader, n_warmup):
        with torch.no_grad():
            _ = model(**batch)
    torch.cuda.synchronize()  # wait for all CUDA ops

    latencies = []
    for batch in itertools.islice(dataloader, n_trials):
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(**batch)
        torch.cuda.synchronize()  # critical: sync before timing end
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        "batch_size": batch_size,
        "mean_ms": statistics.mean(latencies),
        "p50_ms": statistics.median(latencies),
        "p99_ms": statistics.quantiles(latencies, n=100)[98],
        "throughput": batch_size / (statistics.mean(latencies) / 1000),
        "gpu_util": get_gpu_utilization(),
        "memory_gb": torch.cuda.memory_allocated() / 1e9,
    }
```

`torch.cuda.synchronize()` is critical: CUDA operations are asynchronous; without sync, `time.perf_counter()` captures kernel submission time, not completion time, giving misleadingly low latency measurements.

  ↳ Follow-up: "Why do you need warm-up runs before benchmarking?"

  First runs include: CUDA JIT kernel compilation (100ms–2s for the first matmul of a given shape), model weight page-faulting from disk to RAM/VRAM, Python interpreter startup effects (module caching, JIT). Warm-up runs ensure you measure steady-state performance, not startup overhead. For transformers with FlashAttention kernels, warm-up may take 5–10 batches before latency stabilizes.

  ↳ Follow-up: "How do you control for variability in GPU benchmarks?"

  Run in a dedicated GPU environment (no other GPU users). Set `torch.backends.cudnn.benchmark = False` (otherwise cuDNN searches for the fastest algorithm on first run, skewing warm-up latency). Use `torch.cuda.reset_peak_memory_stats()` between runs. Run 10+ trials and report p50 and p99 — variance in GPU benchmarks is typically ±5–15% due to thermal throttling, CUDA context scheduling, and background system processes.

**Q13. ⭐ What is GPU memory utilization vs GPU compute utilization? How do you measure each?**

**GPU memory utilization**: fraction of VRAM used = `torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory`. Target 80–90% for batch inference (leave headroom for activation memory and CUDA workspace). **GPU compute utilization**: fraction of time the GPU SM (streaming multiprocessor) cores are active — measured by `nvidia-smi --query-gpu=utilization.gpu` or `pynvml`. A high memory utilization + low compute utilization indicates memory bandwidth bottleneck (model is too large relative to compute intensity). Low memory + low compute indicates CPU bottleneck (DataLoader not feeding fast enough — increase `num_workers`).

### 1.5 Chunked Reading & Memory Management

**Q14. ⭐ Explain the memory lifecycle of a record from reading to writing in the BatchPipeline.**

```
Record Memory Lifecycle:
────────────────────────────────────────────────────────────────────
1. Read (CPU RAM)
   ChunkedReader yields chunk of N records as strings
   Memory: N × avg_record_size (e.g., 1000 × 500B = 500KB)

2. Tokenize (CPU RAM)
   TextDataset.__getitem__ tokenizes each text
   Memory: N × max_seq_len × 2 (int32 tokens) (e.g., 1000×512×4 = 2MB)

3. DataLoader collation (CPU pinned memory)
   pin_memory=True: batch tensors in page-locked memory
   Memory: batch_size × seq_len × 4B × 3 tensors = ~6MB per batch

4. GPU Transfer (VRAM)
   Async DMA: batch tensors on GPU
   Memory: ~6MB per batch on VRAM

5. Forward pass (VRAM)
   Activations, attention matrices: seq_len^2 × hidden_dim
   Memory: ~200MB–2GB per batch (dominant cost)

6. Output (CPU RAM ← VRAM)
   .cpu().numpy(): move logits back to CPU
   Memory: batch_size × num_classes × 4B = ~64KB

7. Write (Disk)
   ResultWriter appends to Parquet buffer, flushes every 10k records
   Free: input strings, tokenized tensors, GPU memory (del + gc.collect())
────────────────────────────────────────────────────────────────────
Peak concurrent memory: steps 3+4+5 overlap during DataLoader prefetch
```

**Q15. ⭐⭐ How do you handle OOM (Out of Memory) errors during batch inference with automatic batch size reduction?**

```python
def run_with_oom_retry(model, batch, initial_batch_size):
    current_size = initial_batch_size
    while current_size >= 1:
        try:
            split_batches = split_into_batches(batch, current_size)
            results = []
            for sub_batch in split_batches:
                with torch.no_grad(), torch.autocast('cuda', torch.float16):
                    out = model(**sub_batch)
                results.append(out.cpu())
            return torch.cat(results)
        except torch.cuda.OutOfMemoryError:
            current_size //= 2
            torch.cuda.empty_cache()
            gc.collect()
            logger.warning(f"OOM, retrying with batch_size={current_size}")
    raise RuntimeError("OOM even at batch_size=1")
```

After catching `torch.cuda.OutOfMemoryError`, always call `torch.cuda.empty_cache()` to release cached-but-unreferenced VRAM before retrying. Log the OOM event with the input sizes — long sequences often trigger OOM on otherwise-working batch sizes. Alert if OOM frequency exceeds 1% of batches (indicates suboptimal batch size selection).

  ↳ Follow-up: "What causes GPU memory to grow over a long batch inference run (memory leak)?"

  Common causes: (1) Output tensors accumulated in a Python list without `.detach()` — maintains references to the computation graph. (2) `model.eval()` not called — dropout and batch norm in training mode keep extra state. (3) CUDA streams not synchronized — GPU buffers not freed until sync. (4) HuggingFace `model.generate()` with `use_cache=True` — KV cache accumulates across calls. Fix: `del outputs; torch.cuda.empty_cache()` after each batch; use `model.eval()` and `torch.no_grad()`.

  ↳ Follow-up: "How does psutil help diagnose performance bottlenecks in batch inference?"

  ```python
  import psutil, os
  proc = psutil.Process(os.getpid())

  # CPU bottleneck: if cpu_percent approaches 100% during DataLoader loading
  cpu_pct = proc.cpu_percent(interval=1.0)

  # I/O bottleneck: if io_wait is high, DataLoader reading too slow
  io = proc.io_counters()  # read_bytes, write_bytes

  # Memory leak detection: if rss grows over time
  rss_gb = proc.memory_info().rss / 1e9

  # System-level: check if we are bottlenecked by another process
  system_cpu = psutil.cpu_percent(percpu=True)
  ```
  Pattern: CPU-bound on DataLoader workers → increase `num_workers`. Memory growing → memory leak in output accumulation. IO wait high → input data on spinning disk (move to NVMe or RAM disk).

### 1.6 Output Formats: Parquet vs JSONL

**Q16. ⭐ Compare Parquet and JSONL as output formats for batch inference results. When do you choose each?**

| Dimension          | Parquet                                   | JSONL                                  |
|--------------------|-------------------------------------------|----------------------------------------|
| Storage efficiency | 5–10× smaller (columnar + compression)   | Verbose (keys repeated every row)     |
| Read performance   | 100× faster for column scans             | Must parse every field to filter      |
| Streaming write    | Requires row group buffering (10k+ rows) | Fully streaming (append one line)     |
| Schema enforcement | Enforced by PyArrow schema                | None (each line is independent)       |
| Human readability  | Binary (not human-readable)               | Human-readable with `cat`             |
| Downstream tools   | Spark, Athena, BigQuery native            | Most tools, easy to `grep`            |
| Error records      | Poor fit (partial schema)                 | Ideal (flexible schema for errors)    |
| Append to existing | Complex (rewrite row group)               | Trivial (`>>` append)                 |

Use Parquet for main results (consumed by analytics, BI tools, feature stores). Use JSONL for error records (`errors.jsonl`) — each error may have a different schema (different error types), and errors are read by humans debugging issues.

**Q17. ⭐ What Parquet compression codec should you use and why?**

| Codec   | Compression ratio | Speed (compress) | Speed (decompress) | Recommended for       |
|---------|-------------------|------------------|---------------------|-----------------------|
| Snappy  | 2–3×             | Very fast        | Very fast           | Production (default)  |
| LZ4     | 2–3×             | Fastest          | Fastest             | Low-latency reads     |
| ZSTD    | 4–6×             | Moderate         | Fast                | Cold storage, archival|
| Gzip    | 5–7×             | Slow             | Slow                | Compatibility only    |
| None    | 1×               | N/A              | Fastest             | Pre-sorted data       |

Use `snappy` as the default — it achieves good compression with minimal CPU overhead during write. Use `zstd` when storage cost is critical (10M+ record outputs). Row group size: default 128MB per row group; smaller (64MB) for wide-column analytics access patterns.

**Q18. ⭐⭐ What is schema evolution in Parquet and why does it matter for long-running batch pipelines?**

Schema evolution is the ability to read Parquet files written by older pipeline versions even after the schema changes (new columns added, types changed). PyArrow supports: (1) **Column addition**: old files simply return `null` for new columns when merged. (2) **Column removal**: new reader ignores columns not in new schema (projective evolution). (3) **Type widening**: int32 → int64 is safe; int64 → int32 is unsafe (data loss). Problems arise with: column rename (treated as drop + add, all existing data lost), type narrowing, and changing column order without name-based resolution. For long-running pipelines: always add columns (never rename/remove), use `schema=` parameter in `pq.write_table()` to enforce the canonical schema, and validate output schema in CI/CD.

### 1.7 Error Handling & Dead Letter Queues

**Q19. ⭐ Explain the `errors.jsonl` pattern. What fields should each error record contain?**

```json
{
  "record_id": "row_000045892",
  "timestamp_utc": "2024-03-15T14:23:07.123456Z",
  "error_type": "TokenizerOverflowError",
  "error_message": "Sequence length 8192 exceeds max_length 512",
  "stack_trace": "Traceback (most recent call last)...",
  "original_text": "...",
  "original_text_length": 8192,
  "source_file": "s3://data/input/part_003.jsonl",
  "source_line": 45892,
  "pipeline_run_id": "batch_20240315_abc123",
  "model_version": "bert-base-v2.1"
}
```

The `original_text` field enables: (1) manual inspection of problematic inputs, (2) reprocessing with different parameters (e.g., truncation strategy), (3) upstream data quality improvement. The `error_type` enables categorized monitoring: tokenization errors (text too long) vs model errors (NaN output) vs infrastructure errors (GPU OOM). Never truncate the `stack_trace` — it is essential for debugging non-obvious errors.

**Q20. ⭐⭐ Design a dead letter queue (DLQ) architecture for batch inference at scale.**

```
Input Records (S3/Blob)
    │
    ▼
BatchPipeline
    ├── Success Path ──────────────────────→ Output Parquet (S3)
    │
    ├── Retry Path (transient errors):
    │   Error → retry queue (in-memory deque)
    │         → retry 3× with exponential backoff
    │         → if still failing → DLQ
    │
    └── DLQ Path (persistent errors):
        Error → errors.jsonl (immediate write)
              → SQS Dead Letter Queue (for distributed tracking)
                └── CloudWatch Alarm: DLQ depth > 1000
                      → SNS notification
                      → Manual review / re-ingestion workflow

Error Categories:
  TRANSIENT (retry):   GPU OOM → retry at smaller batch
                       Network timeout → retry with backoff
  PERMANENT (DLQ):     Tokenizer overflow (text too long)
                       Invalid encoding (corrupt record)
                       Schema mismatch (wrong field types)
```

---

## 2. System Design Discussions

**Q21. ⭐⭐ Design a scalable batch inference system that processes 100 million records daily with a 6-hour completion SLA.**

```
CAPACITY PLANNING:
──────────────────
100M records / 6h = 4,630 records/sec required throughput

Assume BERT-base inference at batch_size=64:
  - 64 records per batch
  - 72ms per batch on 1× A100
  - Throughput per GPU: 64/0.072 = 889 records/sec
  - GPUs needed: 4,630 / 889 = 5.2 → use 6× A100 (with headroom)

ARCHITECTURE:
─────────────────────────────────────────────────────────────
Input: 100M records split into 1000 shards (100k records each)
    │
    ▼
Job Coordinator (Step Functions / Azure Data Factory)
    ├── Distributes 1000 shards to 6 parallel workers
    ├── Tracks shard completion status (DynamoDB / Azure Table)
    └── Retries failed shards (up to 3×)
         │
         ▼ (each worker independently)
┌─────────────────────────────────┐
│ Batch Worker (Fargate/ACI)      │
│  ChunkedReader → chunk_size=5k  │
│  BatchModelRunner (A100 GPU)    │
│  batch_size=64, FP16 autocast   │
│  ResultWriter → Parquet shards  │
└─────────────────────────────────┘
         │
         ▼
Output: 1000 Parquet shards → merge → final output table
    │
    ▼
Downstream: Spark/Athena query on merged Parquet
```

**Q22. ⭐⭐ Compare pipeline parallelism and data parallelism for batch inference. Which does your project use?**

**Data parallelism**: multiple GPU devices each process different batches simultaneously using the same model. `torch.nn.DataParallel` (single machine, multi-GPU) or `torch.nn.parallel.DistributedDataParallel` (multi-machine). The project uses data parallelism implicitly — each batch worker runs on one GPU. Scaling: add more batch worker containers, each with one GPU.

**Pipeline parallelism**: the model is split across multiple GPUs by layer. GPU 1 runs layers 1-12, GPU 2 runs layers 13-24 (for BERT-large). As GPU 1 finishes a microbatch and passes to GPU 2, GPU 1 starts the next microbatch. Used for models that don't fit in single GPU memory (e.g., GPT-3 175B). Implemented via `torch.distributed.pipeline.sync.Pipe`. The project does NOT need pipeline parallelism — BERT-base fits in 2GB VRAM.

  ↳ Follow-up: "When would you use model sharding (tensor parallelism) instead?"

  Tensor parallelism splits individual weight matrices across GPUs (e.g., each GPU holds 1/N of the attention weight matrix, computes partial output, all-reduce to combine). Used when a single weight matrix is too large for one GPU's VRAM (e.g., LLaMA-65B with 4096×4096×128 weight matrices). Implemented via Megatron-LM, DeepSpeed, or HuggingFace Accelerate. For inference at scale: use NVIDIA Triton Inference Server with multi-GPU model parallelism.

**Q23. ⭐ How would you design a batch inference system to process data incrementally (only new records since last run)?**

```
Incremental Processing Architecture:
─────────────────────────────────────
Watermark table (DynamoDB/Azure Table):
  { "pipeline_name": "sentiment_batch",
    "last_processed_timestamp": "2024-03-14T23:59:59Z",
    "last_processed_offset": 5_000_000 }

Pipeline Logic:
  1. Read watermark: last_ts = get_watermark("sentiment_batch")
  2. Query input: SELECT * FROM events WHERE ts > last_ts
     (or: read Parquet partitions with date > last_date)
  3. Process new records
  4. Write outputs to new partition: output/date=2024-03-15/
  5. Update watermark: set last_ts = max(processed_timestamps)
     (atomic write — update only after successful output write)

Failure safety:
  - If job fails after processing but before watermark update:
    Next run re-processes some records → need idempotent output
    (deduplicate by record_id in downstream analytics)
```

---

## 3. Coding & Implementation Questions

**Q24. ⭐ Write a Click CLI for the batch pipeline with configurable input/output paths and batch size.**

```python
# scripts/run_batch.py
import click
import logging
from pathlib import Path
from src.pipeline import BatchPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

@click.command()
@click.option('--input-path', required=True, type=click.Path(exists=True),
              help='Path to input JSONL or CSV file')
@click.option('--output-dir', required=True, type=click.Path(),
              help='Directory for output Parquet files')
@click.option('--batch-size', default=32, show_default=True,
              type=click.IntRange(1, 512),
              help='Inference batch size')
@click.option('--chunk-size', default=10_000, show_default=True,
              help='Number of records per reading chunk')
@click.option('--model-path', required=True, envvar='MODEL_PATH',
              help='Path or HuggingFace ID of the model')
@click.option('--device', default='cuda', show_default=True,
              type=click.Choice(['cuda', 'cpu', 'mps']))
@click.option('--fp16/--no-fp16', default=True, show_default=True,
              help='Enable FP16 mixed precision')
def run_batch(input_path, output_dir, batch_size, chunk_size,
              model_path, device, fp16):
    """Run batch inference pipeline."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pipeline = BatchPipeline(
        model_path=model_path,
        batch_size=batch_size,
        chunk_size=chunk_size,
        device=device,
        use_fp16=fp16,
    )
    stats = pipeline.run(input_path=input_path, output_dir=output_dir)
    logger.info(f"Pipeline complete: {stats}")
    if stats['error_rate'] > 0.10:
        raise SystemExit(f"Error rate {stats['error_rate']:.1%} exceeds 10% threshold")

if __name__ == '__main__':
    run_batch()
```

**Q25. ⭐ Implement `BatchModelRunner.run()` with autocast, no_grad, and error handling.**

```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging, time
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class BatchModelRunner:
    def __init__(self, model_path: str, batch_size: int,
                 device: str = 'cuda', use_fp16: bool = True):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.use_fp16 = use_fp16 and device == 'cuda'
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval().to(self.device)

    def run(self, texts: List[str]) -> Dict[str, Any]:
        dataset = TextDataset(texts, self.tokenizer)
        loader = DataLoader(dataset, batch_size=self.batch_size,
                           num_workers=4, pin_memory=True,
                           persistent_workers=True)
        results, latencies = [], []
        for batch_idx, batch in enumerate(loader):
            batch = {k: v.to(self.device, non_blocking=True)
                     for k, v in batch.items()}
            t0 = time.perf_counter()
            try:
                ctx = torch.autocast('cuda', torch.float16) if self.use_fp16 \
                      else contextlib.nullcontext()
                with torch.no_grad(), ctx:
                    outputs = self.model(**batch)
                    logits = outputs.logits.cpu().float()
                    probs = torch.softmax(logits, dim=-1)
                torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000)
                results.extend(probs.numpy().tolist())
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                logger.error(f"OOM on batch {batch_idx}, skipping")
                results.extend([None] * len(batch['input_ids']))
            finally:
                del batch
        return {'predictions': results, 'latencies_ms': latencies}
```

**Q26. ⭐⭐ Implement a `ResultWriter` that writes Parquet with error handling for partial writes.**

```python
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import json, logging, tempfile, os
from typing import List, Optional

logger = logging.getLogger(__name__)

class ResultWriter:
    SCHEMA = pa.schema([
        pa.field('record_id', pa.string()),
        pa.field('label', pa.string()),
        pa.field('score', pa.float32()),
        pa.field('confidence', pa.float32()),
    ])

    def __init__(self, output_dir: str, partition_size: int = 100_000):
        self.output_dir = Path(output_dir)
        self.partition_size = partition_size
        self.error_file = self.output_dir / 'errors.jsonl'
        self._buffer: List[dict] = []
        self._partition_idx = 0

    def write_batch(self, records: List[dict],
                    errors: Optional[List[dict]] = None):
        self._buffer.extend(records)
        if errors:
            self._write_errors(errors)
        if len(self._buffer) >= self.partition_size:
            self._flush()

    def _flush(self):
        if not self._buffer:
            return
        table = pa.Table.from_pylist(self._buffer, schema=self.SCHEMA)
        # Atomic write: temp file then rename
        out_path = self.output_dir / f'part_{self._partition_idx:05d}.parquet'
        tmp_path = out_path.with_suffix('.parquet.tmp')
        pq.write_table(table, tmp_path, compression='snappy',
                       row_group_size=100_000)
        os.rename(tmp_path, out_path)  # atomic on same filesystem
        logger.info(f"Wrote {len(self._buffer)} records to {out_path}")
        self._buffer.clear()
        self._partition_idx += 1

    def _write_errors(self, errors: List[dict]):
        with open(self.error_file, 'a', encoding='utf-8') as f:
            for error in errors:
                f.write(json.dumps(error, ensure_ascii=False) + '\n')

    def close(self):
        self._flush()  # flush remaining records
```

---

## 4. Common Bugs & Issues

| # | Bug | Root Cause | Symptom | Fix |
|---|-----|------------|---------|-----|
| 1 | GPU underutilization | `num_workers=0` (default), DataLoader bottleneck | GPU util 20–30%, CPU 100% | Set `num_workers=cpu_count()-1`, enable `persistent_workers=True` |
| 2 | OOM at inference | Batch size too large for model + sequence length | `torch.cuda.OutOfMemoryError` | Implement OOM retry with batch halving; start with batch_size=16 and benchmark |
| 3 | Memory growth over job | Outputs accumulated without `.cpu()` + `.detach()` | RSS grows 1GB/hour; eventual OOM | `del outputs; torch.cuda.empty_cache()` after each batch |
| 4 | Inaccurate benchmark timing | Missing `torch.cuda.synchronize()` before timing end | Measured latency 5× lower than actual | Add `torch.cuda.synchronize()` before `time.perf_counter()` end |
| 5 | FP16 NaN outputs | Input logits overflow FP16 range (>65504) | All predictions NaN; model outputs useless | Switch to BF16 or disable FP16 for this model; inspect logit distributions |
| 6 | Parquet corruption on crash | Partial write without atomic rename | Downstream reads fail with `ArrowInvalid` | Atomic write (temp + rename); validate Parquet on write completion |
| 7 | Tokenizer download hangs | HuggingFace model downloads on first run in air-gapped env | Job hangs indefinitely at tokenizer init | Pre-download to local path; set `TRANSFORMERS_OFFLINE=1` env var |
| 8 | Duplicate outputs on rerun | No checkpoint/idempotency; appending to existing files | Downstream analytics sees duplicate records | Use job_id-scoped output paths; overwrite semantics |
| 9 | Wrong device for tensors | `batch.to('cuda')` but model on CPU (or vice versa) | `RuntimeError: Expected all tensors to be on same device` | Always move batch to `self.device`, not hardcoded 'cuda' |
| 10 | `persistent_workers` crash | Jupyter/interactive environment incompatibility | `RuntimeError: DataLoader worker exited unexpectedly` | Set `persistent_workers=False` in notebooks; enable only in batch scripts |
| 11 | Chunk size too large | Available RAM underestimated | `MemoryError` mid-job | Add RAM check at startup; set `chunk_size = int(psutil.virtual_memory().available * 0.3 / avg_record_bytes)` |
| 12 | Output schema mismatch | NumPy float32 vs Python float in different batches | `ArrowTypeError` during Parquet write | Cast all outputs to Python native types; define explicit PyArrow schema |
| 13 | Error rate silently high | Exception caught but not counted; no abort threshold | 50% records wrong, job "succeeds" | Count errors per batch; abort if `errors/total > 0.10` |
| 14 | Benchmark variance high | Thermal throttling on long benchmark runs | p99 latency 3× p50 latency | Run benchmarks in temperature-stable environment; use p50 not mean |
| 15 | `model.train()` at inference | `model.eval()` not called | Dropout active; non-deterministic outputs | Always call `model.eval()` before inference loop |
| 16 | Click path validation fails | `exists=True` on output dir that doesn't exist yet | CLI refuses to run | Use `type=click.Path()` (not `exists=True`) for output dirs; `mkdir` in pipeline init |
| 17 | CUDA device error silent | `CUDA_LAUNCH_BLOCKING=0` (default) hides assertion errors | Wrong outputs, no exception | Set `CUDA_LAUNCH_BLOCKING=1` for debugging; check `cuda-memcheck` |
| 18 | psutil measures wrong process | `psutil.Process()` without pid measures parent, not worker | Memory stats show OS overhead, not pipeline | `psutil.Process(os.getpid())` to get current process |

---

## 5. Deployment — Azure

**Q27. ⭐ How would you run the batch inference pipeline on Azure Batch?**

```
Azure Batch Architecture:
──────────────────────────────────────────────────────────────────
Input Data (Azure Blob Storage: input-container/run_20240315/)
    │
    ▼
Azure Data Factory (trigger: daily at 01:00 UTC)
    │  Pipeline: CheckNewData → SubmitBatchJob → MonitorJob
    ▼
Azure Batch Account
    ├── Pool: "gpu-pool"
    │     VM size: Standard_NC6s_v3 (1× V100 16GB, 6 vCPU, 112GB RAM)
    │     Target dedicated nodes: 6
    │     Node agent: Ubuntu 20.04 + CUDA 11.8
    │     Start task: docker pull myregistry.azurecr.io/batch-inference:latest
    │
    ├── Job: "inference-20240315"
    │     Task per shard (1000 tasks total):
    │       Command: python scripts/run_batch.py
    │                --input-path $AZ_BATCH_NODE_MOUNTS_DIR/input/shard_$TASK_ID.jsonl
    │                --output-dir $AZ_BATCH_NODE_MOUNTS_DIR/output/shard_$TASK_ID/
    │                --batch-size 64 --fp16
    │       Max retry: 3
    │       Timeout: 2h per task
    │
    └── Output: Azure Blob Storage (output-container/run_20240315/)
                Auto-merge via Azure Synapse or Databricks job

Cost optimization:
  - Use Low-priority VMs (80% cheaper): acceptable for batch (retry on preemption)
  - Set max_low_priority_nodes=6, target_dedicated_nodes=0
  - Add dedicated=1 as fallback when low-priority quota exhausted
```

**Q28. ⭐⭐ Design the Azure observability setup for the batch pipeline.**

```
Azure Observability Stack for Batch Pipeline:
─────────────────────────────────────────────────────────────
BatchPipeline (running on Azure Batch nodes)
    │
    ├── Stats dict → Application Insights custom events
    │   ai.track_event("BatchComplete", {
    │     "throughput": 2341.5,
    │     "error_rate": 0.00146,
    │     "records_processed": 998543,
    │     "gpu_utilization": 84.2,
    │     "model_version": "v2.1"
    │   })
    │
    ├── Structured logs (JSON) → Azure Log Analytics
    │   KQL query: detect error spikes
    │   | where error_rate > 0.05
    │   | project timestamp, shard_id, error_count, error_types
    │
    ├── Azure Monitor Alerts:
    │   - Task failure rate > 5% → email + Slack webhook
    │   - Job duration > 7h (SLA breach) → PagerDuty
    │   - Blob output size 0 (no output written) → critical alert
    │
    └── Azure Data Factory Monitoring:
        - Pipeline run history
        - Activity run duration
        - Alert on ADF pipeline failure
```

**Q29. ⭐ How do you use Azure ML Batch Endpoints as an alternative to Azure Batch for ML workloads?**

Azure ML Batch Endpoints are managed infrastructure for batch inference:
1. Register model in Azure ML Model Registry.
2. Create deployment with compute cluster (Standard_NC6s_v3 × 4 nodes).
3. Define `BatchDriver` class with `init()` (model load) and `run(mini_batch)` (inference).
4. Submit job via SDK: `endpoint.invoke(input=Input(path="azureml://datastores/..."))`.
5. Output: automatically collected to default datastore as CSV/Parquet.

Advantages over raw Azure Batch: managed model versioning, built-in monitoring, SDK-native job submission, automatic output collection. Disadvantages: less flexibility (must conform to AML driver interface), higher cost than bare Azure Batch VMs. Use Azure ML Batch Endpoints when you want MLOps integration (model registry, experiment tracking). Use Azure Batch directly for custom pipelines with full infrastructure control.

  ↳ Follow-up: "How do you configure auto-scaling for an Azure ML compute cluster?"

  Set `min_instances=0` (scale to zero when idle, saves cost) and `max_instances=6` in `AmlCompute` configuration. Set `idle_time_before_scale_down=120` seconds (keep warm for 2 minutes after job completion to handle rapidly successive jobs). Scaling from 0 to 6 nodes takes ~3–5 minutes — factor into job SLA. For time-critical batch jobs, set `min_instances=1` to keep one node warm.

---

## 6. Deployment — AWS

**Q30. ⭐⭐ Design the AWS batch inference architecture using AWS Batch and S3.**

```
AWS Batch Inference Architecture:
──────────────────────────────────────────────────────────────────
Input: S3 (s3://company-data/batch-input/run_20240315/)
    │
    ▼
AWS EventBridge (schedule: cron(0 1 * * ? *) = 01:00 UTC daily)
    │
    ▼
AWS Step Functions (orchestrator)
    ├── State 1: Split input into 1000 shards (Lambda function)
    │     → Write shard manifest to DynamoDB
    ├── State 2: Submit AWS Batch Array Job
    │     → Array size: 1000 (one job per shard)
    ├── State 3: Wait for completion (polling with 5-min interval)
    ├── State 4: Validate outputs (Lambda: check all shards written)
    └── State 5: Merge outputs (AWS Glue ETL job)

AWS Batch Configuration:
────────────────────────
Compute Environment:
  Type: MANAGED
  Instance type: p3.2xlarge (1× V100 16GB)
  Allocation strategy: SPOT_CAPACITY_OPTIMIZED (use Spot for cost)
  Min vCPUs: 0, Max vCPUs: 96 (16 × 6 nodes)

Job Queue: inference-queue
  Priority: 10
  Compute environments: [spot-gpu-env, on-demand-gpu-env] (fallback)

Job Definition:
  Image: ECR_URI/batch-inference:v1.2.3
  vCPUs: 8, Memory: 32000 MB, GPUs: 1
  Environment: MODEL_PATH=s3://company-models/bert-v2.1.tar.gz
  Command: ["python", "scripts/run_batch.py",
            "--input-path", "Ref::input_s3_path",
            "--output-dir", "Ref::output_s3_path",
            "--batch-size", "64", "--fp16"]
  Retry attempts: 3
  Job timeout: 7200 seconds (2h)

Array Job:
  Size: 1000 (AWS_BATCH_JOB_ARRAY_INDEX = shard ID)
```

**Q31. ⭐ How does AWS SageMaker Batch Transform compare to AWS Batch for this pipeline?**

| Dimension | AWS Batch | SageMaker Batch Transform |
|-----------|-----------|--------------------------|
| Setup complexity | Medium (compute env, job def) | Low (managed) |
| Container format | Any Docker image | SageMaker inference container spec |
| Input/output | S3 paths (flexible) | S3 → Transform job → S3 |
| GPU instance types | All EC2 GPU types | ml.p3.* / ml.g4dn.* |
| Monitoring | CloudWatch + custom | SageMaker Model Monitor built-in |
| Spot support | Native (SPOT env) | Via `TransformJobConfig` |
| Cost | Lower (raw EC2 pricing) | Higher (~20% SageMaker surcharge) |
| MLOps integration | Manual | Built-in (model registry, lineage) |

Use SageMaker Batch Transform when: deeply integrated with SageMaker (model registry, pipelines). Use AWS Batch when: custom pipeline logic, existing Docker infrastructure, cost optimization is critical.

  ↳ Follow-up: "How does AWS Step Functions enable retry and error handling for batch jobs?"

  Step Functions state machines support: `Retry` configurations per state (retry on specific error types: `States.TaskFailed`, `Batch.AWSBatchException`), `Catch` transitions (on failure → notify SNS + clean up partial outputs), and `Wait` states with polling. Example: retry `SubmitBatchJob` state 3 times with exponential backoff (initial 60s, max 600s) on `States.TaskFailed`. On exhausted retries, transition to `NotifyFailure` state that sends SNS message and marks DynamoDB job record as FAILED.

  ↳ Follow-up: "How do you download large models from S3 efficiently at AWS Batch task startup?"

  Use `aws s3 cp s3://company-models/bert-v2.1.tar.gz /tmp/model.tar.gz` before the inference script. For large models (10GB+): (1) Use a pre-built AMI with the model baked in (fastest startup, higher storage cost), (2) use EFS (Elastic File System) mounted to the compute environment (model shared across nodes, no download per job), (3) use S3 Transfer Acceleration + `s3 sync --exact-timestamps` to only download changed files. EFS approach is recommended for models > 1GB accessed by many parallel jobs.

**Q32. ⭐⭐ How do you monitor batch inference jobs in AWS CloudWatch?**

```
CloudWatch Monitoring for AWS Batch:
──────────────────────────────────────────────────────────────────
AWS Batch Metrics (automatic):
  - BatchJobsSucceeded/Failed/Runnable/Running
  - JobDuration (custom: Lambda publishes on job completion)

Pipeline Custom Metrics (via EMF logs):
  Python code:
  import json, datetime
  print(json.dumps({
    "_aws": {
      "Timestamp": int(datetime.datetime.now().timestamp() * 1000),
      "CloudWatchMetrics": [{
        "Namespace": "BatchInference",
        "Dimensions": [["ModelVersion", "PipelineRunId"]],
        "Metrics": [
          {"Name": "ThroughputRecordsPerSec", "Unit": "Count/Second"},
          {"Name": "ErrorRatePct", "Unit": "Percent"},
          {"Name": "GPUUtilizationPct", "Unit": "Percent"},
        ]
      }]
    },
    "ThroughputRecordsPerSec": 2341.5,
    "ErrorRatePct": 0.146,
    "GPUUtilizationPct": 84.2,
    "ModelVersion": "v2.1",
    "PipelineRunId": "batch_20240315_abc123"
  }))

CloudWatch Alarms:
  - ErrorRatePct > 5% → SNS → PagerDuty
  - GPUUtilizationPct < 50% for 30min → SNS → investigate bottleneck
  - BatchJobsRunnable > 0 for 20min → SNS → compute env at capacity
  - No Batch job submitted by 02:00 → SNS → scheduler failure
```

---

## 7. Post-Production Issues

| # | Issue | Cause | How to Detect | Solution | Prevention |
|---|-------|-------|---------------|----------|------------|
| 1 | OOM on large records | Input has text 50× longer than training data average; batch size not adaptive | Container exit code 137; `torch.cuda.OutOfMemoryError` in logs | Implement OOM retry with batch halving; add `max_length` truncation in tokenizer | Add input length distribution check at pipeline start; set p99 input length as benchmark |
| 2 | CUDA silent errors | `CUDA_LAUNCH_BLOCKING=0`; device-side assertion (e.g., invalid token ID) crashes GPU kernel silently | Wrong outputs (all zero) with no exception; `nvidia-smi` shows error state | Set `CUDA_LAUNCH_BLOCKING=1` in debug; validate token IDs before forward pass | Add post-inference sanity check: if mean confidence < 0.1 across batch, flag for review |
| 3 | Throughput degradation on mixed-length batches | Padding short sequences to max sequence in batch; matrix multiply cost scales with padding | Throughput drops 40% when mixed with long sequences | Sort inputs by length before batching; use dynamic padding per batch | Implement length-bucket batching: group inputs by similar lengths |
| 4 | Parquet corruption on partial write | Job killed (SIGTERM on Spot) mid-Parquet write; file written without footer | Downstream Parquet reads fail with `ArrowInvalid`; partial files with `.parquet` extension | Use atomic writes (write to `.parquet.tmp`, rename to `.parquet` only on completion) | Validate Parquet readability as pipeline step after write; spot instance preemption hooks |
| 5 | Job hangs on tokenizer download | No internet access in prod VPC; HuggingFace download attempted on startup | Job stuck at `from_pretrained()` for > 30 min; AWS Batch job times out | Set `TRANSFORMERS_OFFLINE=1`; bake model into Docker image or use EFS | Verify offline model availability in pre-flight check; containerize with model files |
| 6 | Checkpoint drift causing duplicates | Job restarted from checkpoint but output files not cleaned; append instead of overwrite | Downstream analytics shows 2× record count | Overwrite output partition on restart; use job_id-scoped paths | Integration test: run pipeline twice, verify output record count equals input |
| 7 | Benchmark over-estimates prod throughput | Benchmark uses short texts (20 tokens); prod has 400-token average | Prod throughput 5× lower than benchmark showed | Re-benchmark with production-representative text length distribution | Benchmark on p50/p95 length samples from production data before capacity planning |
| 8 | Workers exhausting file descriptors | `persistent_workers=True` + many DataLoader worker processes; each opens file handles | `OSError: [Errno 24] Too many open files` | Set `ulimit -n 65536`; reduce `num_workers`; use `persistent_workers=False` | Set `fs.file-max` and `ulimit` in container configuration; monitor fd count with psutil |
| 9 | Errors.jsonl growing unbounded | Long-running daily job; error rate 0.5% × 10M records = 50k error lines; error file accumulates | Disk full on output EFS/NVMe after several months | Rotate error files by date; move to S3 after each run; set size limit | Error files should be compressed and moved to long-term storage as pipeline step |
| 10 | psutil memory readings inaccurate | `proc.memory_info().rss` includes shared memory; multi-GPU setup shares GPU buffers | Reported memory appears higher than actual; false OOM alerts | Use `proc.memory_info().uss` (unique set size, non-shared) for accurate per-process memory | Document memory measurement methodology; calibrate against known memory usage |
| 11 | Model version mismatch in output | Model updated mid-job (Azure Blob race condition); some shards use old model, some new | Inconsistent prediction distributions across shards | Pin model version at job start; load from job-specific snapshot path | Immutable model artifacts: deploy to new path, never overwrite existing |
| 12 | AWS Spot interruption during last shard | EC2 Spot instance reclaimed with 2-minute warning; last shard not completed | Missing output shards; downstream merge fails | Implement 2-minute graceful shutdown hook: checkpoint current position, write partial output | Use AWS Spot interruption handler; save checkpoint on `SIGTERM`; AWS Batch retries interrupted tasks |
| 13 | GPU memory fragmentation | Many different tensor sizes; CUDA allocator cannot reuse fragmented memory | Effective VRAM decreases over 6-hour job run; OOM errors late in job that didn't occur early | `torch.cuda.empty_cache()` every 1000 batches; restart worker processes periodically | Use fixed-size tensors (pad to fixed max_length); monitor VRAM fragmentation with `memory_stats()` |
| 14 | DataLoader workers crash silently | Worker process hits OS limit (swap exhaustion, file descriptor limit) | DataLoader hangs; no exception in main process | Set `timeout=30` in DataLoader constructor to surface worker timeouts | Monitor worker process health; add heartbeat in worker's `collate_fn` |
| 15 | Pyarrow schema mismatch between shards | Different pandas/pyarrow versions across pipeline runs wrote different schemas | Glue/Athena merge fails: incompatible schemas | Enforce schema: pass explicit `pa.schema` to all `pq.write_table` calls | Pin pyarrow version in requirements.txt; validate schema in CI with sample data |
| 16 | Latency spike at batch boundary | Last batch of a chunk is smaller (drop_last=False); smaller batch has proportionally higher overhead | p99 latency 3× p50 latency | Normal behavior; document in benchmarks. Use `drop_last=True` if partial batches are acceptable | Profile batch latency distribution; add annotation in benchmark output |

---

## 8. General ML Interview Topics

**Q33. ⭐ Explain INT8 quantization for batch inference. What are the accuracy trade-offs?**

INT8 quantization replaces FP32 weights and activations with 8-bit integers, reducing model size by 4× and achieving 2–4× inference speedup via integer arithmetic units. Methods: (1) **Post-Training Quantization (PTQ)**: calibrate scale factors using a representative dataset (500–2000 samples), no retraining. BERT-base classification accuracy loss: typically < 0.5% on GLUE benchmarks. (2) **Quantization-Aware Training (QAT)**: simulate quantization during fine-tuning, recovering accuracy at the cost of retraining time. For batch inference with accuracy-sensitive use cases (medical, legal), run PTQ and measure accuracy on a held-out validation set before deploying INT8. Tools: HuggingFace Optimum, ONNX Runtime quantization, Intel Neural Compressor.

**Q34. ⭐⭐ Compare model optimization approaches: quantization vs pruning vs distillation vs ONNX export.**

| Approach | Memory savings | Speed gain | Accuracy loss | Effort |
|----------|----------------|------------|---------------|--------|
| INT8 quantization | 4× | 2–4× | < 1% (PTQ) | Low |
| FP16/BF16 | 2× | 1.5–3× | ~0% | Very low |
| Structured pruning | 2–10× | 2–5× | 1–5% | Medium |
| Knowledge distillation | 3–10× | 3–10× | 1–3% | High |
| ONNX export | 0–1.2× | 1.2–2× | 0% | Low |
| TensorRT | 0–4× | 2–8× | < 1% | Medium |

For batch inference: combine FP16 + ONNX + TensorRT for maximum throughput with minimal accuracy loss. Distillation (e.g., DistilBERT vs BERT-base: 40% smaller, 60% faster, 97% accuracy retention) is ideal when model size is the primary constraint.

**Q35. ⭐ What is the difference between CPU-bound, GPU-bound, and I/O-bound batch inference? How do you diagnose each?**

**CPU-bound**: `htop` shows DataLoader worker CPU cores at 100%; GPU utilization is < 50%. Cause: slow tokenization, insufficient workers, complex preprocessing. Fix: increase `num_workers`, simplify preprocessing, pre-tokenize data. **GPU-bound**: GPU utilization > 90%, CPU idle; this is the desirable state (GPU is the bottleneck as intended). Fix: increase batch size, use FP16, use a faster GPU. **I/O-bound**: `iostat` shows high `%iowait`; `psutil.Process().io_counters()` shows high `read_bytes` per second; GPU and CPU are both underutilized. Cause: reading from slow spinning disk, network-attached storage latency. Fix: move input data to NVMe SSD or memory; use multi-threaded I/O (increase DataLoader `num_workers`).

**Q36. ⭐⭐ How do Kubernetes Jobs and CronJobs differ from Deployments for batch inference?**

**Deployment**: manages long-running, continuously-serving pods. Automatically restarts failed pods. No concept of completion. Used for real-time inference APIs. **Job**: runs pods to completion (exit code 0). Supports parallelism (multiple pods processing different shards). Retries on failure. After all pods complete successfully, the Job is "complete." Used for one-time batch runs. **CronJob**: schedules a Job on a cron schedule. Creates a new Job (and pods) on each trigger. Important: set `concurrencyPolicy: Forbid` to prevent overlap if the previous job is still running when the next is scheduled. Set `successfulJobsHistoryLimit=3` and `failedJobsHistoryLimit=3` to avoid accumulating stale job objects. Set `activeDeadlineSeconds` to enforce maximum job duration.

**Q37. ⭐ What is pipeline parallelism in the context of data preprocessing for ML?**

In batch inference, the pipeline stages are: read → tokenize → GPU inference → serialize → write. Without parallelism, these run sequentially and each stage waits for the previous. With pipeline parallelism: the DataLoader reads and tokenizes stage N+1 while the GPU runs inference on stage N, while the writer serializes stage N-1. The overall throughput is determined by the slowest stage (bottleneck). Implement via: DataLoader's `prefetch_factor` (tokenization prefetch), a write queue (inference results queued for async writing using `concurrent.futures.ThreadPoolExecutor`). This overlap hides I/O latency behind GPU compute time.

**Q38. ⭐ What is the tradeoff between `chunk_size` and memory in the ChunkedReader?**

Larger chunks: fewer chunks (less iteration overhead), better tokenization batching, higher memory usage. Smaller chunks: lower memory usage, more frequent checkpointing (better fault tolerance), higher iteration overhead. Formula: `chunk_size = floor(available_ram * 0.3 / (avg_record_size_bytes * tokenizer_expansion_factor))`. The `tokenizer_expansion_factor` is ~2–4× (tokenized tensor is larger than the raw string). Check `psutil.virtual_memory().available` at startup and set chunk size dynamically. Minimum practical chunk size: 1000 records (below this, DataLoader process creation overhead dominates).

---

## 9. Behavioral / Scenario Questions

**Q39. ⭐ Walk me through how you would optimize a batch pipeline that is processing 100,000 records/hour but needs to process 500,000 records/hour.**

Step 1: Profile the current bottleneck using psutil and `nvidia-smi`. If GPU utilization < 70%: bottleneck is DataLoader (increase `num_workers`, enable `pin_memory`). If GPU utilization > 90%: bottleneck is GPU compute itself (optimize batch size, use FP16, upgrade GPU). If IO wait is high: bottleneck is storage (move to faster storage, parallel reads). Step 2: Benchmark batch sizes — plot the throughput curve (the `benchmark.py` script), identify the optimal batch size. Step 3: Profile inference with `torch.profiler` — identify which layers dominate compute time. Step 4: Consider model optimization (FP16 gives 2–3× throughput). Step 5: If single-GPU is maxed out at target throughput, scale horizontally — run N parallel workers on N GPUs. 5× throughput typically requires 2–3 GPUs with FP16 optimization (some headroom for DataLoader efficiency).

**Q40. ⭐⭐ A batch job that ran successfully for 6 months suddenly started failing 30% of records. How do you diagnose this?**

Step 1: Check `errors.jsonl` — what are the error types? If `TokenizerOverflowError` (text too long): upstream data changed, new records much longer than training data. If `ModelNaNOutput`: model weights corrupted (storage issue) or input distribution shifted causing numerical instability. Step 2: Compare input data statistics (length distribution, character distribution, language distribution) between a successful run and the failing run using pandas profiling. Step 3: Check if the model version changed (unintended model update). Step 4: Check if the tokenizer vocabulary changed (tokenizer update incompatible with model). Step 5: Sample 10 failing records and 10 passing records, run interactively to reproduce. Step 6: Once root cause identified, add input validation (length check, encoding check, language detection) to prevent future silent failures.

**Q41. ⭐ How would you explain the batch size optimization process to a data scientist unfamiliar with GPU infrastructure?**

"A GPU is like a massive array of calculators running in parallel — it is fastest when all calculators are busy at once. If we send only 8 records at a time, most calculators sit idle waiting for the next batch. If we send 64 records, nearly all calculators are active simultaneously. However, we can only hold so many records in GPU memory — like a desk that can only hold so many papers. We run a benchmark that tries batch sizes 8, 16, 32, 64, 128, measures how many records we process per second at each size, and picks the largest size that fits in GPU memory. That's what the `benchmark.py` script does — it draws a 'throughput curve' and we pick the 'knee' where adding more records stops helping."

---

## 10. Quick-Fire Questions

*Answer each in 1-3 sentences.*

**Q42. ⭐** What does `model.eval()` do?
**A:** Switches the model to evaluation mode, disabling dropout (which randomly zeros activations during training) and using running statistics in batch normalization instead of batch statistics. Always call before inference.

**Q43. ⭐** What is `torch.cuda.empty_cache()` and when should you call it?
**A:** Releases cached CUDA memory back to the OS (memory PyTorch has allocated but is no longer using). Call after catching OOM errors, before allocating a large tensor, or every few hundred batches to prevent fragmentation. It does not free memory held by living Python objects.

**Q44. ⭐** What is the difference between `Parquet` and `ORC` file formats?
**A:** Both are columnar formats. Parquet is the default in the Hadoop ecosystem, supported by virtually all big data tools. ORC is native to Hive and optimized for Hive workloads. Parquet is preferred for multi-tool compatibility (Spark, Athena, BigQuery, Pandas all read Parquet natively).

**Q45. ⭐** What is `pyarrow.Table.from_pylist()` vs `pd.DataFrame`?
**A:** Both create tabular structures from Python dicts. `pyarrow.Table` writes directly to Parquet with explicit schema enforcement and is faster for I/O-heavy workflows. `pd.DataFrame` is more convenient for transformations and analytics but adds overhead for pure write pipelines.

**Q46. ⭐** What does `click.IntRange(1, 512)` do?
**A:** Validates that the CLI integer argument is within [1, 512], raising a `click.BadParameter` error with a helpful message if the value is out of range. Prevents invalid batch sizes (0 or negative) from reaching the pipeline.

**Q47. ⭐** What is `torch.backends.cudnn.benchmark` and when should it be True or False?
**A:** When `True`, cuDNN benchmarks multiple convolution algorithms on the first run and selects the fastest. Set `True` for training with fixed input sizes (pays benchmark cost once). Set `False` for inference or variable input sizes (benchmark cost paid every time a new shape appears, hurting performance).

**Q48. ⭐** What is the `drop_last` parameter in DataLoader?
**A:** If `True`, the final batch is dropped when the dataset size is not divisible by batch size. For batch inference: use `drop_last=False` to process all records (missing 1% of records is not acceptable in production). For training: `drop_last=True` prevents batch norm issues with size-1 batches.

**Q49. ⭐⭐** What is `torch.compile()` and how does it help batch inference?
**A:** Introduced in PyTorch 2.0, `torch.compile(model)` JIT-compiles the model using TorchDynamo and TorchInductor, generating optimized CUDA kernels for the specific model architecture and input shapes. Speedup: 20–80% on transformer models. First call pays compilation cost (~30s); subsequent calls use cached compiled kernel. Use in batch inference where the warm-up cost is amortized across millions of records.

**Q50. ⭐** What is `TRANSFORMERS_OFFLINE=1`?
**A:** Environment variable that prevents HuggingFace `transformers` from attempting internet downloads. With this set, `from_pretrained()` only looks in the local cache path. Essential for production environments without internet access (air-gapped VPCs, corporate firewalls).

**Q51. ⭐** What is `snappy` compression and why is it the default for Parquet?
**A:** Snappy is a compression algorithm developed by Google, optimized for speed over compression ratio. It compresses at ~250MB/s and decompresses at ~500MB/s with ~2–3× compression ratio. It is the Parquet default because: fast write speed (not a bottleneck for GPU-bound inference pipelines), fast read speed (analytics queries are not decompression-bound), and wide tool support.

**Q52. ⭐⭐** How does `torch.profiler` help optimize batch inference?
**A:** `torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])` records detailed CPU and GPU kernel execution traces. Use `prof.key_averages()` to see which operations dominate inference time (typically: attention matrix multiply, tokenizer string processing, data transfer). Use the Chrome trace viewer or TensorBoard to visualize the timeline and identify GPU idle time. Optimization targets: reduce CPU→GPU transfer time (use `pin_memory`), reduce sequential operations (fuse them), eliminate GPU idle gaps (pipeline parallelism).

**Q53. ⭐** What is `DataLoader.prefetch_factor`?
**A:** Number of batches each DataLoader worker prefetches ahead. Default is 2 (each worker keeps 2 batches ready). Increase to 4 for I/O-bound DataLoaders (reading from slow storage) — allows more read-ahead. Decrease if the prefetched batches consume too much RAM (each prefetched batch occupies memory before the GPU can consume it).

**Q54. ⭐** What does `os.rename()` provide in terms of atomicity for file writes?
**A:** `os.rename()` is atomic on POSIX file systems (ext4, XFS, APFS) when source and destination are on the same filesystem. The file either fully exists at the new path or does not — there is no intermediate state. This prevents readers from seeing a partially-written Parquet file. Not atomic across filesystems (e.g., rename from /tmp to /mnt/nas) — in that case, use `shutil.move()` which copies then deletes, but this is not atomic.

**Q55. ⭐** What is the `HuggingFace datasets` streaming API and when does it help?
**A:** `load_dataset(..., streaming=True)` returns an `IterableDataset` that reads data lazily without downloading or caching the full dataset. Enables processing datasets larger than RAM (e.g., 500GB Common Crawl). Each call to `next()` fetches and processes one record on-demand. Use `dataset.take(n)` for a subset. Limitation: no random access (cannot shuffle efficiently) — use `dataset.shuffle(buffer_size=10_000)` for approximate shuffling.

**Q56. ⭐** What is `concurrent.futures.ThreadPoolExecutor` and how is it useful in batch inference?
**A:** A thread pool for running I/O-bound tasks concurrently. In batch inference: submit Parquet writes to the thread pool so the main process can start the next inference batch immediately: `executor.submit(writer.write_batch, results)`. Since Python's GIL allows I/O threads to run concurrently with the main thread (I/O releases the GIL), this achieves true parallel writes.

**Q57. ⭐** What is the `JSONL` format and how does it differ from regular JSON?
**A:** JSONL (JSON Lines) stores one JSON object per line, making it a sequence of independent JSON objects rather than a single JSON array. Benefits: (1) streaming-compatible (read/write line by line, not entire file), (2) append-friendly (`>>` append to file), (3) each record independently parseable (corrupted line doesn't break other lines), (4) gzip-compressible line by line. Preferred over JSON array for large datasets where you can't hold all records in memory.

**Q58. ⭐⭐** What is `torch.utils.data.IterableDataset` vs `Dataset`? When should batch inference use each?
**A:** `Dataset` requires `__len__` and `__getitem__` (random access by index). `IterableDataset` only requires `__iter__` (sequential access). Use `IterableDataset` for batch inference with streaming input (HuggingFace streaming datasets, reading from S3 on-the-fly) where dataset size is unknown or too large for indexing. Use `Dataset` when random access is needed (sampling, shuffling) or when the full chunk fits in memory. Note: `IterableDataset` does not support automatic shuffling or `num_workers > 1` without custom worker init functions.

**Q59. ⭐** How do you handle Unicode normalization in the batch tokenizer?
**A:** Apply `unicodedata.normalize('NFC', text)` before tokenization. NFC (Canonical Decomposition, Canonical Composition) ensures characters like `é` are represented as a single code point rather than `e` + combining accent. Without normalization: the same text submitted from different browsers/platforms may tokenize differently, producing inconsistent predictions. Add normalization as the first step in `TextDataset.__getitem__`.

**Q60. ⭐** What is the `AWS_BATCH_JOB_ARRAY_INDEX` environment variable?
**A:** Automatically set by AWS Batch for Array Jobs. Value: 0 to N-1 where N is the array size. Each array child job gets a unique index, which the pipeline uses to determine which input shard to process: `shard_id = int(os.environ['AWS_BATCH_JOB_ARRAY_INDEX'])`. This enables embarrassingly parallel batch processing without a coordinator service.

**Q61. ⭐** What is `psutil.virtual_memory().available` vs `.free`?
**A:** `free` is RAM not currently allocated to any process (immediately available). `available` (recommended) estimates how much memory is available for allocation without swapping, including reclaimable cache (Linux's page cache). `available` is always >= `free` and is the better metric for "how much RAM can I use?" since Linux aggressively uses free memory as disk cache.

**Q62. ⭐** What does `statistics.quantiles(data, n=100)[98]` compute?
**A:** The 99th percentile of `data` using the exclusive method. `quantiles(data, n=100)` returns a list of 99 cut points dividing the data into 100 equal-probability groups. Index 98 is the 99th quantile (0-indexed: 0 = p1, 98 = p99). For small samples (< 100 data points), the 99th percentile estimate may be unreliable; use `numpy.percentile(data, 99)` with the `interpolation='nearest'` method for small samples.

**Q63. ⭐⭐** How would you implement dynamic padding (pad to longest in batch) in a custom `collate_fn`?
**A:**
```python
def collate_fn(batch: List[dict]) -> dict:
    max_len = max(len(item['input_ids']) for item in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, item in enumerate(batch):
        n = len(item['input_ids'])
        input_ids[i, :n] = torch.tensor(item['input_ids'])
        attention_mask[i, :n] = 1
    return {'input_ids': input_ids, 'attention_mask': attention_mask}
```
This pads each batch to the longest sequence in that batch rather than global max_length, significantly reducing compute for batches containing short texts.

**Q64. ⭐** What is Azure Data Factory's "trigger" and how does it schedule batch jobs?
**A:** An ADF trigger defines when a pipeline runs. Types: (1) **Schedule trigger**: cron expression (e.g., `0 1 * * *` = daily at 01:00 UTC), fires on a fixed schedule. (2) **Tumbling window trigger**: like schedule but tracks windows and can backfill missed runs. (3) **Storage event trigger**: fires when a blob is created or deleted in Azure Blob Storage — ideal for event-driven batch inference (trigger when new input file arrives). (4) **Custom event trigger**: fires on Azure Event Grid events.

**Q65. ⭐** What is the `--fp16/--no-fp16` Click option pattern and why is it better than `--fp16 True`?
**A:** `--fp16/--no-fp16` creates a pair of boolean flags where `--fp16` sets the value to `True` and `--no-fp16` sets it to `False`. This is more idiomatic CLI design than `--fp16 True/False` because: boolean flags (no argument) are easier to type, clearer in scripts (`python run.py --no-fp16` vs `python run.py --fp16 False`), and match Unix conventions for enabling/disabling features.

**Q66. ⭐** How do you verify the output Parquet files are correct after the pipeline completes?
**A:** Run a validation step: (1) count total rows across all Parquet shards and verify equals `total_input_records - error_records`, (2) read schema and verify matches expected schema, (3) check for nulls in required columns, (4) sample 100 rows and verify prediction labels are in the expected set, (5) check confidence scores are in [0, 1]. Implement as a post-pipeline validation function that raises an exception if any check fails, preventing corrupted results from reaching downstream consumers.

**Q67. ⭐⭐** What is TensorRT and when should you use it for batch inference?
**A:** TensorRT is NVIDIA's inference optimization platform that compiles PyTorch/ONNX models into optimized CUDA engines for a specific GPU and batch size. Optimizations include: operator fusion (merge multiple CUDA kernels into one), precision calibration (INT8/FP16), memory planning (minimize intermediate buffers), and kernel auto-selection. Speedup over PyTorch inference: 2–8× depending on model architecture. Use TensorRT when: GPU is the bottleneck and further throughput improvement is needed, the batch size is fixed (TensorRT engines are compiled for specific shapes), and NVIDIA GPUs are used. Downside: compilation takes 30–90 minutes for large models, engines are GPU-specific (can't use A100 engine on V100).

**Q68. ⭐** What does `AWS Batch array job` enable that a regular Batch job does not?
**A:** An Array Job launches N child jobs with a single API call, where N is the array size. Each child gets a unique `AWS_BATCH_JOB_ARRAY_INDEX` (0 to N-1). This enables parallel processing of N independent shards without submitting N separate jobs: significantly reduces AWS API call overhead, job submission latency, and allows AWS Batch to schedule all N children as a unit (better queue priority).

**Q69. ⭐** What is `gc.collect()` and when is it necessary in Python ML pipelines?
**A:** `gc.collect()` manually triggers Python's garbage collector to find and free unreachable objects, particularly those involved in reference cycles (object A references B, B references A, both unreachable from global scope). In ML pipelines, deep PyTorch computation graphs create reference cycles. After `del model_output`, a `gc.collect()` ensures the memory is actually freed rather than waiting for the next automatic GC cycle. Important after large inference batches where autograd graph references create cycles.

**Q70. ⭐** What monitoring metric best indicates that your batch pipeline is I/O-bound rather than GPU-bound?
**A:** `psutil.disk_io_counters().read_bytes` per second (or per minute) vs theoretical disk bandwidth. If disk read speed equals disk maximum bandwidth (NVMe: ~3.5GB/s, SATA SSD: ~500MB/s, spinning disk: ~150MB/s) while GPU utilization is low, the pipeline is I/O-bound. Also: if increasing `num_workers` from 4 to 8 does not improve GPU utilization further, the storage device is the bottleneck rather than the CPU tokenization process.

**Q71. ⭐** What is `EFS` (Elastic File System) in the context of AWS Batch batch inference?
**A:** AWS EFS is a managed NFS file system that can be simultaneously mounted by multiple EC2 instances or Fargate tasks. For batch inference: mount the model artifacts directory from EFS so all parallel array job tasks share the same model files without each task independently downloading from S3. This eliminates redundant S3 downloads for large models (10GB+) and provides consistent performance since EFS is low-latency for concurrent reads. Throughput mode: provisioned throughput (fixed bandwidth, predictable performance) for model-serving workloads.

**Q72. ⭐⭐** How do you implement time-bounded batch inference (complete within N hours or terminate gracefully)?
**A:** Register a SIGALRM handler (Unix) or use a background thread with a timer. In the pipeline, check elapsed time every chunk: if elapsed > budget × 0.95 (95% of budget), flush current buffer, write checkpoint, log remaining unprocessed records to a "deferred" queue, and exit cleanly. Return stats indicating partial completion. The 95% threshold leaves headroom for finalization (flush + checkpoint write). Downstream: process the deferred queue in the next scheduled run.

**Q73. ⭐** What is `itertools.islice` and why is it useful in `scripts/benchmark.py`?
**A:** `itertools.islice(iterable, n)` takes only the first N items from an iterable without consuming the rest. In benchmarking: `itertools.islice(dataloader, n_warmup + n_trials)` avoids iterating over the entire dataset just for benchmarking — you need only `n_warmup + n_trials` batches (typically 15–25 batches) to measure performance, not the full 1M-record dataset.

**Q74. ⭐** What is the purpose of `torch.cuda.reset_peak_memory_stats()`?
**A:** Resets the tracked peak VRAM usage counter to the current allocation level. Use before each benchmark trial to get per-trial peak memory: `torch.cuda.reset_peak_memory_stats(); run_inference(); peak = torch.cuda.max_memory_allocated()`. Without reset, `max_memory_allocated()` returns the maximum since process start, which reflects the worst-case batch (not the current batch size being benchmarked).

**Q75. ⭐** What is `AWS Step Functions` `Map` state and how does it enable parallel batch processing?
**A:** The `Map` state iterates over an array (e.g., list of 1000 shard paths from S3) and executes a nested state machine for each element in parallel, up to `MaxConcurrency` (set to the number of GPU workers available). Each nested state machine: starts an AWS Batch job, waits for completion, validates output. The Map state completes when all parallel executions finish. This replaces manual job coordination logic with declarative parallel orchestration.

**Q76. ⭐** How does `HuggingFace AutoTokenizer` handle unknown tokens and why does this matter for batch inference?
**A:** `AutoTokenizer` replaces unknown characters with the `[UNK]` token. If production data contains many characters not in the tokenizer vocabulary (rare Unicode, domain-specific symbols, emojis not in training data), a high proportion of tokens become `[UNK]`, degrading prediction quality. Monitor: log the `[UNK]` token ratio per batch. Alert if average ratio exceeds 5% (indicates vocabulary mismatch with production data, a form of covariate shift). Fix: retrain tokenizer on production data, or filter inputs to known vocabulary.

**Q77. ⭐** What does `@contextlib.contextmanager` vs `torch.autocast` provide as a context manager?
**A:** `@contextlib.contextmanager` is a Python decorator that turns a generator function into a context manager (used in `asynccontextmanager` for FastAPI lifespans and in pipeline resource management). `torch.autocast` is a built-in PyTorch context manager that sets the mixed-precision casting policy for a code block. They are different layers: `contextlib` provides the pattern, `autocast` uses the pattern for a specific ML purpose.

**Q78. ⭐** What is the optimal `num_workers` formula for a DataLoader?
**A:** `num_workers = max(1, min(os.cpu_count() - 1, ceil(batch_size / items_per_worker_capacity)))`. In practice: start with `cpu_count() - 1` and benchmark. If GPU utilization increases when you increase `num_workers`, the DataLoader was the bottleneck. If GPU utilization is already high with fewer workers, adding more does not help and wastes memory (each worker has its own copy of the dataset). Diminishing returns above 8 workers for most tokenization workloads.

**Q79. ⭐** What is the `errors.jsonl` error rate threshold for aborting a batch job?
**A:** Common practice: warn at 1% error rate, abort at 10% error rate. The 10% threshold assumes the downstream use case can tolerate 10% missing predictions; adjust based on requirements. For critical use cases (medical, financial): abort at 0.1% and require manual investigation. Implement: `if error_count / total_processed > ABORT_THRESHOLD: writer.close(); raise BatchAbortError(f"Error rate {rate:.1%} exceeds threshold")`.

**Q80. ⭐⭐** What is the `torch.profiler.schedule` and how do you use it in production batch profiling?
**A:** `torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2)` defines a profiling schedule: skip 1 step (wait), warm up 1 step (collect but discard), actively collect 3 steps, repeat twice. This minimizes profiling overhead — you don't profile every batch (too much data), just enough to identify bottlenecks. Use in production periodically (every 1000 batches) to detect performance regressions. Emit profiling data to TensorBoard or custom analytics for long-term trending.

**Q81. ⭐** What is `HuggingFace Accelerate` and how does it simplify multi-GPU batch inference?
**A:** `accelerate` is a HuggingFace library that abstracts distributed training and inference setup. With minimal code changes: `accelerator = Accelerator(); model, dataloader = accelerator.prepare(model, dataloader)`. It handles: model distribution across multiple GPUs, distributed batch splitting, gradient accumulation, mixed precision. For multi-GPU batch inference: each GPU processes a subset of batches, results gathered at the end. Eliminates boilerplate `DistributedDataParallel` setup and `torch.distributed.init_process_group()` calls.

---

*End of Batch Inference Pipeline Interview Preparation Guide — 81 numbered questions + follow-up chains*
*Total coverage: ~200+ distinct questions including follow-ups*
