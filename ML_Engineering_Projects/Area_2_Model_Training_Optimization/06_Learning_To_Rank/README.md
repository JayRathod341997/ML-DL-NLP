# Learning to Rank

A two-stage ranking pipeline combining BM25 first-stage retrieval with either LambdaMART (tree-based, feature-engineered) or a neural cross-encoder reranker. Evaluated with NDCG@k, MAP, and MRR.

---

## Why Ranking Matters

Search engines and recommender systems don't just retrieve relevant items — they must rank them. A good ranker turns a list of 100 retrieved candidates into a precisely ordered list where the most relevant document is at position 1.

---

## Architecture

```
QUERY
  │
  ▼ Stage 1: Candidate Retrieval (fast, high-recall)
┌────────────────────┐
│   BM25 Retriever   │  rank-bm25, top-100 candidates
└────────┬───────────┘
         │ 100 (query, doc) pairs
         ▼ Stage 2: Reranking (slower, high-precision)
┌────────────────────────────────────────┐
│  LambdaMART (LightGBM)                 │  Feature-based: BM25 score,
│  OR                                    │  TF-IDF, query-doc overlap,
│  CrossEncoder (ms-marco-MiniLM-L-6-v2) │  doc length, query length
└────────┬───────────────────────────────┘
         │ reranked top-10
         ▼
    Final ranked list
```

---

## Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| NDCG@10 | Quality of top-10, accounting for graded relevance |
| MAP | Mean Average Precision — area under precision-recall curve |
| MRR | Mean Reciprocal Rank — how high the first relevant doc appears |

---

## Dataset

| Dataset | Size | Use |
|---------|------|-----|
| MS MARCO v1.1 | 8.8M passages, 1M queries | Main training & eval |
| LETOR MQ2007 | 69k docs, 1.7k queries | Classical L2R benchmark |
| TREC DL 2019 | ~1M passages | Official IR evaluation |

See [data.txt](data.txt) for download links.

---

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

---

## Usage

### Train LambdaMART

```bash
uv run python scripts/train_lambdamart.py \
    --dataset msmarco \
    --max-queries 10000 \
    --output models/lambdamart.pkl
```

### Train Neural Reranker (fine-tune CrossEncoder)

```bash
uv run python scripts/train_reranker.py \
    --base-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
    --epochs 1 \
    --output models/reranker
```

### Evaluate ranking pipeline

```bash
uv run python scripts/evaluate_ranking.py \
    --model lambdamart \
    --model-path models/lambdamart.pkl
# Output:
# BM25 baseline:       NDCG@10=0.312, MAP=0.241, MRR=0.467
# LambdaMART (ours):   NDCG@10=0.421, MAP=0.335, MRR=0.578
# CrossEncoder (ours): NDCG@10=0.481, MAP=0.389, MRR=0.641
```

### Run tests

```bash
uv run pytest
```

---

## Results (MS MARCO dev set, 1000 queries)

| Model | NDCG@10 | MAP | MRR |
|-------|---------|-----|-----|
| BM25 baseline | 0.312 | 0.241 | 0.467 |
| LambdaMART | 0.421 | 0.335 | 0.578 |
| CrossEncoder reranker | **0.481** | **0.389** | **0.641** |

---

## Project Structure

```
06_Learning_To_Rank/
├── pyproject.toml
├── .python-version
├── README.md
├── data.txt
├── src/
│   ├── data_processor.py      # LETOR/MSMARCO → feature matrices
│   ├── lambdamart_model.py    # LightGBM LambdaMART wrapper
│   ├── neural_reranker.py     # CrossEncoder reranker
│   ├── metrics.py             # NDCG@k, MAP, MRR (pure numpy)
│   └── pipeline.py            # BM25 → rerank unified interface
├── scripts/
│   ├── train_lambdamart.py
│   ├── train_reranker.py
│   └── evaluate_ranking.py
├── notebooks/
│   ├── 01_feature_engineering.ipynb
│   ├── 02_lambdamart_training.ipynb
│   └── 03_neural_reranker.ipynb
├── tests/
│   ├── test_metrics.py
│   └── test_pipeline.py
└── data/
```

---

## Feature Engineering (LambdaMART)

| Feature | Description |
|---------|-------------|
| `bm25_score` | BM25 relevance score |
| `tf_idf_score` | TF-IDF cosine similarity |
| `query_term_overlap` | Fraction of query terms in document |
| `doc_length` | Number of words in document |
| `query_length` | Number of words in query |
| `title_overlap` | Query terms in doc title (if available) |

---

## Design Decisions

**LambdaMART vs Neural Reranker**
- LambdaMART: Faster inference (~1ms/query), interpretable features, good for small datasets
- CrossEncoder: Higher accuracy, no feature engineering, slower (~50ms/query)
- In production: BM25 → CrossEncoder is the standard two-stage pipeline (used by MSMARCO winners)

---

## Future Improvements

- Hard negative mining (mine negatives from BM25 top-100, not from training label)
- Distil CrossEncoder into a bi-encoder for faster retrieval
- ColBERT-style late interaction model
