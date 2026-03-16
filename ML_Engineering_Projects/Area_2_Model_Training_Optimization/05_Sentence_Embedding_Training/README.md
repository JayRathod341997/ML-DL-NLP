# Sentence Embedding Model Training

Fine-tune a sentence transformer model using NLI data and MultipleNegativesRankingLoss (MNRL). Evaluates on STS Benchmark (Spearman correlation) and visualises learned semantic clusters with UMAP.

---

## Why Fine-Tune Embeddings?

Off-the-shelf embedding models are general-purpose. Fine-tuning on domain-specific data (e.g., legal, medical, e-commerce) can improve retrieval quality by 5-15% Spearman r on in-domain benchmarks.

---

## Training Approaches

### 1. MultipleNegativesRankingLoss (MNRL)
Uses NLI entailment pairs as (anchor, positive). All other positives in the batch become implicit negatives. Best when you have millions of (query, relevant_doc) pairs.

```
Batch:  [(A1, P1), (A2, P2), ..., (An, Pn)]
Loss:   CrossEntropy over similarity matrix
        (anchor should be most similar to its positive, not others)
```

### 2. CosineSimilarityLoss on STS-B
Uses labelled sentence pairs with float similarity scores [0,1]. Trains model to produce embeddings whose cosine similarity matches the human label. Better for fine-grained semantic similarity.

---

## Architecture

```
Base Model: sentence-transformers/all-MiniLM-L6-v2
    │
    ├── Encoder (DistilBERT, 6 layers, 384-dim)
    └── Mean Pooling → L2 Normalisation
```

---

## Dataset

| Dataset | Pairs | Use |
|---------|-------|-----|
| SNLI | 570k | MNRL training (entailment pairs) |
| MultiNLI | 393k | MNRL training (additional genres) |
| STS-B | 8,628 | Evaluation (Spearman r) |

See [data.txt](data.txt) for download links.

---

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

---

## Usage

### Train with MNRL loss (NLI data)

```bash
uv run python scripts/train_mnrl.py \
    --base-model sentence-transformers/all-MiniLM-L6-v2 \
    --epochs 1 \
    --batch-size 64 \
    --output-dir models/mnrl_model
```

### Train with CosineSimilarity loss (STS-B)

```bash
uv run python scripts/train_sts.py \
    --base-model sentence-transformers/all-MiniLM-L6-v2 \
    --epochs 4 \
    --output-dir models/sts_model
```

### Evaluate a trained model

```bash
uv run python scripts/evaluate_model.py --model-dir models/mnrl_model
# Output:
# STS-B Spearman r: 0.8742  (vs baseline: 0.8221)
```

### Run tests

```bash
uv run pytest
```

---

## Results

| Model | STS-B Spearman r |
|-------|-----------------|
| all-MiniLM-L6-v2 (baseline) | 0.8221 |
| + MNRL fine-tuning (1 epoch) | 0.8534 |
| + STS-B fine-tuning (4 epochs) | 0.8742 |

---

## Project Structure

```
05_Sentence_Embedding_Training/
├── pyproject.toml
├── .python-version
├── README.md
├── data.txt
├── src/
│   ├── data_loader.py     # NLI pairs + STS-B loading
│   ├── losses.py          # MNRL + CosineSimilarity wrappers
│   ├── trainer.py         # SentenceTransformerTrainer wrapper
│   ├── evaluator.py       # Spearman r on STS-B
│   └── model_utils.py     # save, load, push_to_hub
├── scripts/
│   ├── train_mnrl.py
│   ├── train_sts.py
│   └── evaluate_model.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_training.ipynb
│   └── 03_embedding_visualization.ipynb  # UMAP plots
├── tests/
│   ├── test_data_loader.py
│   └── test_evaluator.py
└── data/
```

---

## Design Decisions

**MNRL over Triplet Loss**
MNRL uses all in-batch negatives, making it ~N times more efficient per batch than triplet loss (which uses one explicit negative per sample). This is why it's used by the SBERT authors for training at scale.

**Mean Pooling over [CLS]**
Mean pooling of all token embeddings outperforms using just the [CLS] token for sentence similarity tasks (SBERT paper, 2019).

**Learning rate = 2e-5, warmup = 10%**
Standard recipe from SBERT paper. Higher LR causes catastrophic forgetting of pre-trained representations.

---

## Future Improvements

- Domain-specific fine-tuning (legal, medical using domain-specific NLI)
- Hard negative mining (mine negatives from BM25 retrievals)
- Evaluate on MTEB full benchmark
- Export model to sentence-transformers HuggingFace Hub
