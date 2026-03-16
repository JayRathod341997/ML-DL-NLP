# Text Classification with Transformers (Fine-tune DistilBERT)

Fine-tune a DistilBERT model for multi-class text classification with a complete training pipeline: custom Dataset, training loop with warmup scheduling, evaluation with F1/confusion matrix, and model checkpointing.

---

## Architecture

```
Input Text
    │
    ▼
┌────────────────────────┐
│   Tokenizer            │  distilbert-base-uncased, max_length=128
│   (WordPiece)          │  padding, truncation, attention_mask
└────────────┬───────────┘
             │ {input_ids, attention_mask}
             ▼
┌────────────────────────┐
│   DistilBERT Encoder   │  6 transformer layers, 768 hidden dim
│   (frozen first N      │  Pre-trained on BookCorpus + Wikipedia
│    layers optional)    │
└────────────┬───────────┘
             │ [CLS] token embedding (768-dim)
             ▼
┌────────────────────────┐
│   Dropout (p=0.1)      │
└────────────┬───────────┘
             │
             ▼
┌────────────────────────┐
│   Linear (768 → K)     │  K = number of classes
└────────────┬───────────┘
             │ logits
             ▼
       CrossEntropyLoss / Softmax Predictions
```

---

## Dataset

| Dataset | Classes | Train Size | Test Size |
|---------|---------|------------|-----------|
| AG News | 4 | 120,000 | 7,600 |
| DBpedia | 14 | 560,000 | 70,000 |
| IMDB | 2 | 25,000 | 25,000 |
| SST-2 | 2 | 67,349 | 872 |

Default: **AG News** (fast training, clear classes). See [data.txt](data.txt).

---

## Setup

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. (Optional) Login to Weights & Biases for experiment tracking
uv run wandb login
```

---

## Usage

### Train

```bash
# Train on AG News with default config (3 epochs, lr=2e-5)
uv run python scripts/train.py

# Custom config
uv run python scripts/train.py \
    --dataset ag_news \
    --model distilbert-base-uncased \
    --epochs 5 \
    --lr 3e-5 \
    --batch-size 32 \
    --output-dir checkpoints/run1
```

### Evaluate a checkpoint

```bash
uv run python scripts/evaluate.py --checkpoint checkpoints/best_model
```

### Predict on new text

```bash
uv run python scripts/predict.py --text "Scientists discover new species of dinosaur in Argentina"
# Output: Sci/Tech (confidence: 0.94)

uv run python scripts/predict.py --file texts.txt
```

### Run tests

```bash
uv run pytest
uv run pytest --cov=src --cov-report=term-missing
```

---

## Results (AG News, 3 epochs)

| Metric | Value |
|--------|-------|
| Test Accuracy | 94.2% |
| Macro F1 | 0.942 |
| Training time (CPU) | ~45 min |
| Training time (GPU) | ~8 min |

**Per-class F1:**

| Class | F1 |
|-------|-----|
| World | 0.94 |
| Sports | 0.98 |
| Business | 0.91 |
| Sci/Tech | 0.92 |

**Baseline comparison:**

| Model | Accuracy |
|-------|----------|
| TF-IDF + Logistic Regression | 89.1% |
| DistilBERT (zero-shot) | 62.3% |
| **DistilBERT (fine-tuned, ours)** | **94.2%** |

---

## Project Structure

```
04_Text_Classification_BERT/
├── pyproject.toml
├── .python-version
├── README.md
├── data.txt
├── src/
│   ├── config.py          # TrainConfig dataclass
│   ├── dataset.py         # PyTorch Dataset + tokenisation
│   ├── model.py           # DistilBERT + classification head
│   ├── trainer.py         # Training loop, AdamW, warmup
│   ├── evaluator.py       # F1, confusion matrix, reports
│   └── predict.py         # Inference on new texts
├── scripts/
│   ├── train.py           # CLI training entry point
│   ├── evaluate.py        # Evaluate a saved checkpoint
│   └── predict.py         # Predict on file/text
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_run.ipynb
│   └── 03_error_analysis.ipynb
├── tests/
│   ├── test_dataset.py
│   └── test_model.py
├── checkpoints/           # Saved model weights (gitignored)
└── data/
```

---

## Hyperparameter Choices

| Param | Value | Reason |
|-------|-------|--------|
| Learning rate | 2e-5 | Standard for BERT fine-tuning |
| Warmup steps | 500 | ~10% of training steps |
| Batch size | 32 | Fits 8GB GPU; use 16 for CPU |
| Max length | 128 | 95th percentile of AG News lengths |
| Dropout | 0.1 | DistilBERT default |
| Epochs | 3 | Diminishing returns after 3 |

---

## Future Improvements

- Quantisation (INT8) for 4x smaller inference
- Export to ONNX for language-agnostic serving
- Serve via FastAPI (see Project 07)
- Experiment with RoBERTa or ELECTRA for higher accuracy
