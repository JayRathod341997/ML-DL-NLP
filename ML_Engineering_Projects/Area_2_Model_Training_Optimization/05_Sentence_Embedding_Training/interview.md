# Sentence Embedding Training — Interview Preparation Guide

> **Stack**: sentence-transformers>=3.0, torch, accelerate, scipy, umap-learn
> **Data**: SNLI + MultiNLI (NLI pairs), STS-B (Spearman r evaluation)
> **Model**: sentence-transformers fine-tuned bi-encoder

---

## Quick Reference Card

| Component | Key Detail |
|---|---|
| Architecture | Bi-encoder (Siamese network) |
| Base model | all-MiniLM-L6-v2 or similar |
| Pooling | Mean pooling (default) |
| Primary loss | MultipleNegativesRankingLoss (MNRL) |
| Secondary loss | CosineSimilarityLoss for STS pairs |
| Evaluation metric | Spearman r on STS-B |
| NLI data | SNLI (~550k) + MultiNLI (~433k) |
| STS-B format | Sentence pairs scored 0-5 (normalized 0-1) |
| Similarity | Cosine similarity (requires L2 normalization) |
| Embedding dim | 384 (MiniLM) or 768 (BERT-base) |

---

## 1. Core Concepts & Theory

### 1.1 Bi-encoder vs Cross-encoder Architecture

**Q1. ⭐ What is a bi-encoder and how does it differ from a cross-encoder?**

A **bi-encoder** (also called a Siamese network) encodes each sentence independently: `u = encoder(sentence_A)`, `v = encoder(sentence_B)`, then computes similarity as `cos(u, v)`. The key property is that sentence embeddings can be computed and cached independently — similarity between N sentences against a database of M sentences requires N+M forward passes (plus fast vector search), not N×M. A **cross-encoder** concatenates both sentences as input: `score = encoder([CLS] sentence_A [SEP] sentence_B [SEP])` and produces a single scalar relevance score. Cross-encoders have full token-level interaction between both sentences, making them significantly more accurate but requiring N×M forward passes for exhaustive ranking — O(N×M) complexity that is prohibitive for retrieval from large corpora.

```
Bi-encoder vs Cross-encoder
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 BI-ENCODER                    CROSS-ENCODER
 ───────────                   ─────────────
 Sentence A    Sentence B      [CLS] Sentence A [SEP] Sentence B [SEP]
     │              │                          │
  Encoder A      Encoder B                  Encoder
  (shared wts)   (shared wts)                   │
     │              │                        Score
   Emb A          Emb B
     │              │
     └──cos sim──────┘
          Score

 PROS: Fast inference O(1) per        PROS: High accuracy, full
       query after indexing,                interaction
       cacheable, scalable            CONS: Slow O(N*M), not cacheable
 CONS: Lower accuracy than           USE CASE: Reranking top-k
       cross-encoder
 USE CASE: First-stage retrieval
```

---

**Q2. ⭐ What are the latency and accuracy trade-offs between bi-encoder and cross-encoder?**

Empirically, on MSMARCO passage retrieval: bi-encoder (Dense Passage Retrieval) achieves MRR@10 ~0.31; cross-encoder reranker achieves ~0.37. The cross-encoder is 20% more accurate but requires N×M forward passes. Latency numbers on a single V100: bi-encoder encodes 1 query in ~2ms + ANN search in ~1ms = ~3ms total; cross-encoder scoring 100 candidates takes ~200ms. This 67x latency difference is why the two-stage pipeline (bi-encoder retrieval → cross-encoder reranking) is the production standard. The bi-encoder operates at the retrieval stage on the full corpus; the cross-encoder only operates on the top 100-1000 retrieved candidates.

---

**Q3. ⭐⭐ What is a Siamese network architecture and how does it enable shared weights?**

A Siamese network has two identical branches with tied (shared) weights — any update to one branch is mirrored to the other. For sentence embeddings, this means the same transformer model encodes both sentences in a pair. During training on a pair (A, B), the loss gradients are computed for the similarity score and backpropagated through both branches — since weights are tied, the actual gradient update for each parameter is the sum of gradients from branch A and branch B. Weight sharing is critical: (1) it ensures the embedding space is consistent — both sentences are measured on the same scale; (2) it halves the number of parameters; (3) it acts as regularization, preventing the two branches from diverging. In practice, both sentences in a batch are often processed in a single forward pass by concatenating them, then splitting the output — this avoids overhead from two separate forward passes.

**Follow-up A:** Can a bi-encoder have asymmetric encoders (different weights for query and document)?
Yes — this is called an asymmetric bi-encoder or dual-encoder. Used in production search systems (DPR, E5) where queries are short and documents are long. The query encoder is often a smaller model (e.g., 6-layer) for low-latency query processing, while the document encoder is a larger model (e.g., 12-layer) for higher-quality document representations. The document index is built offline, so document encoder latency doesn't affect query latency.

**Follow-up B:** How does the embedding dimensionality affect the trade-off between compression and expressivity?
Higher dimensionality (768 vs 384) preserves more information but requires more storage and slower ANN search. At 1M documents: 768-dim embeddings take ~3GB (float32); 384-dim take ~1.5GB. Quantization to int8 halves storage again. Matryoshka Representation Learning (MRL) trains a model so that truncating to any prefix dimension (768→512→256→128) still gives useful embeddings, enabling you to dynamically choose dimensionality vs. speed trade-off at query time.

**Follow-up C:** What is the role of the pooling layer in a bi-encoder?
The transformer produces token-level embeddings of shape `(seq_len, hidden_dim)`. The pooling layer reduces this to a single sentence vector `(hidden_dim,)`. Options: **mean pooling** averages all non-padding token embeddings (standard, works best empirically); **CLS pooling** takes the [CLS] token embedding (pretrained BERT has poorly calibrated [CLS] for similarity); **max pooling** takes element-wise maximum across token dimensions (captures peak activation signals). Sentence-BERT (Reimers & Gurevych 2019) showed mean pooling outperforms CLS pooling for semantic similarity tasks.

---

### 1.2 Contrastive Learning & Loss Functions

**Q4. ⭐ What is contrastive learning and how does it apply to sentence embeddings?**

Contrastive learning trains a model to map similar inputs to nearby points in embedding space and dissimilar inputs to far-apart points. The training signal comes from positive pairs (semantically similar sentences) and negative pairs (semantically dissimilar sentences). For sentence embeddings, positives are typically: entailment pairs from NLI, paraphrase pairs, question-answer pairs. Negatives are: contradiction pairs from NLI, random sentences from the corpus (easy negatives), or hard negatives (superficially similar but semantically different). The loss function penalizes high similarity for negative pairs and rewards high similarity for positive pairs. Contrastive learning enables self-supervised or semi-supervised training — the supervision signal comes from the data structure (NLI labels, paraphrase databases) rather than manual sentence-level similarity annotations.

---

**Q5. ⭐ Compare TripletLoss, ContrastiveLoss, CosineSimilarityLoss, and MNRL — when to use each.**

| Loss | Input Format | Pros | Cons | Use Case |
|---|---|---|---|---|
| ContrastiveLoss | (sent_A, sent_B, label=0/1) | Simple, works with binary pairs | Hard threshold margin, needs negative pairs | Small datasets with explicit pos/neg pairs |
| TripletLoss | (anchor, positive, negative) | Explicit negative mining, intuitive | Requires explicit negatives, slower convergence | When you have curated triplets |
| CosineSimilarityLoss | (sent_A, sent_B, score 0-1) | Direct regression on similarity, good for STS | Requires continuous similarity labels | STS-B fine-tuning, graded similarity |
| MNRL | (sent_A, sent_B) positive pairs | Scales with batch size, no explicit negatives needed, state-of-the-art | Large batch required, memory intensive | NLI/paraphrase data, retrieval tasks |

---

**Q6. ⭐⭐ Explain how MultipleNegativesRankingLoss (MNRL) works in detail.**

MNRL treats all other examples in the batch as negatives for each anchor. Given a batch of N positive pairs `{(a_i, b_i)}_{i=1}^N`, for each anchor `a_i`, the positive `b_i` must score higher than all other in-batch sentences `{b_j}_{j≠i}` (and optionally `{a_j}_{j≠i}`). The loss is a cross-entropy over all N similarity scores:
```
L = -1/N * Σ_i log[ exp(cos(a_i, b_i) / τ) / Σ_j exp(cos(a_i, b_j) / τ) ]
```
where τ is a temperature hyperparameter (default 0.05). This is equivalent to N-way classification where the model must identify the correct positive among N-1 negatives. With batch size N=256, each example has 255 in-batch negatives. The larger the batch, the harder the task and the better the learned representations — this is why MNRL fundamentally rewards large batches, unlike other losses.

```
MNRL Batch Visualization (N=4 pairs)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Anchors  a1    a2    a3    a4
         │     │     │     │
         v     v     v     v
       [Encoder][Encoder][Encoder][Encoder]
         │     │     │     │
         u1    u2    u3    u4     ← anchor embeddings

Positives b1   b2    b3    b4
          └────┴─────┴─────┘
               [Encoder]
         v1    v2    v3    v4    ← positive embeddings

Similarity Matrix (scaled by 1/τ):
     v1    v2    v3    v4
u1 [ 0.92  0.21  0.15  0.18 ]  ← want argmax = v1 ✓
u2 [ 0.19  0.88  0.23  0.11 ]  ← want argmax = v2 ✓
u3 [ 0.13  0.17  0.91  0.20 ]  ← want argmax = v3 ✓
u4 [ 0.16  0.12  0.22  0.87 ]  ← want argmax = v4 ✓

Loss = CrossEntropy(sim_matrix, [0,1,2,3])
```

---

**Q7. ⭐⭐ What is the GradCache trick for MNRL with large batch sizes?**

MNRL requires large batches for best performance, but computing embeddings for a batch of 512 sentence pairs (1024 forward passes) doesn't fit in GPU memory. GradCache (Gao et al. 2021) decouples the gradient computation: (1) Forward pass all sentences in mini-batches (no_grad), caching all embeddings. (2) Compute the full MNRL loss over all cached embeddings (differentiable w.r.t. embeddings only). (3) Backward pass to get gradients w.r.t. embeddings. (4) For each mini-batch, re-run forward pass with gradients enabled, using the cached embedding gradient as the chain rule starting point. This allows an effective batch size of 512+ while using only 32-samples-at-a-time GPU memory. The sentence-transformers library implements this as `GradCacheDataLoader`.

**Follow-up A:** Why does a larger effective batch size improve MNRL?
More in-batch negatives means harder discrimination — the model must distinguish the true positive from 511 negatives instead of 31. This harder task produces stronger gradient signal that forces the model to learn more discriminative embeddings. Empirically, MNRL with batch=256 improves Spearman r by ~3-5% on STS-B compared to batch=32.

**Follow-up B:** What is the temperature hyperparameter τ in MNRL and how do you tune it?
τ scales the cosine similarity before softmax: smaller τ (e.g., 0.02) sharpens the distribution (strong signal, but gradients may vanish for clear positives). Larger τ (e.g., 0.1) softens the distribution (weaker signal but more stable gradients). Default 0.05 works well empirically. Treat it as a hyperparameter: tune on STS-B Spearman r. Some implementations use a learned temperature (like CLIP) that is initialized to log(1/0.07) and updated during training.

**Follow-up C:** Can you use MNRL with hard negatives instead of only in-batch negatives?
Yes — this is a key improvement. Add hard negative passages to each training pair: `(anchor, positive, hard_negative_1, hard_negative_2)`. The loss then includes both in-batch and explicit hard negatives in the denominator. This is what `MultipleNegativesRankingLoss` in sentence-transformers supports via `add_swap_loss=True` and `negative_passages`. Hard negatives from BM25 (lexically similar but semantically different) or ANN (embedding similar but wrong answer) are most valuable.

---

**Q8. ⭐ What is the difference between TripletLoss variants: margin-based vs soft margin?**

**Hard margin TripletLoss**: `L = max(0, margin + d(anchor, pos) - d(anchor, neg))`. Training stops providing gradient when `d(anchor, neg) - d(anchor, pos) > margin`. Problem: most triplets quickly become "easy" (already satisfied the margin) and provide no gradient. **Soft margin (batch-hard) TripletLoss**: instead of a fixed margin, use softplus: `L = log(1 + exp(d(anchor, pos) - d(anchor, neg)))`. Provides non-zero gradient for all triplets. **Batch-hard mining**: within each batch, for each anchor, select the hardest positive (most distant positive) and hardest negative (closest negative) to form triplets. Batch-hard mining significantly outperforms random triplet selection.

---

### 1.3 Hard Negative Mining

**Q9. ⭐ What is hard negative mining and why does it matter?**

Hard negatives are sentences that are lexically or semantically close to the query but have different meanings — they are the most challenging negatives for the model to distinguish. **Random negatives**: any sentence from the corpus; most are clearly unrelated, providing little training signal. **BM25 negatives**: retrieve top BM25 results excluding the positive — lexically similar but semantically wrong. **ANN negatives** (dynamic/online): use the current model checkpoint to find nearest neighbors that are not positive — semantically confusable. Training exclusively on random negatives produces an embedding model that fails on hard cases. Adding BM25 negatives improves retrieval MRR@10 by ~5%; adding ANN negatives gives another ~3-5% improvement. The MNRL paper shows this is the single most impactful training data quality improvement.

---

**Q10. ⭐⭐ Compare BM25 negatives vs ANN negatives for hard negative mining.**

**BM25 negatives**: retrieved using keyword matching — hard at a lexical level (share same words/phrases) but potentially easy at a semantic level (model might already distinguish them well). Easy to generate offline. **ANN negatives**: retrieved using current model embeddings — semantically hard (near the decision boundary in embedding space). Must be regenerated as the model improves (dynamic). **False negative filtering**: both methods can retrieve actual positives (relevant documents that weren't labeled) — use deduplication and confidence filtering. **Pipeline**: (1) Offline: BM25 mine all pairs → train for 1 epoch. (2) Online: generate ANN embeddings for full corpus → mine ANN negatives → continue training. This "two-phase" approach is used by training recipes for state-of-the-art models like E5 and GTE.

**Follow-up A:** What is a false negative and how do you handle it in hard negative mining?
A false negative is a retrieved sentence that is actually semantically similar to the query but wasn't labeled as positive — treating it as a negative creates a contradictory training signal. Handling strategies: (1) **Score threshold**: only use BM25/ANN results below a certain similarity threshold as negatives; (2) **NLI filter**: run a cross-encoder on query-candidate pairs, discard candidates with cross-encoder score > 0.8 as likely false negatives; (3) **Deduplication**: use MinHash or near-duplicate detection on the corpus first.

**Follow-up B:** What is cross-encoder knowledge distillation for hard negative mining?
After mining hard negatives with BM25 or ANN, use a cross-encoder teacher to score all (query, candidate) pairs and produce soft labels. Train the bi-encoder student using KL divergence between teacher scores and student cosine similarities. This is the recipe used by MSMARCO-trained models — the cross-encoder provides better training signal than binary positive/negative labels because it captures gradations of relevance.

---

### 1.4 & 1.5 STS Evaluation and MTEB

**Q11. ⭐ What is STS-B and why is Spearman r used over Pearson r?**

STS-B (Semantic Textual Similarity Benchmark) consists of ~8,628 sentence pairs each annotated with a human similarity score from 0-5. The standard evaluation: normalize scores to 0-1, compute cosine similarity between sentence embeddings, report Spearman rank correlation between model scores and human scores. **Spearman r** measures rank correlation (monotonic relationship) — it doesn't require the relationship to be linear, only that higher model scores correspond to higher human scores. **Pearson r** measures linear correlation — if the model outputs cosine similarities between 0.95 and 1.0 but humans rated 0 to 5, the scale mismatch penalizes Pearson even if the ranking is perfect. Since we care about relative ordering (not calibrated absolute similarity), Spearman is more appropriate.

---

**Q12. ⭐ What is the MTEB benchmark and what tasks does it cover?**

MTEB (Massive Text Embedding Benchmark) is the most comprehensive benchmark for sentence embedding evaluation, covering 56 datasets across 8 task categories:
- **Retrieval** (15 datasets): BEIR benchmark tasks, MSMARCO — information retrieval scenarios
- **Clustering** (11 datasets): ArXiv, Reddit clustering — grouping similar texts
- **STS** (10 datasets): STS-B, SICK, etc. — semantic similarity correlation
- **Classification** (12 datasets): sentiment, topic — linear probe on embeddings
- **Reranking** (4 datasets): MRR on reranking candidates
- **PairClassification** (3 datasets): NLI-style entailment detection
- **Summarization** (1 dataset): summary-document correlation
- **Bitext Mining** (1 dataset): cross-lingual alignment

A model's MTEB rank is a holistic measure of embedding quality across diverse tasks. Top models as of 2024: E5-mistral-7b, GTE-Qwen, text-embedding-3-large.

---

**Q13. ⭐⭐ What is the anisotropy problem in transformer embeddings and how does whitening fix it?**

Transformer embeddings (especially from vanilla BERT, not fine-tuned with contrastive objectives) cluster in a narrow cone in vector space — the embedding distribution is anisotropic (has very different variance along different dimensions). This means cosine similarity between random sentence pairs is abnormally high (often 0.8-0.99) because all embeddings point in approximately the same direction. Anisotropy makes raw BERT embeddings poor for similarity tasks. **Whitening** transforms the embedding space to have zero mean and identity covariance: `z = W(h - μ)` where μ is the mean embedding and W is the whitening matrix computed from the covariance matrix. After whitening, the distribution is isotropic (uniform variance across all dimensions), and cosine similarity becomes more discriminative. Contrastive fine-tuning (MNRL, TripletLoss) implicitly reduces anisotropy by training the model to spread representations across the full space.

**Follow-up A:** How does sentence-transformers' MNRL training address anisotropy?
MNRL explicitly pushes non-positive pairs apart in embedding space. With in-batch negatives, the loss penalizes high cosine similarity between any non-paired embeddings in the batch. Over training, this creates pressure to use the full hypersphere for embedding distribution — reducing the clustering behavior. Empirically, a BERT model fine-tuned with MNRL has much more uniform embedding distribution than the original BERT, as measurable by the average cosine similarity between random sentence pairs (drops from ~0.90 to ~0.30).

**Follow-up B:** What is the difference between L2 normalization and whitening?
L2 normalization projects all embeddings to the unit hypersphere (`z = h / ||h||`), which enables cosine similarity = dot product and is required for efficient inner product search (FAISS). But it doesn't address the cone problem — all normalized embeddings still cluster in the same region of the sphere. Whitening additionally addresses the variance structure, making the distribution uniform on the sphere. In practice, fine-tuned sentence transformers with L2 normalization work well enough that whitening is rarely applied in production systems.

---

### 1.6 Embedding Space Analysis

**Q14. ⭐ How do you use UMAP to analyze the quality of sentence embeddings?**

UMAP (Uniform Manifold Approximation and Projection) reduces high-dimensional embeddings (768-dim) to 2D/3D for visualization. In this project, `umap-learn` is used to project sentence embeddings and color-code by label/class. A well-trained embedding model shows: clearly separated clusters for different semantic categories, smooth interpolation within clusters, no major outliers in wrong clusters. Analysis workflow: (1) Encode a balanced sample of test sentences with the model. (2) `reducer = umap.UMAP(n_components=2, metric='cosine', random_state=42)`. (3) `embedding_2d = reducer.fit_transform(sentence_embeddings)`. (4) Plot with matplotlib/plotly colored by label. (5) Compare UMAP before and after fine-tuning to visually confirm clustering improvement.

---

**Q15. ⭐ What is Matryoshka Representation Learning (MRL)?**

MRL trains a single embedding model to produce representations that work at any prefix dimensionality. During training, the loss is computed at multiple scales simultaneously: `L_MRL = Σ_m λ_m * L(embeddings[:m])` for m ∈ {768, 512, 256, 128, 64}. The model learns to pack the most important information in the first dimensions, with each additional dimension providing incrementally useful information. At inference time, you can truncate to 64 dimensions for fast approximate search, or use all 768 for maximum quality. This is implemented in the `sentence-transformers` library via `MatryoshkaLoss`. The OpenAI text-embedding-3 models use MRL, allowing users to specify output dimensions from 256 to 3072.

---

## 2. System Design Discussions

**Q16. ⭐⭐ Design a semantic search system using sentence embeddings at 100M document scale.**

The system has four main components with a two-stage retrieval pipeline.

```
Semantic Search Architecture (100M documents)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [Document Ingestion Pipeline]
  New doc → bi-encoder → 384-dim float32 embedding
         → quantize to int8 → store in FAISS index + metadata DB

  [FAISS Index Structure]
  FAISS IVF+PQ index:
  - 100M vectors × 384-dim × 4 bytes = ~150GB raw
  - After IVF+PQ compression: ~15GB (10x compression)
  - Fits in single A100 GPU memory

  [Query-Time Pipeline]
            Query text
               │
               ▼
        [Bi-encoder]     ~3ms
        query embedding
               │
               ▼
        [FAISS ANN Search] ~5ms
        Top-1000 candidates
               │
               ▼
        [Cross-encoder Reranker] ~50ms
        Top-10 results with scores
               │
               ▼
           Response

  [Caching Layer]
  Redis: cache (query_hash → top-10 results), TTL=1hr
  Cache hit rate for popular queries: ~40%
  Effective p99 latency with cache: ~15ms
```

**Follow-up A:** How do you handle index updates when new documents arrive?
Options: (1) **Batch rebuild**: rebuild the FAISS index nightly from all documents — simple but has a gap window. (2) **Incremental index**: maintain a small "live" index for new documents (FAISS flat index, exact search), merge into main IVF index during low-traffic windows. (3) **Two-tier**: main IVF index + live flat index, merge periodically. (4) **Elasticsearch dense vector** with HNSW indexing supports real-time document addition without full rebuilds. The choice depends on update frequency and acceptable index lag.

**Follow-up B:** What is HNSW and how does it compare to IVF+PQ for approximate nearest neighbor search?
HNSW (Hierarchical Navigable Small World) is a graph-based ANN algorithm with O(log N) search complexity, high recall (>99% at default settings), supports incremental insertions, but high memory (no compression). IVF+PQ is an inverted file index with product quantization — IVF partitions vectors into Voronoi cells (fast coarse search), PQ compresses within cells (10-100x memory reduction). HNSW is preferred when: memory is not constrained, incremental updates needed, maximum recall is important. IVF+PQ is preferred when: 100M+ vectors require compression, memory budget is tight.

---

**Q17. ⭐ Design a domain adaptation pipeline for sentence embeddings on a new domain (e.g., legal documents).**

Standard pretrained models (trained on Wikipedia, CommonCrawl) perform poorly on specialized domains (legal, medical, scientific) because the vocabulary, writing style, and semantic relationships differ. Pipeline: (1) **Unsupervised TSDAE** (Transformer-based Sequential Denoising Auto-Encoder): corrupt sentences by deleting words, train the model to reconstruct originals — adapts language understanding without labeled data. (2) **GPL** (Generative Pseudo Labeling): generate query-document pairs using a T5 model, score with a cross-encoder teacher, use MNRL with these pseudo-labeled pairs. (3) **Supervised fine-tuning**: if domain-specific labeled pairs exist (case law relevance judgments), fine-tune with MNRL directly. (4) Evaluate on domain-specific STS pairs before and after adaptation.

---

## 3. Coding & Implementation Questions

**Q18. ⭐ Walk through the `load_nli_pairs` function in data_loader.py.**

The function loads SNLI (Stanford NLI) and MultiNLI datasets from HuggingFace datasets hub. NLI has 3 labels: entailment (0), neutral (1), contradiction (2). For MNRL training, only **entailment** pairs are used as positive pairs: `(premise, hypothesis)` where label=0. Contradictions are used as explicit hard negatives in some implementations. The function returns a list of `InputExample(texts=[premise, hypothesis], label=1.0)` for entailment pairs. The combined SNLI+MNLI training set has ~983k entailment pairs — a large, diverse source of positive pairs covering many topics from the NLI benchmark corpora.

---

**Q19. ⭐ How does STS-B normalization work in load_stsb_pairs?**

STS-B scores are human-annotated similarity ratings from 0 to 5. For CosineSimilarityLoss training, these must be normalized to [0, 1]: `normalized_score = raw_score / 5.0`. For evaluation with `STSEvaluator`, the raw cosine similarities (from the model) are compared to the normalized scores using Spearman r. Note: the normalization is a monotonic transformation, so Spearman r is unchanged by it. However, if you use CosineSimilarityLoss for training, you must use normalized labels so the MSE loss targets are in the [0,1] range matching cosine similarity output.

---

**Q20. ⭐⭐ Explain the SentenceTransformerTrainer and `model.fit()` workflow.**

In sentence-transformers 3.0+, training uses the `SentenceTransformerTrainer` class which wraps the `Trainer` from HuggingFace Transformers. Setup:

```python
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

model = SentenceTransformer('distilbert-base-uncased')
loss = MultipleNegativesRankingLoss(model)

args = SentenceTransformerTrainingArguments(
    output_dir='./output',
    num_train_epochs=1,
    per_device_train_batch_size=256,
    warmup_ratio=0.1,
    fp16=True,
    eval_strategy='steps',
    eval_steps=500,
)
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=evaluator,
)
trainer.train()
```

The legacy `model.fit()` API still works in 3.0 but wraps this same trainer internally.

**Follow-up A:** How do you combine multiple losses in sentence-transformers training?
Use a dictionary mapping loss names to (loss_function, dataset) tuples, then pass to trainer. The trainer alternates batches from each dataset and accumulates losses. Example: combine MNRL on NLI pairs with CosineSimilarityLoss on STS-B pairs — the MNRL provides contrastive signal while CosineSimilarityLoss fine-tunes the cosine calibration.

**Follow-up B:** What is the `add_swap_loss=True` parameter in MNRL?
When enabled, MNRL also treats `(b_i, a_i)` as positive pairs — swapping anchor and positive. This doubles the number of positive pairs per batch without additional data, and adds in-batch negatives in both directions. For symmetric similarity tasks (STS), this is beneficial. For asymmetric tasks (query-document retrieval), it can hurt because query and document embeddings should not be interchangeable.

---

**Q21. ⭐ Explain how compare_models works in model_utils.py.**

`compare_models(model_a, model_b, eval_dataset)` encodes the same set of sentence pairs with both models, computes Spearman r against STS-B gold scores for each, and returns a comparison dict: `{'model_a_spearman': 0.872, 'model_b_spearman': 0.891, 'improvement': 0.019}`. The function also optionally computes per-task breakdown if MTEB tasks are provided. The comparison is used in the deployment gate: only push model_b to hub if `improvement > min_improvement_threshold` (configurable, default 0.005 Spearman r).

---

**Q22. ⭐ How does `push_to_hub` work in model_utils.py?**

`model.push_to_hub(repo_id, private=True)` uploads the model to HuggingFace Hub. What gets pushed: `pytorch_model.bin` (weights), `config.json`, `tokenizer_config.json`, `tokenizer.json`, `sentence_bert_config.json` (pooling strategy), `modules.json` (sentence-transformers module pipeline), and `README.md` (auto-generated model card). The model card includes evaluation results if `model.push_to_hub(..., eval_results=eval_results)` is called. HuggingFace Hub versioning uses git LFS for large model files — each push creates a new commit. To version semantically, use the `commit_message` parameter.

---

## 4. Common Bugs & Issues

| # | Bug / Issue | Symptom | Root Cause | Fix |
|---|---|---|---|---|
| 1 | Incorrect pooling for model | Low Spearman r despite training | Using CLS pooling instead of mean pooling for BERT-based model | Check `sentence_bert_config.json` pooling_mode |
| 2 | Not L2 normalizing before cosine similarity | Cosine sim values incorrect | Forgetting to normalize embeddings | `model.encode(sentences, normalize_embeddings=True)` |
| 3 | Including neutral NLI pairs in MNRL | Training noise, lower performance | Neutral pairs are not positives, mislead contrastive loss | Filter to label=0 (entailment) pairs only |
| 4 | STS scores not normalized | Loss training with 0-5 labels vs cosine 0-1 target | Using raw STS scores with CosineSimilarityLoss | Divide by 5.0 before creating InputExamples |
| 5 | Batch size too small for MNRL | Low Spearman r, slower convergence | Only 31 in-batch negatives with batch_size=32 | Increase to batch_size=256+, use GradCache if OOM |
| 6 | False negatives in batch | Contradictory training signal | Two semantically similar sentences in same batch, both treated as negatives | Add `negative_examples` filter or use deduplication |
| 7 | Model not in eval mode during STSEvaluator | Dropout active, irreproducible eval | Missing `model.eval()` | sentence-transformers evaluator handles this automatically |
| 8 | Wrong temperature τ | Loss doesn't decrease, NaN | τ too small causes overflow in exp() | Start with τ=0.05, clip cosine similarities to [-1,1] |
| 9 | Embedding cache stale after model update | ANN search returns wrong results | Hard negative mining uses old model embeddings | Regenerate embeddings after each training checkpoint |
| 10 | UMAP with euclidean metric on non-normalized embeddings | Misleading visualization | Euclidean distance on raw embeddings | Use `metric='cosine'` in UMAP for sentence embeddings |
| 11 | push_to_hub fails on model card | Upload error or missing metadata | Repo not initialized with README | Create repo first: `HfApi().create_repo(repo_id)` |
| 12 | Spearman r on wrong test set | Optimistic eval results | Evaluating on STS-B dev set used during training | Report results on STS-B test set |
| 13 | OOM with batch_size=512 and long sequences | CUDA OOM | Attention memory is O(seq_len^2) | Use max_seq_length=128 for NLI (most pairs are short) |
| 14 | symmetric vs asymmetric similarity mismatch | Retrieval performance lower than expected | `add_swap_loss=True` for query-doc asymmetric task | Set `add_swap_loss=False` for retrieval tasks |
| 15 | accelerate DeepSpeed Z3 offloads embeddings | Slow training, CPU memory spikes | Embedding parameters offloaded to CPU | Use Z2 (gradient offload) not Z3 for sentence transformers |
| 16 | scipy version mismatch | Spearman r computation error | `scipy.stats.spearmanr` API change | Pin `scipy>=1.9.0` in requirements |
| 17 | Duplicate sentence pairs in NLI | Over-fitting to specific pairs | SNLI+MNLI overlap on identical premises | Deduplicate training set by (premise, hypothesis) hash |
| 18 | Wrong units for Spearman r | Reporting 0-100 scale vs 0-1 scale | Multiplying Spearman r by 100 for percentage display | Clarify units: "Spearman r = 0.872" not "87.2%" |

---

## 5. Deployment — Azure

**Q23. ⭐ Describe the Azure deployment architecture for a sentence embedding microservice.**

```
Azure Sentence Embedding Service Architecture
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [Azure Blob Storage]
  ├── training-data/nli-pairs/
  ├── training-data/stsb/
  └── model-artifacts/sentence-embedder-v2/

         │
         ▼
  [Azure ML Compute]
  ┌────────────────────────────┐
  │  GPU: Standard_ND96asr_A100│
  │  (8x A100 80GB per node)   │
  │  MNRL training on NLI+STS  │
  │  SentenceTransformerTrainer│
  └──────────┬─────────────────┘
             │ model artifacts
             ▼
  [Azure ML Model Registry]
  ├── sentence-embedder v2.1
  ├── Spearman_r: 0.891
  └── MTEB avg: 0.637

             │ approved model
             ▼
  [Azure Container Registry]
  └── embed-service:v2.1

             │
             ▼
  [Azure Container Apps]           [Azure Cache for Redis]
  ┌──────────────────────┐               │
  │  /embed endpoint      │◄──────────────┤ cache popular
  │  Input: text[]        │               │ embeddings
  │  Output: float[][]    │               │ TTL: 24h
  │  ONNX quantized model │───────────────┘
  │  Batch: up to 64      │
  └──────────┬───────────┘
             │
             ▼
  [Azure AI Search]          ← downstream vector index consumer
  └── vector search index
      (768-dim or 384-dim)

  [Application Insights]
  ├── Embedding latency p50/p99
  ├── Cache hit rate
  └── Daily Spearman r monitoring
```

---

**Q24. ⭐⭐ How do you implement embedding caching in Azure Redis for the embedding microservice?**

Cache key = SHA256 hash of (model_version + normalized_text). Cache value = base64-encoded float32 embedding. TTL = 24 hours for general text, 7 days for highly stable content (product descriptions). Implementation:

```python
import redis, hashlib, base64, numpy as np

r = redis.StrictRedis(host=REDIS_HOST, port=6380, ssl=True,
                      password=REDIS_KEY)

def get_or_compute_embedding(text: str, model_version: str) -> np.ndarray:
    key = hashlib.sha256(f"{model_version}:{text}".encode()).hexdigest()
    cached = r.get(key)
    if cached:
        return np.frombuffer(base64.b64decode(cached), dtype=np.float32)
    embedding = model.encode(text, normalize_embeddings=True)
    r.setex(key, 86400,  # TTL 24h
            base64.b64encode(embedding.tobytes()))
    return embedding
```

Cache invalidation on model update: use `model_version` in the key — old embeddings become inaccessible automatically. For batch invalidation: use Redis SCAN + pattern matching on the old version prefix. Monitor cache hit rate — target >40% for a query-heavy search system.

---

**Q25. ⭐ How does Azure AI Search integrate with the sentence embedding service?**

Azure AI Search (formerly Cognitive Search) supports vector search with Hierarchical Navigable Small World (HNSW) indexing via the `VectorSearch` configuration. Workflow: (1) When documents are added, call the embedding service to get vectors and store them with the document in Azure Search. (2) At query time, embed the query using the same service, then call Azure Search with `VectorQuery(vector=query_embedding, k_nearest_neighbors=50, fields='content_vector')`. (3) Optionally combine with BM25 keyword search for hybrid search (RRF fusion). (4) Pass top-50 to a cross-encoder reranker for final ranking. Azure AI Search supports both L2 and cosine distance metrics; use cosine for normalized embeddings.

---

## 6. Deployment — AWS

**Q26. ⭐ Describe the AWS deployment architecture for the sentence embedding service.**

```
AWS Sentence Embedding Service Architecture
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [Amazon S3]
  ├── s3://bucket/training-data/nli/
  ├── s3://bucket/training-data/stsb/
  └── s3://bucket/model-artifacts/

         │
         ▼
  [SageMaker Training Job]
  ┌────────────────────────────────┐
  │  Instance: ml.p3dn.24xlarge    │
  │  (8x V100 32GB)                │
  │  sentence_transformers trainer │
  │  SageMaker Experiments logging │
  └──────────────┬─────────────────┘
                 │
                 ▼
  [SageMaker Model Registry]
  └── sentence-embedder ModelPackage

                 │ approved
                 ▼
  [SageMaker Endpoint]              [ElastiCache Redis]
  ┌──────────────────────────┐           │
  │  ml.g4dn.2xlarge (T4)    │◄──────────┤ embedding cache
  │  ONNX optimized model    │           │ TTL: 24h
  │  multi-container: tok+inf│───────────┘
  └──────────┬───────────────┘
             │
             ▼
  [Amazon OpenSearch Service]    ← k-NN vector index
  └── knn_vector field, cosine
      HNSW: m=16, ef_construction=512

  [Kinesis Data Streams]
  └── real-time embedding requests for:
      - drift monitoring
      - A/B test logging
      - async index updates

  [CloudWatch]
  ├── InvocationLatency p99
  ├── ModelLatency
  └── Custom metrics: Spearman_r (weekly)
```

---

**Q27. ⭐ How do you export a sentence-transformers model to ONNX for faster inference on AWS?**

```python
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import torch, onnx

model = SentenceTransformer('output/my-model')

# Export encoder + pooling as single ONNX graph
with torch.no_grad():
    dummy_input = {
        'input_ids': torch.zeros(1, 128, dtype=torch.long),
        'attention_mask': torch.ones(1, 128, dtype=torch.long)
    }
    torch.onnx.export(
        model[0].auto_model,  # transformer
        (dummy_input['input_ids'], dummy_input['attention_mask']),
        'encoder.onnx',
        input_names=['input_ids', 'attention_mask'],
        output_names=['last_hidden_state'],
        dynamic_axes={'input_ids': {0: 'batch', 1: 'seq'},
                      'attention_mask': {0: 'batch', 1: 'seq'}},
        opset_version=14
    )
```

Then apply ONNX Runtime optimization: `sess = ort.InferenceSession('encoder.onnx', providers=['CUDAExecutionProvider'])`. For INT8 quantization: use `ort.quantization.quantize_dynamic`. ONNX Runtime typically gives 1.4-2x speedup over PyTorch on CPU inference.

---

**Q28. ⭐⭐ How do you use Amazon OpenSearch's k-NN vector search for semantic retrieval?**

Create an OpenSearch index with k-NN configuration:
```json
{
  "settings": {
    "knn": true,
    "knn.algo_param.ef_search": 512
  },
  "mappings": {
    "properties": {
      "content_vector": {
        "type": "knn_vector",
        "dimension": 384,
        "method": {
          "name": "hnsw",
          "space_type": "cosinesimil",
          "engine": "faiss",
          "parameters": {"ef_construction": 512, "m": 16}
        }
      }
    }
  }
}
```
Query: `{"query": {"knn": {"content_vector": {"vector": query_embedding, "k": 100}}}}`. For hybrid search (semantic + keyword): use `bool` query combining `knn` with `match` query, use reciprocal rank fusion for result merging. OpenSearch's HNSW index updates in near-real-time as documents are indexed, no rebuild needed.

**Follow-up A:** What are the ef_search and m HNSW parameters and how do they affect performance?
`m` (max connections per node): higher m = more edges in the graph = better recall but higher memory and index build time. `m=16` is a good default. `ef_construction`: number of neighbors checked during index build — higher = better index quality but slower build. `ef_search`: number of neighbors checked at query time — trade-off between recall and latency. Set `ef_search=100` for >99% recall at ~5ms per query on 1M vectors.

**Follow-up B:** How does hybrid search (semantic + keyword) work in OpenSearch?
OpenSearch 2.10+ supports hybrid search with normalization. Keyword BM25 score and semantic k-NN score are normalized to [0,1] independently, then combined: `final_score = α * bm25_score + (1-α) * knn_score`. The weight α is tuned on labeled test data — typically α=0.4 works well (semantic weighted more). Reciprocal Rank Fusion (RRF) is an alternative that's rank-based and doesn't require score normalization.

---

## 7. Post-Production Issues

| # | Issue | Detection | Root Cause | Resolution |
|---|---|---|---|---|
| 1 | Embedding drift after model update | Cache hit rate drops, downstream accuracy regresses | New model produces different embedding space | Versioned cache keys, rebuild vector index after model update |
| 2 | Cosine similarity distribution shift | Mean similarity drops 0.3→0.1 after update | New model more discriminative (good) or anisotropy fix | Monitor distribution with daily histogram, retune thresholds |
| 3 | Domain mismatch degradation | Spearman r on internal STS drops 0.89→0.72 | General-domain model deployed on specialized content | Fine-tune on domain data via TSDAE + GPL, re-evaluate |
| 4 | Quantization accuracy loss | INT8 ONNX model Spearman r drops 3% | Outlier activations clipped by INT8 quantization | Use FP16 instead of INT8, or calibrate quantization on domain data |
| 5 | Cache invalidation race condition | Stale embeddings served after model rollout | Redis TTL not expired, new model live | Flush cache by model version prefix on deployment |
| 6 | Near-duplicate vectors in index | Search returns redundant results | Duplicate documents indexed multiple times | Dedup by document hash before indexing |
| 7 | Long text truncation causes embedding instability | Embeddings for related long documents diverge | Different truncation artifacts per sentence | Standardize max_seq_length=512, use first 512 tokens consistently |
| 8 | OOM during batch encoding | Kubernetes OOM kill on embedding service | Large batch requests (1000+ sentences) without limit | Cap batch size at 64, implement request splitting |
| 9 | Spearman r metric regression undetected | Quality degrades silently | No automated weekly evaluation | Schedule weekly Spearman r evaluation on labeled STS sample |
| 10 | Cross-lingual embedding mismatch | Multilingual retrieval fails | English-only model deployed for multilingual content | Switch to multilingual-e5-base or paraphrase-multilingual |
| 11 | FAISS index rebuild takes 8+ hours | Downtime during index rebuild | Full rebuild on 100M doc corpus | Use incremental HNSW (OpenSearch/Weaviate) instead of FAISS IVF |
| 12 | Redis memory overflow | Cache evictions spike, hit rate drops | Embedding cache grows unboundedly | Set Redis maxmemory policy to allkeys-lru, monitor used_memory |
| 13 | Asymmetric similarity for symmetric task | User reports reversed relevance | Using asymmetric query/doc encoders for STS | Use symmetric model for STS tasks |
| 14 | Embedding service SLA breach during traffic spike | p99 latency > 500ms | Single-instance auto-scaling lag | Set min-replicas=2, pre-scale before known traffic events |
| 15 | Model card outdated after fine-tuning | Hub README shows wrong benchmark numbers | push_to_hub without updated eval_results | Always pass updated eval_results to push_to_hub |
| 16 | L2 normalization missing in production | Cosine similarity values wrong (>1) | Forgot `normalize_embeddings=True` | Add normalization check in API input validation |

---

## 8. General ML Interview Topics

**Q29. ⭐ Explain representation learning and how sentence embedding training exemplifies it.**

Representation learning is the paradigm of learning feature representations from data rather than engineering features manually. In sentence embedding training, the goal is to learn a mapping `f: text → R^d` such that the geometry of R^d reflects semantic structure — similar meanings map to nearby points. This is representation learning applied to discrete symbolic text. The learned representation is useful across many downstream tasks (retrieval, classification, clustering) without task-specific retraining. The contrastive training signal (MNRL) is the supervision that shapes the geometry of the embedding space. The evaluation (STS, MTEB) measures how well the geometry reflects human semantic judgments.

---

**Q30. ⭐⭐ How does SimCLR (contrastive self-supervised learning for vision) relate to sentence embedding training?**

SimCLR (Chen et al. 2020) trains image encoders by: (1) creating two augmented views of each image; (2) encoding both with a Siamese network; (3) using NT-Xent loss (equivalent to MNRL/InfoNCE) to maximize similarity between views of the same image vs. all other images in the batch. This is structurally identical to sentence embedding training with MNRL: two sentences with the same meaning (entailment pair) = two augmented views of the same "concept." The difference is data augmentation: for images, augmentation is random crop/flip/color jitter; for text, it is paraphrase/NLI pairs. Both rely on the same principle that in-batch negatives work better with larger batches. SimCLR's key contribution — the projection head (an MLP appended during training, discarded at inference) — is also used in some sentence embedding setups.

---

**Q31. ⭐ What is dimensionality reduction and when do you use UMAP vs PCA vs t-SNE?**

All three reduce high-dimensional embeddings to 2D/3D for visualization:

| Method | Preserves | Speed | Deterministic | When to Use |
|---|---|---|---|---|
| PCA | Global structure, variance | Very fast | Yes | First-pass exploration, understanding principal components |
| t-SNE | Local cluster structure | Slow O(N^2) | No (with seed) | Small datasets (<10k), beautiful cluster visualization |
| UMAP | Both local and global | Fast O(N log N) | No (with seed) | Large datasets, production quality visualization, preserving topology |

For sentence embeddings: UMAP with `metric='cosine'` is the standard choice for its speed and better global structure preservation. t-SNE is good for showing tight clusters but distorts inter-cluster distances. Use `random_state=42` for reproducibility.

---

**Q32. ⭐⭐ What is metric learning and how does contrastive sentence embedding training fit into it?**

Metric learning (or distance metric learning) is the task of learning a distance function `d(x, y)` such that semantically similar items are close and dissimilar items are far. Traditional metric learning methods: (1) Large Margin Nearest Neighbor (LMNN): pull same-class examples together, push different-class examples outside a margin; (2) Neighborhood Components Analysis (NCA); (3) Siamese networks with contrastive loss (Chopra et al. 2005). Sentence embedding training is metric learning applied to text, using cosine distance in a high-dimensional space. The MNRL loss is the modern version of contrastive loss, scaled to use in-batch negatives instead of pre-specified pairs. The key connection: well-trained sentence embeddings enable k-NN classification (no separate classifier needed) because the metric space directly encodes semantic relationships.

**Follow-up A:** What is the difference between metric learning and feature learning?
In metric learning, you explicitly optimize for a distance/similarity criterion between pairs/triplets. In feature learning (autoencoder, masked LM), you optimize for reconstruction or prediction — the metric is not directly optimized. Sentence embedding training is metric learning: MNRL directly optimizes cosine similarity ranking. In contrast, BERT pretraining (MLM) is feature learning: BERT is not trained to produce good cosine similarities between sentence pairs.

**Follow-up B:** How would you evaluate if the learned metric generalizes to unseen classes?
Zero-shot evaluation: train the embedding model on NLI pairs (covering some semantic categories), then evaluate on STS tasks covering different categories without fine-tuning. Compare Spearman r on in-distribution vs. out-of-distribution STS pairs. MTEB is specifically designed for this: models trained on common NLI/paraphrase data are evaluated on domain-specific retrieval tasks (medical, legal, scientific) they weren't trained on.

---

## 9. Behavioral / Scenario Questions

**Q33. ⭐ Describe a scenario where cosine similarity as the metric would fail and what you would use instead.**

Scenario: recommending code snippets for natural language queries. Two Python functions that solve the same problem (sorting a list) might have very different variable names and structure, making them look lexically different. But more importantly, syntactic code structure isn't well-captured by a natural language sentence encoder. Cosine similarity on standard sentence embeddings might rank syntactically similar but semantically wrong code higher. Better approaches: (1) Use a code-specific embedding model (CodeBERT, UniXcoder) trained on code-comment pairs; (2) Use BM25 on function docstrings/comments + code embeddings for hybrid retrieval; (3) Evaluate with MRR@10 on a code retrieval benchmark (CodeSearchNet) rather than generic STS correlation.

---

**Q34. ⭐⭐ Your sentence embedding model's Spearman r drops from 0.89 to 0.74 after deploying to production. What do you do?**

Systematic investigation: (1) **Check preprocessing**: compare text preprocessing pipeline between evaluation and production — any differences in lowercasing, punctuation removal, language filtering? (2) **Check model version**: confirm production service is running the correct model checkpoint, not an older one. (3) **Check input distribution**: are production texts significantly longer, shorter, or different language distribution than STS-B evaluation texts? (4) **Compute embedding distribution**: generate embeddings for a sample of production texts vs. STS-B texts, compare cosine similarity distributions. (5) **Domain analysis**: STS-B is general domain — if production is legal or medical text, distribution shift explains the drop. Resolution: collect domain-specific labeled pairs (if possible), fine-tune with TSDAE + GPL, re-evaluate.

---

**Q35. ⭐ How would you explain Spearman rank correlation to a product manager?**

"We evaluate our sentence embedding model by comparing how it ranks sentence similarity against how humans rank the same pairs. If a human judges pair A as more similar than pair B, our model should also give pair A a higher similarity score. The Spearman score measures how often the model agrees with human ranking — 1.0 means perfect agreement, 0 means random, and -1.0 means perfectly inverted. Our model achieves 0.89, meaning it agrees with human judgment on ~94% of comparisons. The state of the art is ~0.93. The 0.04 gap translates to approximately 6% of cases where the model ranks similarity differently than a human would — acceptable for our search use case, where exact ranking of closely-similar items matters less than surfacing clearly relevant results."

---

## 10. Quick-Fire Questions

1. ⭐ What is a bi-encoder? — Encodes each sentence independently, similarity = cosine(emb_A, emb_B)
2. ⭐ What is a cross-encoder? — Encodes sentence pair jointly, produces a single relevance score
3. ⭐ Why is bi-encoder faster for retrieval? — Sentence embeddings can be pre-computed and indexed; O(1) query lookup vs O(N) cross-encoder
4. ⭐ What does MNRL stand for? — MultipleNegativesRankingLoss
5. ⭐ How does MNRL use in-batch negatives? — All other positive pairs in the batch serve as negatives for each anchor
6. ⭐ Why do larger batches improve MNRL? — More in-batch negatives = harder task = stronger gradient signal
7. ⭐ What pooling strategy is default in sentence-transformers? — Mean pooling over all non-padding tokens
8. ⭐ Why mean pooling over CLS pooling? — Mean pooling empirically outperforms CLS for semantic similarity; CLS is optimized for classification, not similarity in pretrained BERT
9. ⭐ What dataset is used for STS evaluation? — STS-B (Semantic Textual Similarity Benchmark)
10. ⭐ Why Spearman r over Pearson r for STS evaluation? — Spearman measures rank correlation, more robust to scale differences between model output and human scores
11. ⭐ What is anisotropy in transformer embeddings? — Embeddings cluster in a narrow cone, making random pairs have artificially high cosine similarity
12. ⭐ What does whitening do to embeddings? — Transforms to zero mean, identity covariance — makes distribution isotropic
13. ⭐ What is MTEB? — Massive Text Embedding Benchmark: 56 datasets, 8 task categories
14. ⭐ What NLI datasets are used in this project? — SNLI (~550k) and MultiNLI (~433k)
15. ⭐ What label type from NLI is used as positive pairs? — Entailment pairs (label=0)
16. ⭐ What are hard negatives? — Semantically confusable negatives, near the decision boundary
17. ⭐ What is BM25 hard negative mining? — Use BM25 to retrieve lexically similar but semantically wrong documents as negatives
18. ⭐ What is ANN hard negative mining? — Use current model embeddings + ANN search to find semantically similar but wrong documents
19. ⭐ What is knowledge distillation in the context of sentence embeddings? — Train bi-encoder student to match cross-encoder teacher's relevance scores
20. ⭐ What is GradCache? — Technique to use large effective batch sizes for MNRL without OOM by decoupling forward/backward passes
21. ⭐ What tool is used for 2D embedding visualization? — UMAP (umap-learn)
22. ⭐ What distance metric should UMAP use for sentence embeddings? — `metric='cosine'`
23. ⭐ What is MRL? — Matryoshka Representation Learning: training for variable-dimension truncatable embeddings
24. ⭐ What is TSDAE? — Transformer-based Sequential Denoising Autoencoder for unsupervised domain adaptation
25. ⭐ What is GPL? — Generative Pseudo Labeling: query generation + cross-encoder scoring for domain adaptation
26. ⭐⭐ What is the temperature τ in MNRL? — Scaling factor for cosine similarities before softmax, controls sharpness
27. ⭐⭐ What are false negatives in contrastive learning? — Samples that are actually positive but treated as negatives
28. ⭐⭐ What is the projection head in SimCLR? — MLP head added during training and discarded at inference, improves representation quality
29. ⭐⭐ What is NT-Xent loss? — Normalized Temperature-scaled Cross-Entropy: SimCLR's contrastive loss, equivalent to MNRL
30. ⭐⭐ What is symmetric vs asymmetric bi-encoder? — Symmetric: same encoder for both inputs; asymmetric (dual-encoder): different encoders for query and document
31. ⭐⭐ What is DPR? — Dense Passage Retrieval: Facebook's asymmetric dual-encoder for open-domain QA
32. ⭐⭐ What is HNSW? — Hierarchical Navigable Small World: graph-based ANN algorithm, supports incremental insertion
33. ⭐⭐ What is IVF+PQ? — Inverted File Index + Product Quantization: compressed FAISS index for large-scale search
34. ⭐⭐ What is BEIR? — Benchmark for Information Retrieval: zero-shot evaluation across 18 diverse retrieval tasks
35. ⭐ What AWS service provides vector search? — Amazon OpenSearch Service with k-NN plugin
36. ⭐ What Azure service provides vector search? — Azure AI Search with vector search capability
37. ⭐ What is ElastiCache used for in the AWS architecture? — Caching sentence embeddings to avoid recomputation
38. ⭐ What is Azure Cache for Redis used for? — Caching embeddings with model_version-keyed TTL
39. ⭐ What does `normalize_embeddings=True` do? — Applies L2 normalization so cosine similarity = dot product
40. ⭐ What is the output of `model.encode()`? — NumPy array of shape (num_sentences, embedding_dim)
41. ⭐⭐ What is RRF? — Reciprocal Rank Fusion: rank-based score combination for hybrid search
42. ⭐⭐ How does hybrid search combine BM25 and semantic scores? — Normalize both to [0,1], combine with weighted sum or RRF
43. ⭐ What is the STS-B score range? — 0 (completely dissimilar) to 5 (completely equivalent), normalized to 0-1
44. ⭐ What does `save_pretrained` save vs `push_to_hub`? — `save_pretrained`: local disk; `push_to_hub`: HuggingFace Hub
45. ⭐ What is the `sentence_bert_config.json` file? — Specifies pooling strategy and other sentence-transformers configuration
46. ⭐⭐ What is in-batch hard negative augmentation? — Add explicit hard negatives per pair in addition to random in-batch negatives
47. ⭐ What SageMaker instance is recommended for sentence transformer training? — ml.p3dn.24xlarge (8x V100 32GB) for large-batch MNRL
48. ⭐⭐ What is `add_swap_loss` in MNRL? — Adds reversed pairs (b_i, a_i) as additional positives, doubling training signal
49. ⭐ What is L2 normalization and why is it needed for cosine similarity? — Divides each embedding by its L2 norm; enables cosine similarity = dot product = FAISS inner product search
50. ⭐⭐ What is contrastive self-supervised learning? — Learning representations without labels by treating augmented views of the same example as positives and all others as negatives

---

*End of Sentence Embedding Training Interview Guide — 200+ questions covered.*
