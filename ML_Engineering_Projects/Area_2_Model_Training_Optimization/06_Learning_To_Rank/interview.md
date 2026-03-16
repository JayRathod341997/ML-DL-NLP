# Learning to Rank - Interview Preparation Guide

> Comprehensive interview guide for ML Engineers covering learning to rank systems, ranking models, IR metrics, and production ranking pipelines.

---

## Table of Contents

1. [Production-Level Interview Questions](#1-production-level-interview-questions)
2. [System Design Discussions](#2-system-design-discussions)
3. [Common Bugs and Solutions](#3-common-bugs-and-possible-solutions)
4. [Deployment Strategies](#4-deployment-strategies)
   - [Azure Deployment](#azure-deployment)
   - [AWS Deployment](#aws-deployment)
5. [Post-Production Issues](#5-post-production-issues)
6. [Visualizations](#6-visualizations)
7. [General ML/MLOps Topics](#7-general-mlmlops-topics)

---

## 1. Production-Level Interview Questions

### 1.1 Learning to Rank Fundamentals

**Q1: What is Learning to Rank (LTR) and why is it important?**

Learning to Rank (LTR) is a machine learning approach that trains a model to optimize the ordering of items (documents, products, search results) for a given query. Unlike traditional relevance scoring, LTR directly optimizes ranking-specific metrics.

```
┌────────────────────────────────────────────────────────────────────────┐
│                    RANKING VS RELEVANCE SCORING                         │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  RELEVANCE SCORING:                                                   │
│  ────────────────────                                                 │
│  Query: "iPhone price"                                               │
│  Doc1: "iPhone 15 Pro - $999"     → relevance: 0.95               │
│  Doc2: "iPhone charger - $29"      → relevance: 0.80               │
│  Doc3: "Android phone prices"      → relevance: 0.30               │
│                                                                        │
│  RANKING (LTR):                                                       │
│  ─────────────                                                        │
│  Query: "iPhone price"                                                │
│  Doc1: "iPhone 15 Pro - $999"     → rank: 1                        │
│  Doc2: "iPhone charger - $29"      → rank: 2                        │
│  Doc3: "Android phone prices"      → rank: 3                        │
│                                                                        │
│  Key insight: LTR optimizes the ORDER, not just relevance scores!     │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

**Why LTR matters:**
- Directly optimizes business metrics (click-through, conversion)
- Learns complex feature interactions
- Outperforms simple relevance scoring
- Powers search engines, recommender systems, ad ranking

---

**Q2: Explain the three main approaches to Learning to Rank.**

```
┌────────────────────────────────────────────────────────────────────────┐
│                    LTR APPROACHES                                       │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  POINTWISE                                                             │
│  ─────────                                                             │
│  Treats each (query, document) pair independently.                    │
│  Predicts a relevance score for each document.                         │
│                                                                        │
│  Query: "python教程"                                                    │
│  Doc1: "Python tutorial"     → score: 0.9                            │
│  Doc2: "Python book"         → score: 0.7                            │
│  Doc3: "Java tutorial"       → score: 0.2                            │
│                                                                        │
│  Loss: MSE(predicted_score, true_label)                               │
│  Pros: Simple, interpretable                                           │
│  Cons: Ignores document ranking                                         │
│                                                                        │
│  PAIRWISE                                                              │
│  ─────────                                                             │
│  Optimizes pairwise ordering between documents.                        │
│  For each query, learns which doc should rank higher.                  │
│                                                                        │
│  Query: "python教程"                                                    │
│  (Doc1 > Doc2): True    (correct)                                    │
│  (Doc1 > Doc3): True    (correct)                                    │
│  (Doc2 > Doc3): True    (correct)                                    │
│                                                                        │
│  Loss: Hinge loss or Ranking SVM                                      │
│  Pros: Handles label noise better                                     │
│  Cons: More pairs = slower                                            │
│                                                                        │
│  LISTWISE                                                              │
│  ─────────                                                             │
│  Optimizes entire ranking list directly.                               │
│  Uses NDCG, MAP as the loss function.                                  │
│                                                                        │
│  Query: "python教程"                                                    │
│  Rankings: [Doc1, Doc2, Doc3]                                         │
│  Loss: NDCG(ranking, ground_truth)                                    │
│  Pros: Directly optimizes ranking metrics                              │
│  Cons: Complex, harder to optimize                                     │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

**Q3: What are the key evaluation metrics for ranking systems?**

```
┌────────────────────────────────────────────────────────────────────────┐
│                    RANKING EVALUATION METRICS                           │
└────────────────────────────────────────────────────────────────────────┘

1. NDCG (Normalized Discounted Cumulative Gain)
   ──────────────────────────────────────────────
   Gold standard for ranking evaluation
   
   Formula:
   DCG@k = Σ (rel_i / log2(i+1)) for i = 1 to k
   NDCG@k = DCG@k / IDCG@k
   
   Example:
   Query: "best smartphones"
   
   Ideal:    [iPhone, Samsung, OnePlus, Nokia, BlackBerry]
   IDCG@4 = 1/log2(2) + 1/log2(3) + 1/log2(4) + 1/log2(5) = 3.03
   
   Our Rank: [Samsung, iPhone, OnePlus, Nokia]
   DCG@4 = 1/log2(2) + 1/log2(3) + 1/log2(4) + 1/log2(5) = 2.56
   
   NDCG@4 = 2.56 / 3.03 = 0.845

2. MAP (Mean Average Precision)
   ──────────────────────────────
   Average precision across all queries
   
   AP = Σ (Precision@k × rel_k) / total_relevant
   
   Example:
   Query1: Relevant docs at positions 2, 4 → AP = (1/2 + 2/4)/2 = 0.5
   Query2: Relevant doc at position 1 → AP = 1/1 = 1.0
   MAP = (0.5 + 1.0) / 2 = 0.75

3. MRR (Mean Reciprocal Rank)
   ────────────────────────────
   Focuses on first relevant result
   
   Example:
   Query1: First relevant at position 1 → RR = 1/1 = 1.0
   Query2: First relevant at position 3 → RR = 1/3 = 0.33
   Query3: No relevant docs → RR = 0
   MRR = (1.0 + 0.33 + 0) / 3 = 0.44

4. Precision@K and Recall@K
   ──────────────────────────
   Precision@K = relevant_in_top_k / k
   Recall@K = relevant_in_top_k / total_relevant

Visual comparison:
┌────────────────────────────────────────────────────────────────────────┐
│  Metric      │ Best For              │ Sensitive To                    │
├──────────────┼───────────────────────┼────────────────────────────────┤
│  NDCG@k     │ Overall ranking       │ Position of relevant items     │
│  MAP        │ Information Retrieval │ All relevant items            │
│  MRR        │ Question answering    │ First relevant item            │
│  Precision@K│ Search engines        │ Top-K results quality         │
└──────────────┴───────────────────────┴────────────────────────────────┘
```

---

### 1.2 LambdaMART Deep Dive

**Q4: How does LambdaMART work?**

LambdaMART combines gradient boosting with a ranking-specific loss function (Lambda).

```
┌────────────────────────────────────────────────────────────────────────┐
│                    LAMBAMART ALGORITHM                                  │
└────────────────────────────────────────────────────────────────────────┘

KEY INSIGHT: Lambda = gradient of ranking loss (NDCG)

Step 1: Compute Lambda (desired gradient) for each document pair
─────────────────────────────────────────────────────────────

Query: "python tutorial"
Documents: [DocA, DocB, DocC, DocD]
Ground truth relevance: [3, 2, 1, 0]

For each document pair (i, j):
- If rel_i > rel_j: should rank higher
- Lambda_ij = |ΔNDCG(i,j)| = how much swapping affects NDCG

Example calculations:
- DocA(3) vs DocB(2): ΔNDCG = |3-2|/log2(3) ≈ 0.27 → Lambda large
- DocB(2) vs DocC(1): ΔNDCG = |2-1|/log2(3) ≈ 0.21 → Lambda moderate
- DocC(1) vs DocD(0): ΔNDCG = |1-0|/log2(3) ≈ 0.21 → Lambda moderate

Step 2: Aggregate lambdas for each document
─────────────────────────────────────────────────────────────

Lambda_i = Σ Lambda_ij for all j where rel_i > rel_j
          - Σ Lambda_ij for all j where rel_i < rel_j

For DocA (highest relevance):
Lambda_A = +0.27 (wants to go up) + 0.21 + 0.21 = +0.69

For DocD (lowest relevance):
Lambda_D = -0.21 - 0.21 - 0.27 = -0.69

Step 3: Use lambdas as pseudo-labels for gradient boosting
─────────────────────────────────────────────────────────────

Build decision trees to predict these Lambda values.
Each tree reduces the ranking loss.

Step 4: Combine trees (ensemble)
─────────────────────────────────────────────────────────────

Final score = Σ (learning_rate × tree_output)

Visual:
Iteration 1: 
  DocA: +0.69
  DocB: +0.21
  DocC: -0.21
  DocD: -0.69
  
Iteration 2 (after first tree):
  Combined: DocA > DocB > DocC > DocD (improved NDCG!)

Python Implementation:
─────────────────────
import lightgbm as lgb

params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [1, 3, 5, 10],
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 200,
}

train_data = lgb.Dataset(
    X_train, 
    label=y_train,  # relevance scores
    group=query_sizes  # documents per query
)

model = lgb.train(params, train_data)
```

---

**Q5: What features would you use for LambdaMART?**

```
┌────────────────────────────────────────────────────────────────────────┐
│                    LAMBAMART FEATURES                                   │
└────────────────────────────────────────────────────────────────────────┘

COMMON FEATURES:

Text Match Features:
───────────────────
1. BM25 Score           - Classic IR relevance score
2. TF-IDF Cosine       - Term frequency based similarity
3. Query Term Overlap  - Fraction of query terms in document
4. Jaccard Similarity - Set overlap between query and doc

Query Features:
──────────────
5. Query Length        - Number of terms in query
6. Query IDF           - Average IDF of query terms
7. Query Type          - Is it a question? Brand? Category?

Document Features:
─────────────────
8. Document Length     - Number of words in document
9. Title Length       - Length of document title
10. URL Depth         - URL path depth

Popularity Features:
───────────────────
11. Click Count       - Historical clicks
12. Conversion Rate   - Historical conversions
13. Pogo-sticking    - Users who quickly clicked away

Semantic Features:
─────────────────
14. Sentence Embedding Similarity
15. Cross-Encoder Score
16. Word Mover's Distance

Advanced Features:
─────────────────
17. PageRank          - Document importance
18. Site Quality Score
19. Freshness         - How recent is the document

Feature Engineering Tips:
─────────────────────────
- Log transform skewed features (clicks, views)
- Normalize features per query
- Use interaction features (e.g., query_length × doc_length)
- Combine lexical and semantic features
```

---

### 1.3 Neural Ranking Models

**Q6: Compare different neural ranking approaches.**

```
┌────────────────────────────────────────────────────────────────────────┐
│                    NEURAL RANKING MODELS                                │
└────────────────────────────────────────────────────────────────────────┘

BI-ENCODER (Sentence Transformers / DPR)
────────────────────────────────────────
Architecture:
Query → Encoder → q_embedding
Doc   → Encoder → d_embedding
Score = similarity(q_embedding, d_embedding)

Pros:
✓ Fast inference (embed once, compare many)
✓ Good for dense retrieval with FAISS
✓ Easy to scale

Cons:
✗ Loses fine-grained interaction
✗ "Apple the fruit" vs "Apple the company" issue

Example:
from sentence_transformers import CrossEncoder
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
scores = model.predict([(query, doc) for doc in docs])

CROSS-ENCODER
─────────────
Architecture:
[CLS] query [SEP] doc [SEP] → Transformer → score

Pros:
✓ Full query-document interaction
✓ Best accuracy on MS MARCO
✓ Handles complex relevance

Cons:
✗ Slower (must process each pair)
✗ O(n) scoring for n documents

Example:
from sentence_transformers import CrossEncoder
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
scores = model.predict([(query, doc) for doc in docs])

COLBERT (Late Interaction)
──────────────────────────
Architecture:
Query → BERT → query_tokens (per-token embeddings)
Doc   → BERT → doc_tokens (per-token embeddings)
Score = Σ max_sim(query_token, doc_tokens)

Pros:
✓ Balances speed and accuracy
✓ Fine-grained interactions
✓ 10-100x faster than cross-encoder

Cons:
✗ More complex implementation
✗ Requires MaxSim operator

Visual:
┌────────────────────────────────────────────────────────────────────────┐
│  MODEL        │ SPEED    │ ACCURACY │ USE CASE                        │
├───────────────┼──────────┼──────────┼─────────────────────────────────┤
│  Bi-Encoder   │ Fastest  │ Good     │ First-stage retrieval          │
│  Cross-Encoder│ Slow     │ Best     │ Re-ranking top-100             │
│  ColBERT      │ Medium   │ Very Good│ Production two-stage pipeline  │
└───────────────┴──────────┴──────────┴─────────────────────────────────┘
```

---

### 1.4 Two-Stage Ranking Pipeline

**Q7: Design a production two-stage ranking pipeline.**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TWO-STAGE RANKING PIPELINE                               │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────┐
                              │  QUERY   │
                              └────┬─────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: RETRIEVAL (Recall)                                                │
│  ─────────────────────────────────                                         │
│  Goal: Get relevant documents in the candidate set                         │
│  Method: Fast, lower precision                                             │
│  Output: Top 100-1000 candidates                                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                  │
│  │   BM25      │     │ Dense       │     │  Filter    │                  │
│  │  (Sparse)   │  +  │ Retrieval  │  +  │  Rules     │                  │
│  │             │     │ (Bi-encoder)│     │            │                  │
│  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘                  │
│         │                   │                   │                          │
│         └───────────────────┴───────────────────┘                          │
│                             │                                              │
│                             ▼                                              │
│                    Top 100-1000 candidates                                │
└──────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: RERANKING (Precision)                                            │
│  ────────────────────────────────                                         │
│  Goal: Order candidates for best relevance                                 │
│  Method: Slow, higher precision                                            │
│  Output: Top 10-50 results                                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                  │
│  │  Cross-     │     │ LambdaMART  │     │  Learning   │                  │
│  │  Encoder    │  +  │ (GBRT)     │  +  │  to Rank    │                  │
│  │             │     │             │     │  Ensemble   │                  │
│  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘                  │
│         │                   │                   │                          │
│         └───────────────────┴───────────────────┘                          │
│                             │                                              │
│                             ▼                                              │
